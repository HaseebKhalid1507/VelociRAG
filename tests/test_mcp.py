"""Integration tests for Velocirag MCP server via FastMCP Client.

Tests the full MCP protocol flow: Client → MCP protocol → Server → tool dispatch → response.
Same path Claude Desktop, Cursor, and other MCP clients would use.
"""
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Check if fastmcp is available
try:
    from fastmcp.client import Client
    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False

pytestmark = pytest.mark.skipif(not HAS_FASTMCP, reason="fastmcp not installed")


@pytest.fixture(scope="module")
def mcp_server():
    """Create MCP server with a temp database."""
    tmpdir = tempfile.mkdtemp(prefix="velocirag_mcp_test_")
    os.environ['VELOCIRAG_DB'] = tmpdir

    # Reset engine state for fresh init
    from velocirag import mcp_server
    mcp_server._engine.clear()

    return mcp_server.mcp, tmpdir


@pytest.fixture(scope="module")
def test_docs(mcp_server):
    """Create test markdown files for indexing."""
    _, tmpdir = mcp_server
    docs_dir = Path(tmpdir) / "test_docs"
    docs_dir.mkdir()

    (docs_dir / "python.md").write_text(
        "---\ntitle: Python Guide\ntags: [python, programming]\n---\n"
        "# Python Programming\n\n## Basics\nPython is a versatile language.\n\n"
        "## Data Structures\nLists, dicts, sets, and tuples are fundamental.\n"
    )
    (docs_dir / "security.md").write_text(
        "---\ntitle: Network Security\ntags: [security, networking]\n---\n"
        "# Network Security\n\n## Firewalls\nFirewalls filter network traffic.\n\n"
        "## Encryption\nTLS 1.3 provides strong encryption for data in transit.\n"
    )
    (docs_dir / "docker.md").write_text(
        "---\ntitle: Docker Guide\ntags: [docker, containers]\n---\n"
        "# Docker Containers\n\n## Images\nDocker images are layered filesystems.\n\n"
        "## Networking\nDocker uses bridge networks by default.\n"
    )
    return str(docs_dir)


def run_async(coro):
    """Helper to run async tests."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


class TestMCPToolRegistration:
    """Test that all MCP tools are properly registered."""

    def test_all_tools_registered(self, mcp_server):
        server, _ = mcp_server

        async def check():
            async with Client(server) as client:
                tools = await client.list_tools()
                names = {t.name for t in tools}
                assert names == {"search", "index", "add_document", "health", "list_sources"}

        run_async(check())

    def test_tools_have_descriptions(self, mcp_server):
        server, _ = mcp_server

        async def check():
            async with Client(server) as client:
                tools = await client.list_tools()
                for tool in tools:
                    assert tool.description, f"Tool {tool.name} has no description"

        run_async(check())


class TestMCPHealth:
    """Test health check tool."""

    def test_health_returns_structure(self, mcp_server):
        server, _ = mcp_server

        async def check():
            async with Client(server) as client:
                result = await client.call_tool("health", {})
                data = json.loads(result.content[0].text)
                assert "total_documents" in data
                assert "layers" in data
                assert "components" in data
                assert data["components"]["unified_search"] == True

        run_async(check())


class TestMCPIndex:
    """Test indexing tools."""

    def test_index_directory(self, mcp_server, test_docs):
        server, _ = mcp_server

        async def check():
            async with Client(server) as client:
                result = await client.call_tool("index", {
                    "directory": test_docs,
                    "build_graph": False,
                    "extract_metadata": False
                })
                data = json.loads(result.content[0].text)
                assert data["files_processed"] == 3
                assert data["chunks_added"] > 0
                assert "error" not in data

        run_async(check())

    def test_index_nonexistent_directory(self, mcp_server):
        server, _ = mcp_server

        async def check():
            async with Client(server) as client:
                result = await client.call_tool("index", {
                    "directory": "/tmp/nonexistent_xyz_123"
                })
                data = json.loads(result.content[0].text)
                assert "error" in data

        run_async(check())

    def test_add_single_document(self, mcp_server, test_docs):
        server, _ = mcp_server

        async def check():
            async with Client(server) as client:
                # Index first so store is initialized
                md_path = str(Path(test_docs) / "python.md")
                result = await client.call_tool("add_document", {
                    "file_path": md_path
                })
                data = json.loads(result.content[0].text)
                assert data["success"] == True
                assert data["chunks_added"] > 0

        run_async(check())

    def test_add_nonexistent_file(self, mcp_server):
        server, _ = mcp_server

        async def check():
            async with Client(server) as client:
                result = await client.call_tool("add_document", {
                    "file_path": "/tmp/nope_doesnt_exist.md"
                })
                data = json.loads(result.content[0].text)
                assert data["success"] == False
                assert "error" in data

        run_async(check())

    def test_add_non_markdown_file(self, mcp_server):
        server, _ = mcp_server

        async def check():
            async with Client(server) as client:
                with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w') as f:
                    f.write("not markdown")
                    txt_path = f.name
                try:
                    result = await client.call_tool("add_document", {
                        "file_path": txt_path
                    })
                    data = json.loads(result.content[0].text)
                    assert data["success"] == False
                finally:
                    os.unlink(txt_path)

        run_async(check())


class TestMCPSearch:
    """Test search tool (requires index to be built first)."""

    def test_search_returns_results(self, mcp_server, test_docs):
        """Search indexed content."""
        server, _ = mcp_server

        async def check():
            async with Client(server) as client:
                # Ensure indexed
                await client.call_tool("index", {
                    "directory": test_docs,
                    "build_graph": False,
                    "extract_metadata": False
                })

                result = await client.call_tool("search", {
                    "query": "Python programming",
                    "limit": 3
                })
                data = json.loads(result.content[0].text)
                assert data["total_results"] > 0
                assert data["search_time_ms"] > 0
                assert len(data["results"]) <= 3

                # Check result structure
                for r in data["results"]:
                    assert "content" in r
                    assert "score" in r
                    assert "file_path" in r

        run_async(check())

    def test_search_empty_query(self, mcp_server):
        server, _ = mcp_server

        async def check():
            async with Client(server) as client:
                result = await client.call_tool("search", {"query": "", "limit": 3})
                data = json.loads(result.content[0].text)
                assert "error" in data
                assert data["total_results"] == 0

        run_async(check())

    def test_search_limit_clamped(self, mcp_server, test_docs):
        server, _ = mcp_server

        async def check():
            async with Client(server) as client:
                result = await client.call_tool("search", {
                    "query": "data",
                    "limit": 100
                })
                data = json.loads(result.content[0].text)
                assert len(data["results"]) <= 50

        run_async(check())


class TestMCPListSources:
    """Test list_sources tool."""

    def test_list_sources(self, mcp_server, test_docs):
        server, _ = mcp_server

        async def check():
            async with Client(server) as client:
                result = await client.call_tool("list_sources", {"limit": 10})
                data = json.loads(result.content[0].text)
                assert "sources" in data
                assert "total_sources" in data
                assert isinstance(data["sources"], list)

        run_async(check())

    def test_list_sources_respects_limit(self, mcp_server, test_docs):
        server, _ = mcp_server

        async def check():
            async with Client(server) as client:
                result = await client.call_tool("list_sources", {"limit": 2})
                data = json.loads(result.content[0].text)
                assert len(data["sources"]) <= 2

        run_async(check())
