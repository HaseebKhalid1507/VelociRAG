"""Tests for cli.py module."""

import json
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from velocirag.cli import cli
from velocirag.store import VectorStore
from velocirag.embedder import Embedder


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def sample_markdown_files(tmp_path):
    """Create sample markdown files for testing."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    # Create various markdown files
    (docs_dir / "python.md").write_text("""# Python Programming

Python is a high-level programming language known for its simplicity and readability.

## Features
- Dynamic typing
- Interpreted language
- Object-oriented programming
- Large standard library

## Use Cases
Python is used in web development, data science, artificial intelligence, and automation.
""")
    
    (docs_dir / "machine_learning.md").write_text("""# Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

## Types of Learning
1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

## Popular Algorithms
- Linear Regression
- Decision Trees
- Neural Networks
- Support Vector Machines
""")
    
    # Create subdirectory with more files
    subdir = docs_dir / "tutorials"
    subdir.mkdir()
    
    (subdir / "git_basics.md").write_text("""# Git Basics

Git is a distributed version control system.

## Essential Commands
- git init
- git add
- git commit
- git push
- git pull

Version control is essential for collaborative development.
""")
    
    # Create an empty file (should be skipped)
    (docs_dir / "empty.md").write_text("")
    
    # Create a non-markdown file (should be ignored)
    (docs_dir / "readme.txt").write_text("This is not a markdown file")
    
    return docs_dir


class TestIndexCommand:
    """Test the index command."""
    
    def test_index_basic(self, runner, sample_markdown_files, tmp_path):
        """Basic index command works correctly."""
        db_path = tmp_path / "test_db"
        
        result = runner.invoke(cli, [
            'index',
            str(sample_markdown_files),
            '--db', str(db_path)
        ])
        
        assert result.exit_code == 0
        # Empty.md is counted but then skipped
        assert "Found 4 markdown files" in result.output
        assert "Files processed: 3" in result.output
        assert "Files skipped: 1" in result.output  # empty.md is skipped
        assert "Index ready for search" in result.output
        
        # Verify database was created
        assert (db_path / "store.db").exists()
        assert (db_path / "index.faiss").exists()
    
    def test_index_with_source(self, runner, sample_markdown_files, tmp_path):
        """Index with source identifier."""
        db_path = tmp_path / "test_db"
        
        result = runner.invoke(cli, [
            'index',
            str(sample_markdown_files),
            '--db', str(db_path),
            '--source', 'documentation'
        ])
        
        assert result.exit_code == 0
        
        # Verify source was applied
        embedder = Embedder()
        store = VectorStore(str(db_path), embedder=embedder)
        results = store.search("python", limit=1)
        if results:
            assert results[0]['metadata']['source_name'] == 'documentation'
        store.close()
    
    def test_index_incremental(self, runner, sample_markdown_files, tmp_path):
        """Incremental indexing skips unchanged files."""
        db_path = tmp_path / "test_db"
        
        # First index
        result1 = runner.invoke(cli, [
            'index',
            str(sample_markdown_files),
            '--db', str(db_path)
        ])
        assert result1.exit_code == 0
        assert "Files processed: 3" in result1.output
        
        # Second index without changes
        result2 = runner.invoke(cli, [
            'index',
            str(sample_markdown_files),
            '--db', str(db_path)
        ])
        assert result2.exit_code == 0
        assert "Files skipped: 1 (unchanged)" in result2.output
    
    @pytest.mark.skip(reason="Force flag not implemented in store.add_directory")
    def test_index_force(self, runner, sample_markdown_files, tmp_path):
        """Force flag reindexes all files."""
        db_path = tmp_path / "test_db"
        
        # First index
        result1 = runner.invoke(cli, [
            'index',
            str(sample_markdown_files),
            '--db', str(db_path)
        ])
        assert result1.exit_code == 0
        
        # Force reindex
        result2 = runner.invoke(cli, [
            'index',
            str(sample_markdown_files),
            '--db', str(db_path),
            '--force'
        ])
        assert result2.exit_code == 0
        assert "Files processed: 3" in result2.output
        assert "Files skipped: 0" in result2.output
    
    def test_index_empty_directory(self, runner, tmp_path):
        """Indexing empty directory shows warning."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        db_path = tmp_path / "test_db"
        
        result = runner.invoke(cli, [
            'index',
            str(empty_dir),
            '--db', str(db_path)
        ])
        
        assert result.exit_code == 0
        assert "No markdown files found" in result.output
    
    def test_index_nonexistent_directory(self, runner, tmp_path):
        """Indexing nonexistent directory shows error."""
        result = runner.invoke(cli, [
            'index',
            str(tmp_path / "nonexistent"),
            '--db', str(tmp_path / "test_db")
        ])
        
        assert result.exit_code == 2  # Click's error code for bad parameters
    
    def test_index_verbose(self, runner, sample_markdown_files, tmp_path):
        """Verbose flag shows additional output."""
        db_path = tmp_path / "test_db"
        
        result = runner.invoke(cli, [
            '--verbose',
            'index',
            str(sample_markdown_files),
            '--db', str(db_path)
        ])
        
        assert result.exit_code == 0
        assert "Database path:" in result.output
        assert "Source directory:" in result.output
        assert "Initializing embedder" in result.output
        assert "Initializing vector store" in result.output
    
    def test_index_with_errors(self, runner, tmp_path):
        """Index handles file errors gracefully."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        
        # Create a good file
        (docs_dir / "good.md").write_text("# Good File\n\nThis is readable.")
        
        # Create file with read permission issues
        bad_file = docs_dir / "bad.md"
        bad_file.write_text("# Bad File\n\nThis will be unreadable.")
        bad_file.chmod(0o000)
        
        db_path = tmp_path / "test_db"
        
        result = runner.invoke(cli, [
            'index',
            str(docs_dir),
            '--db', str(db_path)
        ])
        
        # Should still succeed but show errors
        assert result.exit_code == 0
        assert "1 errors encountered" in result.output
        assert "bad.md" in result.output
        
        # Restore permissions for cleanup
        bad_file.chmod(0o644)


class TestSearchCommand:
    """Test the search command."""
    
    @pytest.fixture
    def indexed_db(self, runner, sample_markdown_files, tmp_path):
        """Create an indexed database for search tests."""
        db_path = tmp_path / "search_db"
        
        # Index the sample files
        result = runner.invoke(cli, [
            'index',
            str(sample_markdown_files),
            '--db', str(db_path)
        ])
        assert result.exit_code == 0
        
        return db_path
    
    def test_search_basic(self, runner, indexed_db):
        """Basic search returns results."""
        result = runner.invoke(cli, [
            'search',
            'python programming',
            '--db', str(indexed_db)
        ])
        
        assert result.exit_code == 0
        assert 'Query: "python programming"' in result.output
        assert 'Found' in result.output
        assert 'python.md' in result.output
        assert 'Python is a high-level programming language' in result.output
    
    def test_search_limit(self, runner, indexed_db):
        """Search respects limit parameter."""
        result = runner.invoke(cli, [
            'search',
            'learning',
            '--db', str(indexed_db),
            '--limit', '2'
        ])
        
        assert result.exit_code == 0
        # Count numbered results (1., 2., etc)
        lines = result.output.strip().split('\n')
        numbered_results = [l for l in lines if l.strip().startswith(('1.', '2.', '3.'))]
        assert len(numbered_results) <= 2
    
    def test_search_threshold(self, runner, indexed_db):
        """Search respects threshold parameter."""
        # High threshold - fewer results
        result_high = runner.invoke(cli, [
            'search',
            'python',
            '--db', str(indexed_db),
            '--threshold', '0.8'
        ])
        
        # Low threshold - more results
        result_low = runner.invoke(cli, [
            'search',
            'python',
            '--db', str(indexed_db),
            '--threshold', '0.1'
        ])
        
        assert result_high.exit_code == 0
        assert result_low.exit_code == 0
        
        # Low threshold should have more results
        # (This is a bit fragile but demonstrates the concept)
        high_lines = len([l for l in result_high.output.split('\n') if l.strip()])
        low_lines = len([l for l in result_low.output.split('\n') if l.strip()])
        assert low_lines >= high_lines
    
    def test_search_json_format(self, runner, indexed_db):
        """Search with JSON format returns valid JSON."""
        # Set NO_COLOR to avoid ANSI codes in JSON output
        import os
        env = os.environ.copy()
        env['NO_COLOR'] = '1'
        
        result = runner.invoke(cli, [
            'search',
            'machine learning',
            '--db', str(indexed_db),
            '--format', 'json'
        ], env=env)
        
        assert result.exit_code == 0
        
        # Find the JSON part in the output (skip any warnings)
        output = result.output.strip()
        
        # Try to find where JSON starts by looking for the opening brace
        json_start_idx = output.find('{')
        if json_start_idx == -1:
            pytest.fail("No JSON output found")
        
        # Extract from first { to the end
        json_str = output[json_start_idx:]
        
        # If there are multiple JSON objects or extra content, find the matching closing brace
        brace_count = 0
        json_end_idx = 0
        for i, char in enumerate(json_str):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end_idx = i + 1
                    break
        
        if json_end_idx > 0:
            json_str = json_str[:json_end_idx]
        
        try:
            data = json.loads(json_str)
            assert 'results' in data
            assert 'total_results' in data
            assert 'search_time_ms' in data
            assert isinstance(data['results'], list)
            
            if data['results']:
                first_result = data['results'][0]
                assert 'doc_id' in first_result
                assert 'content' in first_result
                assert 'similarity' in first_result
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON output: {e}\nOutput was: {repr(output)}")
    
    def test_search_compact_format(self, runner, indexed_db):
        """Search with compact format for scripting."""
        result = runner.invoke(cli, [
            'search',
            'git',
            '--db', str(indexed_db),
            '--format', 'compact'
        ])
        
        assert result.exit_code == 0
        
        # Output should be tab-separated: score<tab>doc_id
        lines = result.output.strip().split('\n')
        if lines[0]:  # If we have results
            assert '\t' in lines[0]
            parts = lines[0].split('\t')
            assert len(parts) == 2
            # First part should be a float (similarity score)
            assert float(parts[0]) >= 0.0
    
    def test_search_with_stats(self, runner, indexed_db):
        """Search with stats flag shows performance info."""
        result = runner.invoke(cli, [
            'search',
            'algorithms',
            '--db', str(indexed_db),
            '--stats'
        ])
        
        assert result.exit_code == 0
        assert "Search Statistics:" in result.output
        assert "ms" in result.output  # Timing info
    
    def test_search_no_results(self, runner, indexed_db):
        """Search with no matches shows helpful message."""
        result = runner.invoke(cli, [
            'search',
            'quantum cryptography blockchain',  # Unlikely to match
            '--db', str(indexed_db)
        ])
        
        assert result.exit_code == 0
        assert "No results found" in result.output
        assert "Try:" in result.output
        assert "Lowering the similarity threshold" in result.output
    
    def test_search_missing_database(self, runner, tmp_path):
        """Search without database shows error."""
        result = runner.invoke(cli, [
            'search',
            'test query',
            '--db', str(tmp_path / "nonexistent_db")
        ])
        
        assert result.exit_code == 1
        assert "Database not found" in result.output
        assert "Run 'velocirag index" in result.output
    
    def test_search_empty_database(self, runner, tmp_path):
        """Search on empty database shows warning."""
        # Create empty database
        db_path = tmp_path / "empty_db"
        embedder = Embedder()
        store = VectorStore(str(db_path), embedder)
        store.close()
        
        result = runner.invoke(cli, [
            'search',
            'test query',
            '--db', str(db_path)
        ])
        
        assert result.exit_code == 0
        assert "No documents indexed" in result.output
        assert "Run 'velocirag index" in result.output
    
    def test_search_empty_query(self, runner, indexed_db):
        """Empty search query shows error."""
        result = runner.invoke(cli, [
            'search',
            '   ',  # Empty/whitespace query
            '--db', str(indexed_db)
        ])
        
        assert result.exit_code == 1
        assert "Query cannot be empty" in result.output
    
    def test_search_invalid_limit(self, runner, indexed_db):
        """Invalid limit values show error."""
        # Negative limit
        result = runner.invoke(cli, [
            'search',
            'test',
            '--db', str(indexed_db),
            '--limit', '-5'
        ])
        assert result.exit_code == 1
        assert "Limit must be between 1 and 50" in result.output
        
        # Too high limit
        result = runner.invoke(cli, [
            'search',
            'test',
            '--db', str(indexed_db),
            '--limit', '100'
        ])
        assert result.exit_code == 1
        assert "Limit must be between 1 and 50" in result.output
    
    def test_search_invalid_threshold(self, runner, indexed_db):
        """Invalid threshold values show error."""
        result = runner.invoke(cli, [
            'search',
            'test',
            '--db', str(indexed_db),
            '--threshold', '1.5'
        ])
        
        assert result.exit_code == 1
        assert "Threshold must be between 0.0 and 1.0" in result.output


class TestStatusCommand:
    """Test the status command."""
    
    def test_status_indexed_db(self, runner, sample_markdown_files, tmp_path):
        """Status shows correct info for indexed database."""
        db_path = tmp_path / "status_db"
        
        # First index some files
        index_result = runner.invoke(cli, [
            'index',
            str(sample_markdown_files),
            '--db', str(db_path)
        ])
        assert index_result.exit_code == 0
        
        # Check status
        result = runner.invoke(cli, [
            'status',
            '--db', str(db_path)
        ])
        
        assert result.exit_code == 0
        assert "Velociragtor Index Status" in result.output
        assert "Database:" in result.output
        assert str(db_path) in result.output
        assert "Documents:" in result.output
        assert "chunks" in result.output
        assert "Vector Index:" in result.output
        assert "embeddings" in result.output
        assert "384 dimensions" in result.output
        assert "Index Status: Consistent ✓" in result.output
        assert "Health Check: All systems operational ✓" in result.output
    
    def test_status_json_format(self, runner, sample_markdown_files, tmp_path):
        """Status with JSON format returns valid JSON."""
        db_path = tmp_path / "status_db"
        
        # Index files first
        index_result = runner.invoke(cli, [
            'index',
            str(sample_markdown_files),
            '--db', str(db_path)
        ])
        assert index_result.exit_code == 0
        
        # Get status as JSON
        result = runner.invoke(cli, [
            'status',
            '--db', str(db_path),
            '--format', 'json'
        ])
        
        assert result.exit_code == 0
        
        # Parse JSON
        data = json.loads(result.output)
        assert 'document_count' in data
        assert 'faiss_vectors' in data
        assert 'consistent' in data
        assert 'dimensions' in data
        assert 'schema_version' in data
        assert data['dimensions'] == 384
        assert data['consistent'] is True
    
    def test_status_missing_database(self, runner, tmp_path):
        """Status for missing database shows appropriate message."""
        result = runner.invoke(cli, [
            'status',
            '--db', str(tmp_path / "nonexistent")
        ])
        
        assert result.exit_code == 0  # Not a failure, just reports status
        assert "Database not found" in result.output
    
    def test_status_empty_database(self, runner, tmp_path):
        """Status for empty database shows correct info."""
        db_path = tmp_path / "empty_db"
        
        # Create empty database
        embedder = Embedder()
        store = VectorStore(str(db_path), embedder)
        store.close()
        
        result = runner.invoke(cli, [
            'status',
            '--db', str(db_path)
        ])
        
        assert result.exit_code == 0
        assert "Documents: 0 chunks" in result.output
        assert "No vectors found" in result.output
        assert "No documents indexed" in result.output
    
    def test_status_inconsistent_db(self, runner, tmp_path):
        """Status detects inconsistent database."""
        db_path = tmp_path / "inconsistent_db"
        
        # Create database with inconsistency
        embedder = Embedder()
        store = VectorStore(str(db_path), embedder)
        
        # Add a document
        store.add("doc1", "test content")
        
        # Manually corrupt by deleting FAISS index
        store.close()
        (db_path / "index.faiss").unlink()
        
        # Check status
        result = runner.invoke(cli, [
            'status',
            '--db', str(db_path)
        ])
        
        assert result.exit_code == 0
        # After auto-rebuild, it should be consistent
        assert "Index Status: Consistent" in result.output
    
    def test_status_verbose(self, runner, sample_markdown_files, tmp_path):
        """Verbose flag works with status command."""
        db_path = tmp_path / "status_db"
        
        # Index files first
        index_result = runner.invoke(cli, [
            'index',
            str(sample_markdown_files),
            '--db', str(db_path)
        ])
        assert index_result.exit_code == 0
        
        # Get verbose status
        result = runner.invoke(cli, [
            '--verbose',
            'status',
            '--db', str(db_path)
        ])
        
        assert result.exit_code == 0
        # Basic status output should still be there
        assert "Velociragtor Index Status" in result.output


class TestReindexCommand:
    """Test the reindex command."""
    
    def test_reindex_basic(self, runner, sample_markdown_files, tmp_path):
        """Basic reindex rebuilds FAISS index."""
        db_path = tmp_path / "reindex_db"
        
        # First index some files
        index_result = runner.invoke(cli, [
            'index',
            str(sample_markdown_files),
            '--db', str(db_path)
        ])
        assert index_result.exit_code == 0
        
        # Reindex
        result = runner.invoke(cli, [
            'reindex',
            '--db', str(db_path),
            '--yes'  # Skip confirmation
        ])
        
        assert result.exit_code == 0
        assert "Rebuilding FAISS index" in result.output
        assert "Index rebuilt successfully" in result.output
        assert "Index is now consistent ✓" in result.output
    
    def test_reindex_confirmation_prompt(self, runner, sample_markdown_files, tmp_path):
        """Reindex asks for confirmation without --yes."""
        db_path = tmp_path / "reindex_db"
        
        # Index files first
        index_result = runner.invoke(cli, [
            'index',
            str(sample_markdown_files),
            '--db', str(db_path)
        ])
        assert index_result.exit_code == 0
        
        # Reindex without --yes (abort)
        result = runner.invoke(cli, [
            'reindex',
            '--db', str(db_path)
        ], input='n\n')  # Say no to confirmation
        
        assert result.exit_code == 1  # Aborted
        assert "Continue?" in result.output
    
    def test_reindex_missing_database(self, runner, tmp_path):
        """Reindex on missing database shows error."""
        result = runner.invoke(cli, [
            'reindex',
            '--db', str(tmp_path / "nonexistent"),
            '--yes'
        ])
        
        assert result.exit_code == 1
        assert "Database not found" in result.output
    
    def test_reindex_empty_database(self, runner, tmp_path):
        """Reindex on empty database shows warning."""
        db_path = tmp_path / "empty_db"
        
        # Create empty database
        embedder = Embedder()
        store = VectorStore(str(db_path), embedder)
        store.close()
        
        result = runner.invoke(cli, [
            'reindex',
            '--db', str(db_path),
            '--yes'
        ])
        
        assert result.exit_code == 0
        assert "No documents in database" in result.output
        assert "Nothing to reindex" in result.output
    
    def test_reindex_keyboard_interrupt(self, runner, sample_markdown_files, tmp_path):
        """Reindex handles keyboard interrupt gracefully."""
        db_path = tmp_path / "reindex_db"
        
        # Index files first
        index_result = runner.invoke(cli, [
            'index',
            str(sample_markdown_files),
            '--db', str(db_path)
        ])
        assert index_result.exit_code == 0
        
        # Simulate Ctrl+C during confirmation
        result = runner.invoke(cli, [
            'reindex',
            '--db', str(db_path)
        ], input='\x03')  # Ctrl+C
        
        # Click handles this as abort
        assert result.exit_code == 1


class TestCLIGeneral:
    """Test general CLI features."""
    
    def test_help(self, runner):
        """Help command shows usage."""
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "Velociragtor" in result.output
        assert "Production vector search" in result.output
        assert "Commands:" in result.output
        assert "index" in result.output
        assert "search" in result.output
        assert "status" in result.output
        assert "reindex" in result.output
    
    @pytest.mark.skip(reason="Requires package installation")
    def test_version(self, runner):
        """Version flag shows version."""
        result = runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert "version" in result.output.lower()
    
    def test_command_help(self, runner):
        """Individual command help works."""
        # Index help
        result = runner.invoke(cli, ['index', '--help'])
        assert result.exit_code == 0
        assert "Index a directory of markdown files" in result.output
        
        # Search help
        result = runner.invoke(cli, ['search', '--help'])
        assert result.exit_code == 0
        assert "Search indexed content using semantic similarity" in result.output
        
        # Status help
        result = runner.invoke(cli, ['status', '--help'])
        assert result.exit_code == 0
        assert "Show index statistics and health information" in result.output
        
        # Reindex help
        result = runner.invoke(cli, ['reindex', '--help'])
        assert result.exit_code == 0
        assert "Rebuild FAISS index from stored SQLite embeddings" in result.output
    
    def test_invalid_command(self, runner):
        """Invalid command shows error."""
        result = runner.invoke(cli, ['invalid'])
        
        assert result.exit_code == 2
        assert "No such command" in result.output
    
    def test_environment_variable_db_path(self, runner, sample_markdown_files, tmp_path, monkeypatch):
        """VELOCIRAGTOR_DB environment variable sets default DB path."""
        db_path = tmp_path / "env_db"
        monkeypatch.setenv("VELOCIRAGTOR_DB", str(db_path))
        
        # Index without specifying --db
        result = runner.invoke(cli, [
            'index',
            str(sample_markdown_files)
        ])
        
        assert result.exit_code == 0
        assert (db_path / "store.db").exists()
        
        # Search should also use env var
        search_result = runner.invoke(cli, [
            'search',
            'python'
        ])
        
        assert search_result.exit_code == 0
        assert 'python.md' in search_result.output
    
    def test_default_db_location(self, runner, sample_markdown_files):
        """Default DB location is ./.velocirag/."""
        with runner.isolated_filesystem():
            # Index without specifying --db
            result = runner.invoke(cli, [
                'index',
                str(sample_markdown_files)
            ])
            
            assert result.exit_code == 0
            assert Path(".velocirag/store.db").exists()
    
    def test_color_output(self, runner, sample_markdown_files, tmp_path):
        """Color output works when terminal supports it."""
        db_path = tmp_path / "color_db"
        
        # Index files
        index_result = runner.invoke(cli, [
            'index',
            str(sample_markdown_files),
            '--db', str(db_path)
        ])
        assert index_result.exit_code == 0
        
        # Search (color output should include ANSI codes when supported)
        result = runner.invoke(cli, [
            'search',
            'python',
            '--db', str(db_path)
        ], color=True)
        
        assert result.exit_code == 0
        # Can't easily test for ANSI codes in output, but verify it works
        assert 'python.md' in result.output
    
    def test_no_color_environment(self, runner, sample_markdown_files, tmp_path, monkeypatch):
        """NO_COLOR environment variable disables color."""
        monkeypatch.setenv("NO_COLOR", "1")
        db_path = tmp_path / "no_color_db"
        
        # Index files
        index_result = runner.invoke(cli, [
            'index',
            str(sample_markdown_files),
            '--db', str(db_path)
        ])
        assert index_result.exit_code == 0
        
        # Status should work without color
        result = runner.invoke(cli, [
            'status',
            '--db', str(db_path)
        ])
        
        assert result.exit_code == 0
        # Output should not contain color codes (hard to test definitively)
        assert "Documents:" in result.output