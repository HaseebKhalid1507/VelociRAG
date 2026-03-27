"""
VelociRAG Search Daemon - Keeps embedder + FAISS + search engine warm in memory.

A lightweight daemon that maintains loaded components for instant search responses.
Routes CLI search through Unix socket when available, falls back to cold search.
"""

import os
import sys
import json
import socket
import signal
import struct
import time
import warnings
from threading import Thread
from queue import Queue
from pathlib import Path
from datetime import datetime


# Default config
SOCKET_PATH = os.environ.get('VELOCIRAG_SOCKET', '/tmp/velocirag-daemon.sock')
PID_FILE = '/tmp/velocirag-daemon.pid'


class ResultHolder:
    """Thread-safe container for passing results between worker and connection threads."""
    __slots__ = ('response', 'error', 'ready')
    
    def __init__(self):
        self.response = None
        self.error = None
        self.ready = False


class VelociragDaemon:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._work_queue = Queue(maxsize=100)
        self.running = False
        self.request_count = 0
        self.start_time = None
        
        # Component refs - loaded in worker thread for SQLite thread safety
        self.unified = None
        
    def _worker_loop(self):
        """Worker thread - loads engine, processes queries sequentially."""
        try:
            # Suppress model loading noise
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                from .embedder import Embedder
                from .store import VectorStore
                from .searcher import Searcher
                from .unified import UnifiedSearch
                from .graph import GraphStore
                from .metadata import MetadataStore
                
                # Initialize components in this thread
                embedder = Embedder()
                store = VectorStore(self.db_path, embedder)
                searcher = Searcher(store, embedder)
                
                # Optional components
                graph_store = None
                metadata_store = None
                graph_db = Path(self.db_path) / "graph.db"
                metadata_db = Path(self.db_path) / "metadata.db"
                
                if graph_db.exists():
                    graph_store = GraphStore(str(graph_db))
                if metadata_db.exists():
                    metadata_store = MetadataStore(str(metadata_db))
                
                self.unified = UnifiedSearch(searcher, graph_store, metadata_store)
                
        except Exception as e:
            # Worker failed to load - daemon will be non-functional
            sys.stderr.write(f"Worker thread failed to load: {e}\n")
            return
            
        # Process work queue
        while self.running:
            try:
                holder = self._work_queue.get(timeout=1.0)
                if holder is None:  # Shutdown signal
                    break
                    
                try:
                    request = holder.response  # Request is stored in response initially
                    cmd = request.get("cmd")
                    
                    if cmd == "search":
                        holder.response = self._handle_search(request)
                    elif cmd == "health":
                        holder.response = self._handle_health()
                    elif cmd == "status":
                        holder.response = self._handle_status()
                    elif cmd == "ping":
                        holder.response = {"pong": True}
                    else:
                        holder.response = {"error": f"unknown command: {cmd}"}
                        
                except Exception as e:
                    holder.response = {"error": str(e)}
                    
                finally:
                    holder.ready = True
                    self.request_count += 1
                    
            except Exception:
                continue  # Timeout, keep looping
                
    def _handle_search(self, request):
        """Handle unified search request."""
        query = request.get("query", "")
        limit = request.get("limit", 5)
        threshold = request.get("threshold", 0.3)
        
        if not query:
            return {"error": "query parameter required"}
            
        if not self.unified:
            return {"error": "search engine not loaded"}
            
        try:
            results = self.unified.search(
                query=query,
                limit=limit,
                threshold=threshold,
                enrich_graph=True
            )
            return results
        except Exception as e:
            return {"error": f"search failed: {str(e)}"}
    
    def _handle_health(self):
        """Return comprehensive health info."""
        if not self.unified:
            return {"error": "components not loaded"}
            
        try:
            # Get component stats
            store_stats = self.unified.searcher.store.stats()
            
            health = {
                "status": "ok",
                "uptime_seconds": int((datetime.now() - self.start_time).total_seconds()) if self.start_time else 0,
                "requests_served": self.request_count,
                "total_documents": store_stats.get('document_count', 0),
                "faiss_vectors": store_stats.get('faiss_vectors', 0),
                "consistent": store_stats.get('consistent', False),
                "components": {
                    "unified_search": True,
                    "vector_store": True,
                    "embedder": True,
                    "graph_store": self.unified.graph_store is not None,
                    "metadata_store": self.unified.metadata_store is not None
                }
            }
            
            # Add graph stats if available
            if self.unified.graph_store:
                try:
                    graph_stats = self.unified.graph_store.stats()
                    health["graph_nodes"] = graph_stats.get('nodes', 0)
                    health["graph_edges"] = graph_stats.get('edges', 0)
                except Exception:
                    pass
                    
            return health
            
        except Exception as e:
            return {"error": f"health check failed: {str(e)}"}
    
    def _handle_status(self):
        """Return basic status info."""
        return {
            "running": True,
            "pid": os.getpid(),
            "uptime": int((datetime.now() - self.start_time).total_seconds()) if self.start_time else 0,
            "requests": self.request_count,
            "db_path": self.db_path
        }
    
    def _encode_response(self, data):
        """Encode response with 4-byte length prefix."""
        payload = json.dumps(data, default=str).encode('utf-8')
        return struct.pack('>I', len(payload)) + payload
    
    def _read_message(self, conn):
        """Read length-prefixed JSON message."""
        # Read 4-byte length
        raw_len = conn.recv(4)
        if not raw_len or len(raw_len) < 4:
            return None
            
        msg_len = struct.unpack('>I', raw_len)[0]
        if msg_len > 1_000_000:  # 1MB limit
            return None
            
        # Read message
        data = b""
        while len(data) < msg_len:
            chunk = conn.recv(min(msg_len - len(data), 65536))
            if not chunk:
                return None
            data += chunk
            
        return json.loads(data.decode('utf-8'))
    
    def handle_client(self, conn):
        """Handle a client connection."""
        try:
            conn.settimeout(30.0)
            
            request = self._read_message(conn)
            if not request:
                return
                
            # Submit to worker queue (bounded — rejects if overloaded)
            holder = ResultHolder()
            holder.response = request  # Store request in holder initially
            try:
                self._work_queue.put(holder, timeout=5.0)
            except Exception:
                conn.sendall(self._encode_response({"error": "Server overloaded, try again later"}))
                return
            
            # Wait for worker to process
            timeout = 30.0
            start = time.time()
            while not holder.ready and (time.time() - start) < timeout:
                time.sleep(0.01)
                
            if holder.ready:
                conn.sendall(self._encode_response(holder.response))
            else:
                conn.sendall(self._encode_response({"error": "query timed out"}))
                
        except Exception as e:
            try:
                conn.sendall(self._encode_response({"error": str(e)}))
            except Exception:
                pass
        finally:
            conn.close()
    
    def start(self, foreground=False):
        """Start the daemon."""
        # Check if already running
        if os.path.exists(SOCKET_PATH):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(SOCKET_PATH)
                s.close()
                print("Daemon already running")
                return
            except Exception:
                os.unlink(SOCKET_PATH)
        
        if not foreground:
            # Daemonize
            pid = os.fork()
            if pid > 0:
                print(f"VelociRAG daemon started (PID {pid})")
                return
                
            os.setsid()
            pid = os.fork()
            if pid > 0:
                os._exit(0)
                
            # Redirect stdin/stdout/stderr
            devnull = open(os.devnull, 'r+')
            os.dup2(devnull.fileno(), sys.stdin.fileno())
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            devnull.close()
        
        # Start serving
        self._serve()
        
    def _serve(self):
        """Main server loop."""
        # Check for stale PID file
        if os.path.exists(PID_FILE):
            try:
                with open(PID_FILE) as f:
                    old_pid = int(f.read().strip())
                os.kill(old_pid, 0)  # Check if alive
                # If we get here, old daemon is still running
            except (ProcessLookupError, ValueError, FileNotFoundError):
                os.unlink(PID_FILE)  # Stale PID, remove it
        
        # Write PID file
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
            
        # Create socket
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
            
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        os.chmod(SOCKET_PATH, 0o600)  # Owner only — set before listen() to close race window
        server.listen(8)
        server.settimeout(1.0)
        
        self.running = True
        self.start_time = datetime.now()
        
        def shutdown(signum, frame):
            self.running = False
            
        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        
        # Start worker thread
        worker = Thread(target=self._worker_loop, daemon=True)
        worker.start()
        
        print(f"VelociRAG daemon ready (PID {os.getpid()})")
        print(f"Socket: {SOCKET_PATH}")
        
        # Accept connections
        while self.running:
            try:
                conn, _ = server.accept()
                Thread(target=self.handle_client, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    sys.stderr.write(f"Accept error: {e}\n")
                    
        # Cleanup
        self._work_queue.put(None)  # Signal worker to stop
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
        if os.path.exists(PID_FILE):
            os.unlink(PID_FILE)
    
    @staticmethod        
    def stop_daemon():
        """Stop the daemon by reading PID and sending SIGTERM."""
        if not os.path.exists(PID_FILE):
            print("Daemon not running")
            return
            
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
            
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"VelociRAG daemon stopped (PID {pid})")
        except ProcessLookupError:
            print("Daemon not running (stale PID)")
            os.unlink(PID_FILE)


# Client functions for CLI integration

def _recv_response(sock) -> bytes | None:
    """Read a length-prefixed response from a daemon socket. Returns None on failure."""
    raw_len = sock.recv(4)
    if not raw_len or len(raw_len) < 4:
        return None
    msg_len = struct.unpack('>I', raw_len)[0]
    response_data = b""
    while len(response_data) < msg_len:
        chunk = sock.recv(min(msg_len - len(response_data), 65536))
        if not chunk:
            return None
        response_data += chunk
    return response_data


def daemon_search(query: str, limit: int = 5, threshold: float = 0.3,
                  socket_path: str = SOCKET_PATH) -> dict | None:
    """Query the daemon. Returns None if daemon not running."""
    sock = None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect(socket_path)

        request = {
            "cmd": "search",
            "query": query,
            "limit": limit,
            "threshold": threshold
        }

        data = json.dumps(request).encode('utf-8')
        sock.sendall(struct.pack('>I', len(data)) + data)

        response_data = _recv_response(sock)
        if not response_data:
            return None

        return json.loads(response_data.decode('utf-8'))

    except (ConnectionRefusedError, FileNotFoundError):
        return None  # Daemon not running
    finally:
        if sock:
            sock.close()


def daemon_health(socket_path: str = SOCKET_PATH) -> dict | None:
    """Get daemon health info."""
    sock = None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(socket_path)

        request = {"cmd": "health"}
        data = json.dumps(request).encode('utf-8')
        sock.sendall(struct.pack('>I', len(data)) + data)

        response_data = _recv_response(sock)
        if not response_data:
            return None

        return json.loads(response_data.decode('utf-8'))

    except (ConnectionRefusedError, FileNotFoundError):
        return None
    finally:
        if sock:
            sock.close()


def daemon_ping(socket_path: str = SOCKET_PATH) -> bool:
    """Test if daemon is responsive."""
    sock = None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect(socket_path)

        request = {"cmd": "ping"}
        data = json.dumps(request).encode('utf-8')
        sock.sendall(struct.pack('>I', len(data)) + data)

        response_data = _recv_response(sock)
        if not response_data:
            return False

        response = json.loads(response_data.decode('utf-8'))
        return response.get("pong") is True

    except Exception:
        return False
    finally:
        if sock:
            sock.close()