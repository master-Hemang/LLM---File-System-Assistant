#!/usr/bin/env python3
"""
MCP File System Server  JSON RPC 2.0 compliant server exposing file operations.
Implements watch_directory() and batch_process() as MCP resources.
"""

import os
import json
import time
import threading
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Use same file tools from Milestone 1
from fs_tools import FileSystemTools

# ============================================================================
# MCP Server Core
# ============================================================================

class MCPServer:
    """Model Context Protocol server – JSON-RPC 2.0 over HTTP."""
    
    def __init__(self, resumes_dir: str = "./sample_data"):
        self.resumes_dir = Path(resumes_dir).absolute()
        self.resumes_dir.mkdir(exist_ok=True)
        self.fs = FileSystemTools()
        self.watchers = {}  # path -> last_modified timestamp
        self._watch_thread = None
        self._stop_watch = False
        
        # Register available methods
        self.methods = {
            # Core file tools
            "read_file": self.read_file,
            "list_files": self.list_files,
            "write_file": self.write_file,
            "search_in_file": self.search_in_file,
            # New MCP capabilities
            "watch_directory": self.watch_directory,
            "batch_process": self.batch_process,
            "get_file_info": self.get_file_info,
            # Resource discovery
            "list_resources": self.list_resources,
            "get_resource": self.get_resource,
        }
    
    # ------------------------------------------------------------------------
    # Core File Methods (from Milestone 1, adapted to MCP)
    # ------------------------------------------------------------------------
    
    def read_file(self, params: Dict) -> Dict:
        """Read a file and return content with metadata."""
        filepath = params.get("filepath")
        if not filepath:
            return self._error("Missing 'filepath' parameter", -32602)
        result = self.fs.read_file(filepath)
        if result["success"]:
            return {
                "content": result["content"],
                "metadata": result["metadata"]
            }
        return self._error(result["error"], -32000)
    
    def list_files(self, params: Dict) -> Dict:
        """List files in a directory, optionally filtered by extension."""
        directory = params.get("directory", str(self.resumes_dir))
        extension = params.get("extension")
        result = self.fs.list_files(directory, extension)
        if result["success"]:
            return {
                "files": result["files"],
                "count": result["count"],
                "directory": result["directory"]
            }
        return self._error(result["error"], -32000)
    
    def write_file(self, params: Dict) -> Dict:
        """Write content to a file."""
        filepath = params.get("filepath")
        content = params.get("content")
        if not filepath or content is None:
            return self._error("Missing 'filepath' or 'content'", -32602)
        result = self.fs.write_file(filepath, content)
        if result["success"]:
            return {"message": result["message"], "filepath": result["filepath"]}
        return self._error(result["error"], -32000)
    
    def search_in_file(self, params: Dict) -> Dict:
        """Search for keyword in a file with context."""
        filepath = params.get("filepath")
        keyword = params.get("keyword")
        if not filepath or not keyword:
            return self._error("Missing 'filepath' or 'keyword'", -32602)
        result = self.fs.search_in_file(filepath, keyword)
        if result["success"]:
            return {
                "matches": result["matches"],
                "count": result["count"]
            }
        return self._error(result["error"], -32000)
    
    # ------------------------------------------------------------------------
    # New MCP Capabilities
    # ------------------------------------------------------------------------
    
    def watch_directory(self, params: Dict) -> Dict:
        """
        Start monitoring a directory for new/changed files.
        Returns a watch ID that can be used to stop watching.
        """
        directory = params.get("directory", str(self.resumes_dir))
        callback_url = params.get("callback_url")  # optional webhook
        interval = params.get("interval_seconds", 5)
        
        watch_id = hashlib.md5(f"{directory}_{time.time()}".encode()).hexdigest()[:8]
        
        # Store initial state
        current_files = {}
        for f in Path(directory).iterdir():
            if f.is_file():
                current_files[f.name] = f.stat().st_mtime
        
        self.watchers[watch_id] = {
            "directory": directory,
            "callback_url": callback_url,
            "interval": interval,
            "last_state": current_files,
            "running": True
        }
        
        # Start background thread if not already running
        if not self._watch_thread or not self._watch_thread.is_alive():
            self._start_watch_thread()
        
        return {
            "watch_id": watch_id,
            "message": f"Watching {directory} (every {interval}s)",
            "status": "started"
        }
    
    def _start_watch_thread(self):
        """Background thread that checks for changes in watched directories."""
        def watcher_loop():
            while not self._stop_watch:
                for watch_id, config in list(self.watchers.items()):
                    if not config.get("running", True):
                        continue
                    directory = config["directory"]
                    last_state = config["last_state"]
                    changes = {"added": [], "modified": [], "removed": []}
                    
                    # Current state
                    current = {}
                    for f in Path(directory).iterdir():
                        if f.is_file():
                            current[f.name] = f.stat().st_mtime
                    
                    # Detect added/modified
                    for name, mtime in current.items():
                        if name not in last_state:
                            changes["added"].append(name)
                        elif mtime != last_state[name]:
                            changes["modified"].append(name)
                    
                    # Detect removed
                    for name in last_state:
                        if name not in current:
                            changes["removed"].append(name)
                    
                    if any(changes.values()):
                        # Update stored state
                        self.watchers[watch_id]["last_state"] = current
                        # If callback provided, send notification (mock)
                        if config.get("callback_url"):
                            self._send_notification(config["callback_url"], {
                                "watch_id": watch_id,
                                "directory": directory,
                                "changes": changes,
                                "timestamp": datetime.now().isoformat()
                            })
                        # For simplicity, we store the last change for resource query
                        self.watchers[watch_id]["last_change"] = changes
                time.sleep(min([c["interval"] for c in self.watchers.values()], default=5))
        
        self._watch_thread = threading.Thread(target=watcher_loop, daemon=True)
        self._watch_thread.start()
    
    def _send_notification(self, url: str, data: Dict):
        """Mock webhook – could be implemented with requests."""
        # In production, use `requests.post(url, json=data)`
        pass
    
    def batch_process(self, params: Dict) -> Dict:
        """
        Process multiple files efficiently.
        Parameters:
          - filepaths: list of file paths
          - operation: "read", "search", or "metadata"
          - keyword: (for search operation)
        """
        filepaths = params.get("filepaths", [])
        operation = params.get("operation", "read")
        keyword = params.get("keyword")
        
        if not filepaths:
            return self._error("No filepaths provided", -32602)
        
        results = []
        errors = []
        
        for fp in filepaths:
            try:
                if operation == "read":
                    res = self.fs.read_file(fp)
                    if res["success"]:
                        results.append({"filepath": fp, "content": res["content"], "metadata": res["metadata"]})
                    else:
                        errors.append({"filepath": fp, "error": res["error"]})
                elif operation == "search":
                    if not keyword:
                        errors.append({"filepath": fp, "error": "Missing keyword for search"})
                        continue
                    res = self.fs.search_in_file(fp, keyword)
                    if res["success"]:
                        results.append({"filepath": fp, "matches": res["matches"], "count": res["count"]})
                    else:
                        errors.append({"filepath": fp, "error": res["error"]})
                elif operation == "metadata":
                    res = self.fs.read_file(fp)
                    if res["success"]:
                        results.append({"filepath": fp, "metadata": res["metadata"]})
                    else:
                        errors.append({"filepath": fp, "error": res["error"]})
                else:
                    errors.append({"filepath": fp, "error": f"Unknown operation: {operation}"})
            except Exception as e:
                errors.append({"filepath": fp, "error": str(e)})
        
        return {
            "operation": operation,
            "total": len(filepaths),
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors
        }
    
    def get_file_info(self, params: Dict) -> Dict:
        """Get detailed info about a file without reading content."""
        filepath = params.get("filepath")
        if not filepath:
            return self._error("Missing 'filepath'", -32602)
        path = Path(filepath)
        if not path.exists():
            return self._error("File not found", -32002)
        stat = path.stat()
        return {
            "name": path.name,
            "path": str(path.absolute()),
            "size_bytes": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_readable": os.access(filepath, os.R_OK),
            "is_writable": os.access(filepath, os.W_OK),
        }
    
    # ------------------------------------------------------------------------
    # Resource Discovery (MCP specific)
    # ------------------------------------------------------------------------
    
    def list_resources(self, params: Dict) -> Dict:
        """Return list of available MCP resources (tools)."""
        resources = []
        for method_name in self.methods.keys():
            resources.append({
                "name": method_name,
                "description": self._get_description(method_name),
                "parameters": self._get_parameters_schema(method_name)
            })
        return {"resources": resources, "count": len(resources)}
    
    def get_resource(self, params: Dict) -> Dict:
        """Get detailed description of a specific resource."""
        resource_name = params.get("name")
        if not resource_name or resource_name not in self.methods:
            return self._error(f"Resource '{resource_name}' not found", -32601)
        return {
            "name": resource_name,
            "description": self._get_description(resource_name),
            "parameters": self._get_parameters_schema(resource_name),
            "example": self._get_example(resource_name)
        }
    
    def _get_description(self, method: str) -> str:
        desc = {
            "read_file": "Read the entire content of a file (TXT, PDF, DOCX).",
            "list_files": "List all files in a directory, optionally filtered by extension.",
            "write_file": "Write text content to a file, creating directories if needed.",
            "search_in_file": "Search for a keyword inside a file, returning context lines.",
            "watch_directory": "Start monitoring a directory for file changes (add/modify/remove).",
            "batch_process": "Perform operations (read/search/metadata) on multiple files efficiently.",
            "get_file_info": "Retrieve metadata about a file without reading its content.",
            "list_resources": "Discover all available MCP resources.",
            "get_resource": "Get detailed information about a specific resource.",
        }
        return desc.get(method, "No description available.")
    
    def _get_parameters_schema(self, method: str) -> Dict:
        schemas = {
            "read_file": {"type": "object", "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]},
            "list_files": {"type": "object", "properties": {"directory": {"type": "string"}, "extension": {"type": "string"}}},
            "write_file": {"type": "object", "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}}, "required": ["filepath", "content"]},
            "search_in_file": {"type": "object", "properties": {"filepath": {"type": "string"}, "keyword": {"type": "string"}}, "required": ["filepath", "keyword"]},
            "watch_directory": {"type": "object", "properties": {"directory": {"type": "string"}, "callback_url": {"type": "string"}, "interval_seconds": {"type": "integer"}}},
            "batch_process": {"type": "object", "properties": {"filepaths": {"type": "array"}, "operation": {"type": "string"}, "keyword": {"type": "string"}}, "required": ["filepaths", "operation"]},
            "get_file_info": {"type": "object", "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]},
        }
        return schemas.get(method, {"type": "object", "properties": {}})
    
    def _get_example(self, method: str) -> Dict:
        examples = {
            "read_file": {"jsonrpc": "2.0", "method": "read_file", "params": {"filepath": "./sample_data/resume.txt"}, "id": 1},
            "list_files": {"jsonrpc": "2.0", "method": "list_files", "params": {"directory": "./sample_data", "extension": ".pdf"}, "id": 2},
            "batch_process": {"jsonrpc": "2.0", "method": "batch_process", "params": {"filepaths": ["a.txt", "b.pdf"], "operation": "read"}, "id": 3},
        }
        return examples.get(method, {})
    
    # ------------------------------------------------------------------------
    # JSON-RPC 2.0 Handler
    # ------------------------------------------------------------------------
    
    def handle_request(self, request_body: str) -> str:
        """Parse JSON-RPC request and return response."""
        try:
            req = json.loads(request_body)
        except json.JSONDecodeError:
            return self._error("Parse error", -32700)
        
        # Validate JSON-RPC structure
        if req.get("jsonrpc") != "2.0":
            return self._error("Invalid JSON-RPC version", -32600)
        
        method = req.get("method")
        params = req.get("params", {})
        req_id = req.get("id")
        
        if not method:
            return self._error("Method not specified", -32600)
        
        if method not in self.methods:
            return self._error(f"Method '{method}' not found", -32601, req_id)
        
        try:
            result = self.methods[method](params)
            return json.dumps({
                "jsonrpc": "2.0",
                "result": result,
                "id": req_id
            })
        except Exception as e:
            return self._error(str(e), -32000, req_id)
    
    def _error(self, message: str, code: int, req_id=None) -> str:
        return json.dumps({
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": req_id
        })


# ============================================================================
# HTTP Server Wrapper
# ============================================================================

class MCPHTTPHandler(BaseHTTPRequestHandler):
    server_instance = None
    
    def log_message(self, format, *args):
        # Suppress default logging for cleaner output
        pass
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        response = self.server.server_instance.handle_request(body)
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))
    
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = """
            <html><body>
            <h1>MCP File System Server</h1>
            <p>JSON-RPC 2.0 endpoint: POST /</p>
            <p>Available methods:</p>
            <ul>
                <li>read_file</li><li>list_files</li><li>write_file</li><li>search_in_file</li>
                <li>watch_directory</li><li>batch_process</li><li>get_file_info</li>
                <li>list_resources</li><li>get_resource</li>
            </ul>
            </body></html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(host='localhost', port=8000, resumes_dir="./sample_data"):
    server = HTTPServer((host, port), MCPHTTPHandler)
    mcp = MCPServer(resumes_dir)
    MCPHTTPHandler.server_instance = mcp
    print(f"MCP File System Server running at http://{host}:{port}")
    print("JSON-RPC 2.0 endpoint: POST /")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resumes-dir", default="./sample_data")
    args = parser.parse_args()
    run_server(args.host, args.port, args.resumes_dir)