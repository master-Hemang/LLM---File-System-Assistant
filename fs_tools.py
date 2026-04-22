
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
import json

# Optional imports for different file types
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class FileSystemTools:
    """Core file system operations with structured responses"""
    
    @staticmethod
    def read_file(filepath: str) -> Dict[str, Any]:
        """
        Read resume files (PDF, TXT, DOCX)
        
        Args:
            filepath: Path to the file
            
        Returns:
            Dictionary with content, metadata, and status
        """
        try:
            path = Path(filepath)
            
            if not path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {filepath}",
                    "content": None,
                    "metadata": None
                }
            
            # Get file metadata
            stat = path.stat()
            metadata = {
                "filename": path.name,
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": path.suffix.lower()
            }
            
            # Extract content based on file type
            extension = path.suffix.lower()
            content = None
            
            if extension == '.txt':
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
            elif extension == '.pdf':
                if not PDF_AVAILABLE:
                    return {
                        "success": False,
                        "error": "PyPDF2 not installed. Run: pip install PyPDF2",
                        "content": None,
                        "metadata": metadata
                    }
                content = FileSystemTools._read_pdf(filepath)
                
            elif extension == '.docx':
                if not DOCX_AVAILABLE:
                    return {
                        "success": False,
                        "error": "python-docx not installed. Run: pip install python-docx",
                        "content": None,
                        "metadata": metadata
                    }
                content = FileSystemTools._read_docx(filepath)
                
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {extension}",
                    "content": None,
                    "metadata": metadata
                }
            
            return {
                "success": True,
                "error": None,
                "content": content,
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": None,
                "metadata": None
            }
    
    @staticmethod
    def _read_pdf(filepath: str) -> str:
        """Extract text from PDF file"""
        content = []
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                content.append(page.extract_text())
        return '\n'.join(content)
    
    @staticmethod
    def _read_docx(filepath: str) -> str:
        """Extract text from DOCX file"""
        doc = Document(filepath)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    
    @staticmethod
    def list_files(directory: str, extension: Optional[str] = None) -> Dict[str, Any]:
        """
        List all files in a directory with optional extension filter
        
        Args:
            directory: Directory path to search
            extension: Optional file extension filter (e.g., '.pdf')
            
        Returns:
            List of file metadata
        """
        try:
            path = Path(directory)
            
            if not path.exists():
                return {
                    "success": False,
                    "error": f"Directory not found: {directory}",
                    "files": []
                }
            
            if not path.is_dir():
                return {
                    "success": False,
                    "error": f"Path is not a directory: {directory}",
                    "files": []
                }
            
            files = []
            for file_path in path.iterdir():
                if file_path.is_file():
                    # Apply extension filter
                    if extension and file_path.suffix.lower() != extension.lower():
                        continue
                    
                    stat = file_path.stat()
                    files.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "size_bytes": stat.st_size,
                        "size_kb": round(stat.st_size / 1024, 2),
                        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "extension": file_path.suffix.lower()
                    })
            
            return {
                "success": True,
                "error": None,
                "files": files,
                "count": len(files),
                "directory": str(path.absolute())
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "files": []
            }
    
    @staticmethod
    def write_file(filepath: str, content: str) -> Dict[str, Any]:
        """
        Write content to file, creating directories if needed
        
        Args:
            filepath: Path where to write the file
            content: Content to write
            
        Returns:
            Status dictionary
        """
        try:
            path = Path(filepath)
            
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Get file info
            stat = path.stat()
            
            return {
                "success": True,
                "error": None,
                "filepath": str(path.absolute()),
                "size_bytes": stat.st_size,
                "message": f"Successfully wrote {stat.st_size} bytes to {filepath}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "filepath": filepath,
                "message": f"Failed to write file: {str(e)}"
            }
    
    @staticmethod
    def search_in_file(filepath: str, keyword: str, context_chars: int = 100) -> Dict[str, Any]:
        """
        Search for keywords in file content with surrounding context
        
        Args:
            filepath: Path to the file
            keyword: Keyword to search for (case-insensitive)
            context_chars: Number of context characters around match
            
        Returns:
            Dictionary with matches and context
        """
        try:
            # First read the file
            read_result = FileSystemTools.read_file(filepath)
            
            if not read_result["success"]:
                return {
                    "success": False,
                    "error": read_result["error"],
                    "matches": [],
                    "count": 0
                }
            
            content = read_result["content"]
            keyword_lower = keyword.lower()
            content_lower = content.lower()
            
            matches = []
            start = 0
            
            while True:
                # Find next occurrence
                pos = content_lower.find(keyword_lower, start)
                if pos == -1:
                    break
                
                # Extract context
                context_start = max(0, pos - context_chars)
                context_end = min(len(content), pos + len(keyword) + context_chars)
                context = content[context_start:context_end]
                
                # Add ellipsis if context is truncated
                if context_start > 0:
                    context = "..." + context
                if context_end < len(content):
                    context = context + "..."
                
                matches.append({
                    "position": pos,
                    "line_number": content[:pos].count('\n') + 1,
                    "context": context,
                    "keyword": keyword
                })
                
                start = pos + len(keyword)
            
            return {
                "success": True,
                "error": None,
                "filepath": filepath,
                "keyword": keyword,
                "matches": matches,
                "count": len(matches)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "matches": [],
                "count": 0
            }


# Tool definitions for LLM function calling
TOOLS_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a resume file (supports PDF, TXT, DOCX formats)",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in a directory, optionally filtered by extension",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to search"
                    },
                    "extension": {
                        "type": "string",
                        "description": "Optional file extension filter (e.g., '.pdf')"
                    }
                },
                "required": ["directory"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates directories if needed)",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path where to write the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["filepath", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": "Search for a keyword in a file and return matches with surrounding context",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to search"
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Keyword to search for (case-insensitive)"
                    }
                },
                "required": ["filepath", "keyword"]
            }
        }
    }
]