import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Optional imports
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
    @staticmethod
    def read_file(filepath: str) -> Dict[str, Any]:
        try:
            path = Path(filepath)
            if not path.exists():
                return {"success": False, "error": f"File not found: {filepath}"}
            stat = path.stat()
            metadata = {
                "filename": path.name,
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": path.suffix.lower()
            }
            ext = path.suffix.lower()
            content = None
            if ext == '.txt':
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif ext == '.pdf':
                if not PDF_AVAILABLE:
                    return {"success": False, "error": "PyPDF2 not installed"}
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    content = '\n'.join([page.extract_text() for page in reader.pages if page.extract_text()])
            elif ext == '.docx':
                if not DOCX_AVAILABLE:
                    return {"success": False, "error": "python-docx not installed"}
                doc = Document(filepath)
                content = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
            else:
                return {"success": False, "error": f"Unsupported type: {ext}"}
            return {"success": True, "content": content, "metadata": metadata}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def list_files(directory: str, extension: Optional[str] = None) -> Dict[str, Any]:
        try:
            path = Path(directory)
            if not path.exists() or not path.is_dir():
                return {"success": False, "error": "Invalid directory"}
            files = []
            for f in path.iterdir():
                if f.is_file():
                    if extension and f.suffix.lower() != extension.lower():
                        continue
                    stat = f.stat()
                    files.append({
                        "name": f.name,
                        "path": str(f),
                        "size_bytes": stat.st_size,
                        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
            return {"success": True, "files": files, "count": len(files)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def write_file(filepath: str, content: str) -> Dict[str, Any]:
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return {"success": True, "message": f"Written to {filepath}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def search_in_file(filepath: str, keyword: str) -> Dict[str, Any]:
        read_res = FileSystemTools.read_file(filepath)
        if not read_res["success"]:
            return read_res
        content = read_res["content"]
        matches = []
        idx = 0
        while True:
            pos = content.lower().find(keyword.lower(), idx)
            if pos == -1:
                break
            start = max(0, pos - 100)
            end = min(len(content), pos + len(keyword) + 100)
            context = content[start:end].replace('\n', ' ')
            matches.append({"position": pos, "context": context})
            idx = pos + len(keyword)
        return {"success": True, "matches": matches, "count": len(matches)}