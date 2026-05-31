"""
resume_rag.py - RAG System with FAISS (no build tools required)
"""

import os
import re
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from fs_tools import FileSystemTools


@dataclass
class ResumeChunk:
    chunk_id: str
    resume_path: str
    candidate_name: str
    section: str
    content: str
    metadata: Dict[str, Any]


class ResumeProcessor:
    # (same as before – no changes)
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.fs = FileSystemTools()
        self.section_patterns = {
            "skills": r'(?i)(technical skills|skills|core competencies|technologies?)(?:\s*:)?\s*\n([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\n(?=[A-Z][a-z]+:)|\Z)',
            "experience": r'(?i)(work experience|employment|professional experience|experience)(?:\s*:)?\s*\n([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\n(?=[A-Z][a-z]+:)|\Z)',
            "education": r'(?i)(education|academic background|qualifications)(?:\s*:)?\s*\n([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\n(?=[A-Z][a-z]+:)|\Z)'
        }

    def load_resume(self, filepath: str) -> Optional[str]:
        result = self.fs.read_file(filepath)
        return result["content"] if result["success"] else None

    def extract_metadata(self, content: str, filepath: str) -> Dict[str, Any]:
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        name = lines[0] if lines else Path(filepath).stem.replace('_', ' ').title()
        common_skills = [
            'Python','Java','JavaScript','React','AWS','Docker','SQL','TensorFlow'
        ]
        content_lower = content.lower()
        skills = [s for s in common_skills if s.lower() in content_lower]
        exp_match = re.search(r'(\d+)\+?\s*years?', content_lower)
        exp_years = float(exp_match.group(1)) if exp_match else 0.0
        education = []
        edu_keywords = ['bachelor','master','phd','b.sc','m.sc','b.tech','mba']
        for line in lines:
            if any(k in line.lower() for k in edu_keywords):
                education.append(line[:200])
        return {
            "candidate_name": name,
            "skills": skills,
            "experience_years": exp_years,
            "education": education[:3],
            "filepath": filepath
        }

    def chunk_document(self, content: str, filepath: str, metadata: Dict) -> List[ResumeChunk]:
        chunks = []
        idx = 0
        found = False
        for sec, pat in self.section_patterns.items():
            for m in re.finditer(pat, content, re.DOTALL):
                sec_content = m.group(2).strip()
                if sec_content:
                    if len(sec_content) > self.chunk_size:
                        sub = self._chunk_text(sec_content, sec)
                        for sub_text, _ in sub:
                            chunks.append(ResumeChunk(
                                chunk_id=self._make_id(filepath, sec, idx),
                                resume_path=filepath,
                                candidate_name=metadata["candidate_name"],
                                section=sec,
                                content=sub_text,
                                metadata=metadata
                            ))
                            idx += 1
                    else:
                        chunks.append(ResumeChunk(
                            chunk_id=self._make_id(filepath, sec, idx),
                            resume_path=filepath,
                            candidate_name=metadata["candidate_name"],
                            section=sec,
                            content=sec_content,
                            metadata=metadata
                        ))
                        idx += 1
                    found = True
        if not found:
            text_chunks = self._chunk_text(content, "general")
            for sub_text, _ in text_chunks:
                chunks.append(ResumeChunk(
                    chunk_id=self._make_id(filepath, "general", idx),
                    resume_path=filepath,
                    candidate_name=metadata["candidate_name"],
                    section="general",
                    content=sub_text,
                    metadata=metadata
                ))
                idx += 1
        return chunks

    def _chunk_text(self, text: str, section: str) -> List[tuple]:
        words = text.split()
        step = self.chunk_size - self.overlap
        result = []
        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i+self.chunk_size])
            result.append((chunk, i//step))
        return result if result else [(text, 0)]

    def _make_id(self, filepath: str, section: str, idx: int) -> str:
        raw = f"{filepath}_{section}_{idx}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]


class VectorStoreManager:
    """FAISS-based vector store (no compilation needed)"""

    def __init__(self, persist_dir: str = "./faiss_index"):
        self.persist_dir = persist_dir
        self.index = None
        self.chunks = []          # list of (chunk_id, content, metadata)
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = self.embed_model.get_embedding_dimension()

    def add_chunks(self, chunks: List[ResumeChunk], batch_size: int = 100):
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)   # Inner Product (cosine after normalisation)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [c.content for c in batch]
            embeddings = self.embed_model.encode(texts)
            # Normalise for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            for c in batch:
                meta = {
                    "chunk_id": c.chunk_id,
                    "resume_path": c.resume_path,
                    "candidate_name": c.candidate_name,
                    "section": c.section,
                    "skills": ", ".join(c.metadata.get("skills", [])),
                    "experience_years": float(c.metadata.get("experience_years", 0)),
                    "education": ", ".join(c.metadata.get("education", []))
                }
                self.chunks.append((c.chunk_id, c.content, meta))
            print(f"  Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

        # Save index and chunks to disk
        self.save()

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q_emb = self.embed_model.encode([query])
        faiss.normalize_L2(q_emb)
        distances, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            chunk_id, content, meta = self.chunks[idx]
            results.append({
                "chunk_id": chunk_id,
                "content": content,
                "metadata": meta,
                "distance": float(1 - distances[0][i])   # convert similarity to distance
            })
        return results

    def save(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        # Save FAISS index
        if self.index:
            faiss.write_index(self.index, os.path.join(self.persist_dir, "index.faiss"))
        # Save chunks metadata
        with open(os.path.join(self.persist_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self):
        index_path = os.path.join(self.persist_dir, "index.faiss")
        chunks_path = os.path.join(self.persist_dir, "chunks.pkl")
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            return True
        return False


class ResumeRAGSystem:
    def __init__(self, resumes_dir: str = "./sample_data"):
        self.resumes_dir = resumes_dir
        self.processor = ResumeProcessor()
        self.vector_store = VectorStoreManager()
        # Try to load existing index
        if not self.vector_store.load():
            print("No existing FAISS index found. Run --build to create one.")

    def build_index(self, force_rebuild: bool = False):
        if force_rebuild:
            import shutil
            if Path("./faiss_index").exists():
                shutil.rmtree("./faiss_index")
                print("Removed old faiss_index.")
            self.vector_store = VectorStoreManager()

        # Check if already built
        if not force_rebuild and self.vector_store.load():
            print("Index already exists. Use force_rebuild=True to rebuild.")
            return

        # List resumes
        result = self.processor.fs.list_files(self.resumes_dir)
        if not result["success"]:
            print(f"Error: {result.get('error')}")
            return

        all_chunks = []
        for file_info in tqdm(result["files"], desc="Processing resumes"):
            path = file_info["path"]
            ext = Path(path).suffix.lower()
            if ext not in ['.txt', '.pdf', '.docx']:
                continue
            content = self.processor.load_resume(path)
            if not content:
                continue
            metadata = self.processor.extract_metadata(content, path)
            chunks = self.processor.chunk_document(content, path, metadata)
            all_chunks.extend(chunks)

        self.vector_store.add_chunks(all_chunks)
        print(f"✅ Indexed {len(all_chunks)} chunks from {len(result['files'])} files.")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        return self.vector_store.search(query, top_k)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query", type=str)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    rag = ResumeRAGSystem()
    if args.build:
        rag.build_index(force_rebuild=True)
    if args.query:
        results = rag.search(args.query, top_k=args.top_k)
        for i, r in enumerate(results, 1):
            score = 1 - r['distance']
            print(f"{i}. {r['metadata']['candidate_name']} (score: {score:.3f})")
            print(f"   Section: {r['metadata']['section']}")
            print(f"   Preview: {r['content'][:150]}...\n")


if __name__ == "__main__":
    main()