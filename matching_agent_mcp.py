"""
matching_agent_mcp.py  Refactored agent using MCP client instead of direct tools.
Connects to filesystem_mcp_server.py via JSON-RPC.
"""

import os
import json
import re
import traceback
import requests
from typing import Dict, List, Any, Optional, Annotated, TypedDict
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import RAG and job matcher (these still use local FAISS for embeddings)
from resume_rag import ResumeRAGSystem
from job_matcher import JobMatcher, JobDescription

# ============================================================================
# MCP Client (JSON-RPC 2.0)
# ============================================================================

class MCPClient:
    """Simple JSON-RPC 2.0 client for MCP server."""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.request_id = 0
    
    def _call(self, method: str, params: Dict) -> Dict:
        self.request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self.request_id
        }
        try:
            resp = requests.post(self.server_url, json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            if "error" in result:
                raise Exception(f"MCP error {result['error']['code']}: {result['error']['message']}")
            return result.get("result", {})
        except Exception as e:
            raise Exception(f"MCP call failed: {e}")
    
    # Convenience methods
    def read_file(self, filepath: str) -> Dict:
        return self._call("read_file", {"filepath": filepath})
    
    def list_files(self, directory: str, extension: str = None) -> Dict:
        params = {"directory": directory}
        if extension:
            params["extension"] = extension
        return self._call("list_files", params)
    
    def write_file(self, filepath: str, content: str) -> Dict:
        return self._call("write_file", {"filepath": filepath, "content": content})
    
    def search_in_file(self, filepath: str, keyword: str) -> Dict:
        return self._call("search_in_file", {"filepath": filepath, "keyword": keyword})
    
    def watch_directory(self, directory: str, callback_url: str = None, interval: int = 5) -> Dict:
        return self._call("watch_directory", {"directory": directory, "callback_url": callback_url, "interval_seconds": interval})
    
    def batch_process(self, filepaths: List[str], operation: str, keyword: str = None) -> Dict:
        params = {"filepaths": filepaths, "operation": operation}
        if keyword:
            params["keyword"] = keyword
        return self._call("batch_process", params)
    
    def get_file_info(self, filepath: str) -> Dict:
        return self._call("get_file_info", {"filepath": filepath})
    
    def list_resources(self) -> Dict:
        return self._call("list_resources", {})
    
    def get_resource(self, name: str) -> Dict:
        return self._call("get_resource", {"name": name})


# ============================================================================
# Agent State (same as before)
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[List[Dict[str, str]], operator.add]
    current_job_description: Optional[str]
    job_requirements: Optional[Dict[str, Any]]
    all_candidates: List[Dict[str, Any]]
    shortlisted_candidates: List[Dict[str, Any]]
    final_recommendations: List[Dict[str, Any]]
    current_stage: str
    waiting_for_human: bool
    human_feedback: Optional[str]
    reasoning_trail: List[Dict[str, str]]
    match_explanations: Dict[str, str]


# ============================================================================
# Agent Tools – Now use MCP client instead of direct fs_tools
# ============================================================================

class AgentToolsMCP:
    def __init__(self, resumes_dir: str = "./sample_data", mcp_url: str = "http://localhost:8000"):
        self.resumes_dir = resumes_dir
        self.mcp = MCPClient(mcp_url)
        # Still need RAG for embeddings (could also be an MCP, but keep local for simplicity)
        self.rag_system = ResumeRAGSystem(resumes_dir)
        self.job_matcher = JobMatcher(resumes_dir)
        
        # Ensure FAISS index exists
        vs = self.rag_system.vector_store
        if vs.index is None or vs.index.ntotal == 0:
            print("Building RAG index...")
            self.rag_system.build_index()
        
        # Verify MCP server is reachable
        try:
            self.mcp.list_resources()
            print(f"Connected to MCP server at {mcp_url}")
        except Exception as e:
            print(f"Warning: Could not connect to MCP server: {e}")
    
    # Replace file system operations with MCP calls
    def list_resumes(self, extension=None):
        return self.mcp.list_files(self.resumes_dir, extension)
    
    def read_resume(self, filepath):
        return self.mcp.read_file(filepath)
    
    def search_in_resume(self, filepath, keyword):
        return self.mcp.search_in_file(filepath, keyword)
    
    # Batch process example
    def batch_read_resumes(self, filepaths):
        return self.mcp.batch_process(filepaths, "read")
    
    def watch_resumes_folder(self, callback=None, interval=5):
        return self.mcp.watch_directory(self.resumes_dir, callback, interval)
    
    # The rest (semantic search, ranking) remain same as before
    def extract_requirements(self, job_description: str) -> Dict[str, Any]:
        if not job_description:
            return {
                "must_have_skills": [],
                "nice_to_have_skills": [],
                "min_experience_years": 0.0,
                "education_requirement": "",
                "requirements_text": []
            }
        jd = self.job_matcher.parse_job_description(job_description)
        return {
            "must_have_skills": jd.must_have_skills,
            "nice_to_have_skills": jd.nice_to_have_skills,
            "min_experience_years": jd.min_experience_years,
            "education_requirement": jd.education_requirement,
            "requirements_text": jd.requirements_text
        }
    
    def semantic_search_resumes(self, query: str, top_k: int = 20) -> List[Dict]:
        if not query:
            return []
        results = self.rag_system.search(query, top_k=top_k)
        # ... same aggregation as before ...
        candidates = {}
        for r in results:
            name = r['metadata']['candidate_name']
            if name not in candidates:
                candidates[name] = {
                    "candidate_name": name,
                    "resume_path": r['metadata']['resume_path'],
                    "chunks": [],
                    "relevance_scores": []
                }
            candidates[name]["chunks"].append(r)
            score = 1 - r.get('distance', 1.0)
            candidates[name]["relevance_scores"].append(score)
        result_list = []
        for name, data in candidates.items():
            avg_score = sum(data["relevance_scores"]) / len(data["relevance_scores"])
            result_list.append({
                "candidate_name": name,
                "resume_path": data["resume_path"],
                "relevance_score": avg_score,
                "matched_chunks": data["chunks"][:3]
            })
        result_list.sort(key=lambda x: x["relevance_score"], reverse=True)
        return result_list[:top_k]
    
    def rank_candidates(self, candidates: List[Dict], requirements: Dict) -> List[Dict]:
        # same as original matching_agent.py – identical code
        ranked = []
        for cand in candidates:
            cand_skills = []
            for chunk in cand.get("matched_chunks", []):
                skills_meta = chunk["metadata"].get("skills", "")
                if skills_meta:
                    cand_skills.extend([s.strip() for s in skills_meta.split(",")])
            cand_skills = list(set(cand_skills))
            must_have = set(requirements.get("must_have_skills", []))
            matched_must = list(must_have & set(cand_skills))
            missing_must = list(must_have - set(cand_skills))
            skill_score = (len(matched_must) / max(len(must_have), 1)) * 100
            exp_years = 0.0
            for chunk in cand.get("matched_chunks", []):
                exp_meta = chunk["metadata"].get("experience_years", 0)
                exp_years = max(exp_years, float(exp_meta))
            required_exp = requirements.get("min_experience_years", 0)
            if required_exp > 0:
                exp_score = 100 if exp_years >= required_exp else (exp_years / required_exp) * 100
            else:
                exp_score = 100
            semantic_score = cand.get("relevance_score", 0) * 100
            final_score = skill_score * 0.6 + exp_score * 0.3 + semantic_score * 0.1
            reasoning_parts = []
            if matched_must:
                reasoning_parts.append(f"✓ Matched: {', '.join(matched_must[:3])}")
            if missing_must:
                reasoning_parts.append(f"✗ Missing: {', '.join(missing_must[:2])}")
            reasoning_parts.append(f"📅 Exp: {exp_years:.1f}/{required_exp} yrs")
            improvement = ""
            if 40 <= final_score < 70:
                if missing_must:
                    improvement = f"Suggestion: Add {', '.join(missing_must[:2])}."
            ranked.append({
                "candidate_name": cand["candidate_name"],
                "resume_path": cand["resume_path"],
                "match_score": round(final_score, 1),
                "matched_skills": matched_must,
                "missing_skills": missing_must,
                "experience_years": exp_years,
                "reasoning": " | ".join(reasoning_parts),
                "improvement_suggestions": improvement,
                "relevant_excerpts": [chunk["content"][:300] for chunk in cand.get("matched_chunks", [])[:2]]
            })
        ranked.sort(key=lambda x: x["match_score"], reverse=True)
        return ranked


# ============================================================================
# LangGraph Agent (identical to previous but using AgentToolsMCP)
# ============================================================================

# The rest of the MatchingAgent class is nearly identical to milestone 3,
# only the initialization uses AgentToolsMCP instead of direct tools.
# I'll show the relevant __init__ and keep the rest unchanged.

class MatchingAgentMCP:
    def __init__(self, resumes_dir: str = "./sample_data", mcp_url: str = "http://localhost:8000"):
        self.tools = AgentToolsMCP(resumes_dir, mcp_url)
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.current_state = None
    
    def _build_graph(self):
        # Same graph as milestone 3 (parse_jd → extract_requirements → ...)
        # Reuse the same node implementations, only the tool calls are now via MCP.
        # (The code is identical to previous `_build_graph`, so omitted for brevity.)
        # Refer to milestone 3 code.
        pass
    
    # All node methods remain unchanged – they call self.tools.methods which now use MCP.
    # (For full code, see the final matching_agent.py from Milestone 3, replacing AgentTools with AgentToolsMCP.)


# ============================================================================
# Interactive Session (same as before)
# ============================================================================

def interactive_session():
    print("\n" + "="*70)
    print("🤖 MCP-Based Resume Matching Agent")
    print("="*70)
    print("Connecting to MCP server at http://localhost:8000 ...")
    agent = MatchingAgentMCP(resumes_dir="./sample_data", mcp_url="http://localhost:8000")
    # The rest is identical to Milestone 3 interactive loop.
    # ...
    print("Interactive session ready. Type 'help' for commands.")


if __name__ == "__main__":
    # First ensure MCP server is running before starting agent.
    print("Please start the MCP server in a separate terminal: python filesystem_mcp_server.py")
    input("Press Enter after server is running...")
    interactive_session()