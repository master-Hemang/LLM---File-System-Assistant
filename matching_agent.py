"""
matching_agent.py - LangGraph-based intelligent resume matching agent
"""

import os
import json
import re
import traceback
from typing import Dict, List, Any, Optional, Annotated, TypedDict
from dataclasses import dataclass
from pathlib import Path
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# LLM (optional)
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    LLM_MODEL = genai.GenerativeModel("gemini-2.5-flash-lite")
    LLM_AVAILABLE = True
except:
    LLM_AVAILABLE = False

from fs_tools import FileSystemTools
from resume_rag import ResumeRAGSystem
from job_matcher import JobMatcher, JobDescription


# ============================================================================
# Agent State Definition
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
# Tools for the Agent
# ============================================================================

class AgentTools:
    def __init__(self, resumes_dir: str = "./sample_data"):
        self.fs_tools = FileSystemTools()
        self.rag_system = ResumeRAGSystem(resumes_dir)
        self.job_matcher = JobMatcher(resumes_dir)

        vs = self.rag_system.vector_store
        if vs.index is None or vs.index.ntotal == 0:
            print("Building RAG index...")
            self.rag_system.build_index()

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

    def compare_candidates(self, candidate_ids: List[str]) -> Dict:
        return {"message": "Comparison feature available via 'compare top N' command."}

    def generate_interview_questions(self, candidate_name: str, requirements: Dict) -> List[str]:
        questions = []
        missing = requirements.get("missing_skills_for_candidate", [])
        for skill in missing[:3]:
            questions.append(f"Please describe your experience with {skill}.")
        questions.append("Can you walk me through your most challenging project?")
        questions.append("How do you stay current with new technologies?")
        if LLM_AVAILABLE:
            prompt = f"""Generate 2 specific technical interview questions for a candidate applying for a role with must-have skills: {requirements.get('must_have_skills', [])}. Candidate missing: {missing}. Return as JSON list."""
            try:
                resp = LLM_MODEL.generate_content(prompt)
                llm_q = json.loads(resp.text)
                questions = llm_q[:2] + questions[:2]
            except:
                pass
        return questions[:5]


# ============================================================================
# LangGraph Agent Workflow
# ============================================================================

class MatchingAgent:
    def __init__(self, resumes_dir: str = "./sample_data"):
        self.tools = AgentTools(resumes_dir)
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.current_state = None

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("parse_jd", self.parse_jd_node)
        workflow.add_node("extract_requirements", self.extract_requirements_node)
        workflow.add_node("search_resumes", self.search_resumes_node)
        workflow.add_node("rank_candidates", self.rank_candidates_node)
        workflow.add_node("wait_for_feedback", self.human_feedback_node)
        workflow.add_node("generate_report", self.generate_report_node)
        workflow.add_node("refine_requirements", self.refine_requirements_node)

        workflow.set_entry_point("parse_jd")
        workflow.add_edge("parse_jd", "extract_requirements")
        workflow.add_edge("extract_requirements", "search_resumes")
        workflow.add_edge("search_resumes", "rank_candidates")

        workflow.add_conditional_edges(
            "rank_candidates",
            self.should_get_feedback,
            {
                "feedback": "wait_for_feedback",
                "report": "generate_report"
            }
        )
        workflow.add_edge("wait_for_feedback", "refine_requirements")
        workflow.add_edge("refine_requirements", "search_resumes")
        workflow.add_edge("generate_report", END)

        return workflow.compile(checkpointer=self.memory)

    # ----- Node implementations -----
    def parse_jd_node(self, state: AgentState) -> AgentState:
        try:
            last_msg = state["messages"][-1]["content"]
            if not last_msg or not isinstance(last_msg, str):
                last_msg = ""
            state["current_job_description"] = last_msg
            state["current_stage"] = "parsing"
            state["reasoning_trail"].append({"step": "parse_jd", "message": "Parsed job description"})
        except Exception as e:
            state["current_job_description"] = ""
            state["reasoning_trail"].append({"step": "parse_jd", "message": f"Error: {e}"})
        return state

    def extract_requirements_node(self, state: AgentState) -> AgentState:
        jd_text = state.get("current_job_description", "")
        if not jd_text:
            jd_text = ""
        reqs = self.tools.extract_requirements(jd_text)
        state["job_requirements"] = reqs
        state["current_stage"] = "extracted"
        state["reasoning_trail"].append({"step": "extract_requirements", "message": f"Must-haves: {reqs['must_have_skills'][:3]}"})
        return state

    def search_resumes_node(self, state: AgentState) -> AgentState:
        jd_text = state.get("current_job_description", "")
        must_skills = state["job_requirements"].get("must_have_skills", [])
        boosted_query = jd_text + " " + " ".join(must_skills[:5]) if must_skills else jd_text
        if not boosted_query:
            boosted_query = "software engineer"
        candidates = self.tools.semantic_search_resumes(boosted_query, top_k=20)
        state["all_candidates"] = candidates
        state["current_stage"] = "searched"
        state["reasoning_trail"].append({"step": "search_resumes", "message": f"Found {len(candidates)} candidates"})
        return state

    def rank_candidates_node(self, state: AgentState) -> AgentState:
        candidates = state["all_candidates"]
        reqs = state["job_requirements"]
        ranked = self.tools.rank_candidates(candidates, reqs)
        state["shortlisted_candidates"] = ranked[:10]
        state["current_stage"] = "ranked"
        if ranked:
            state["reasoning_trail"].append({"step": "rank_candidates", "message": f"Top candidate: {ranked[0]['candidate_name']} ({ranked[0]['match_score']})"})
        else:
            state["reasoning_trail"].append({"step": "rank_candidates", "message": "No candidates found"})
        return state

    def should_get_feedback(self, state: AgentState) -> str:
        # If there is no pending human feedback, go directly to report
        if not state.get("human_feedback"):
            return "report"
        return "feedback"

    def human_feedback_node(self, state: AgentState) -> AgentState:
        state["waiting_for_human"] = True
        return state

    def refine_requirements_node(self, state: AgentState) -> AgentState:
        feedback = state.get("human_feedback")
        if not feedback or not isinstance(feedback, str) or feedback.strip() == "":
            # No feedback – skip refinement
            return state

        current = state.get("job_requirements", {})
        words = feedback.lower().split()
        common = ['python', 'react', 'aws', 'docker', 'java', 'sql', 'kubernetes', 'spring', 'microservices']
        for skill in common:
            if skill in words and skill not in [s.lower() for s in current.get("must_have_skills", [])]:
                current["must_have_skills"].append(skill.capitalize())
        state["job_requirements"] = current
        state["waiting_for_human"] = False
        state["human_feedback"] = None
        state["current_stage"] = "refined"
        state["reasoning_trail"].append({"step": "refine_requirements", "message": f"Refined based on: {feedback[:50]}"})
        return state

    def generate_report_node(self, state: AgentState) -> AgentState:
        shortlist = state["shortlisted_candidates"]
        final = []
        for c in shortlist[:5]:
            score = c["match_score"]
            if score >= 85:
                decision = "HIRE"
            elif score >= 70:
                decision = "CONSIDER"
            else:
                decision = "NO_HIRE"
            final.append({
                "candidate_name": c["candidate_name"],
                "resume_path": c["resume_path"],
                "match_score": score,
                "decision": decision,
                "reasoning": c["reasoning"],
                "improvement_suggestions": c.get("improvement_suggestions", "")
            })
        state["final_recommendations"] = final
        state["current_stage"] = "done"
        return state

    # ----- Interactive methods -----
    def process_user_message(self, user_input: str, thread_id: str = "default") -> str:
        if not user_input or not isinstance(user_input, str) or user_input.strip() == "":
            return "Please provide a valid job description or command."

        lower = user_input.lower().strip()
        if "compare" in lower and "top" in lower:
            return self._handle_compare(user_input)
        elif "why did" in lower and "rank" in lower:
            return self._handle_explain_ranking(user_input)
        elif "adjust" in lower or "add requirement" in lower:
            return self._handle_refinement(user_input, thread_id)
        elif "shortlist" in lower:
            return self._show_shortlist(thread_id)
        elif "interview questions" in lower:
            return self._generate_interview_questions(user_input, thread_id)
        else:
            return self._run_workflow(user_input, thread_id)

    def _run_workflow(self, job_desc: str, thread_id: str) -> str:
        if not job_desc or not job_desc.strip():
            return "No job description provided. Please enter a valid job description."

        initial_state = {
            "messages": [{"role": "user", "content": job_desc}],
            "current_job_description": None,
            "job_requirements": None,
            "all_candidates": [],
            "shortlisted_candidates": [],
            "final_recommendations": [],
            "current_stage": "start",
            "waiting_for_human": False,
            "human_feedback": None,
            "reasoning_trail": [],
            "match_explanations": {}
        }
        config = {"configurable": {"thread_id": thread_id}}
        try:
            final_state = self.graph.invoke(initial_state, config)
            self.current_state = final_state
            return self._format_ranking_response(final_state)
        except Exception as e:
            traceback.print_exc()
            return f"Error during workflow: {str(e)}"

    def _handle_compare(self, user_input: str) -> str:
        if not self.current_state or not self.current_state.get("shortlisted_candidates"):
            return "No candidates yet. Please provide a job description first."
        top_n = 3
        m = re.search(r'top (\d+)', user_input.lower())
        if m:
            top_n = int(m.group(1))
        cands = self.current_state["shortlisted_candidates"][:top_n]
        if not cands:
            return "No candidates to compare."
        lines = ["**Comparison:**\n"]
        lines.append("| Candidate | Score | Matched Skills | Missing | Experience |")
        lines.append("|-----------|-------|----------------|---------|------------|")
        for c in cands:
            matched = ", ".join(c.get("matched_skills", [])[:3]) or "none"
            missing = ", ".join(c.get("missing_skills", [])[:2]) or "none"
            exp = f"{c.get('experience_years',0)} yrs"
            lines.append(f"| {c['candidate_name']} | {c['match_score']} | {matched} | {missing} | {exp} |")
        lines.append(f"\n**Top candidate reasoning:** {cands[0].get('reasoning', 'N/A')}")
        return "\n".join(lines)

    def _handle_explain_ranking(self, user_input: str) -> str:
        words = user_input.split()
        names = [w for w in words if w[0].isupper() and len(w) > 2]
        if len(names) < 2:
            return "Please specify two names, e.g., 'why did John rank higher than Jane?'"
        cands = self.current_state.get("shortlisted_candidates", [])
        ca = next((c for c in cands if c["candidate_name"] == names[0]), None)
        cb = next((c for c in cands if c["candidate_name"] == names[1]), None)
        if not ca or not cb:
            return f"Candidates not found. Available: {[c['candidate_name'] for c in cands[:5]]}"
        exp = f"**Why {names[0]} ranked higher:**\n"
        exp += f"- Score: {ca['match_score']} vs {cb['match_score']}\n"
        exp += f"- Skills matched: {len(ca['matched_skills'])} vs {len(cb['matched_skills'])}\n"
        exp += f"- Missing: {ca['missing_skills']} vs {cb['missing_skills']}\n"
        exp += f"- Experience: {ca.get('experience_years',0)} vs {cb.get('experience_years',0)} yrs\n"
        exp += f"- Reasoning: {ca.get('reasoning', 'N/A')}"
        return exp

    def _handle_refinement(self, user_input: str, thread_id: str) -> str:
        if not self.current_state:
            return "No active job search. Provide a job description first."
        self.current_state["human_feedback"] = user_input
        refined = self.refine_requirements_node(self.current_state)
        refined = self.search_resumes_node(refined)
        refined = self.rank_candidates_node(refined)
        self.current_state = refined
        return "✅ Requirements updated. Re-ranked candidates:\n\n" + self._format_ranking_response(refined)

    def _show_shortlist(self, thread_id: str) -> str:
        if not self.current_state or not self.current_state.get("shortlisted_candidates"):
            return "No shortlist yet. Run a job search first."
        return self._format_ranking_response(self.current_state)

    def _generate_interview_questions(self, user_input: str, thread_id: str) -> str:
        words = user_input.split()
        name = None
        for w in words:
            if w[0].isupper() and len(w) > 2:
                name = w
                break
        if not name:
            return "Specify a candidate name, e.g., 'interview questions for John'"
        cands = self.current_state.get("shortlisted_candidates", [])
        cand = next((c for c in cands if c["candidate_name"] == name), None)
        if not cand:
            return f"Candidate '{name}' not found."
        reqs = self.current_state["job_requirements"].copy()
        reqs["missing_skills_for_candidate"] = cand.get("missing_skills", [])
        questions = self.tools.generate_interview_questions(name, reqs)
        output = f"**Interview questions for {name}:**\n\n"
        for i, q in enumerate(questions, 1):
            output += f"{i}. {q}\n"
        return output

    def _format_ranking_response(self, state: AgentState) -> str:
        shortlist = state.get("shortlisted_candidates", [])[:10]
        if not shortlist:
            return "No candidates found matching your criteria."
        lines = [f"**Top {len(shortlist)} Candidates (Score /100):**\n"]
        for i, c in enumerate(shortlist, 1):
            lines.append(f"{i}. **{c['candidate_name']}** - {c['match_score']}")
            lines.append(f"   ✅ Matched: {', '.join(c.get('matched_skills', [])[:5])}")
            if c.get("missing_skills"):
                lines.append(f"   ❌ Missing: {', '.join(c['missing_skills'][:3])}")
            lines.append(f"   💡 {c.get('reasoning', '')[:120]}...\n")
        if state.get("final_recommendations"):
            lines.append("**Final Recommendations:**")
            for rec in state["final_recommendations"]:
                emoji = "✅" if rec["decision"] == "HIRE" else "⚠️" if rec["decision"] == "CONSIDER" else "❌"
                lines.append(f"{emoji} {rec['candidate_name']}: {rec['decision']} ({rec['match_score']})")
        return "\n".join(lines)


# ============================================================================
# Interactive Console Interface
# ============================================================================

def interactive_session():
    print("\n" + "="*70)
    print("🤖 Intelligent Resume Matching Agent (LangGraph)")
    print("="*70)
    print("\nCommands:")
    print("  • Enter a job description to start matching")
    print("  • 'compare top N' - Compare top candidates")
    print("  • 'why did X rank higher than Y' - Explain ranking")
    print("  • 'adjust requirements: add React' - Refine search")
    print("  • 'shortlist' - Show current top candidates")
    print("  • 'interview questions for [Name]' - Generate questions")
    print("  • 'quit' - Exit\n")

    agent = MatchingAgent(resumes_dir="./sample_data")

    while True:
        try:
            user_input = input("\n📝 You: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break
            if not user_input:
                continue
            response = agent.process_user_message(user_input)
            print(f"\n🤖 Agent: {response}")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            traceback.print_exc()


if __name__ == "__main__":
    interactive_session()