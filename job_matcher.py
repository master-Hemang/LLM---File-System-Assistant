import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from resume_rag import ResumeRAGSystem

@dataclass
class JobDescription:
    must_have_skills: List[str]
    nice_to_have_skills: List[str]
    min_experience_years: float
    education_requirement: str
    requirements_text: List[str]

class JobMatcher:
    def __init__(self, resumes_dir: str = "./sample_data"):
        self.rag = ResumeRAGSystem(resumes_dir)
        # Check if FAISS index exists; if not, build it
        if self.rag.vector_store.index is None or self.rag.vector_store.index.ntotal == 0:
            print("No existing FAISS index found. Building...")
            self.rag.build_index()

    def parse_job_description(self, jd_text: str) -> JobDescription:
        common_skills = [
            'Python','Java','JavaScript','React','AWS','Docker','SQL',
            'TensorFlow','Kubernetes','Django','Flask','Machine Learning'
        ]
        must = []
        nice = []
        for skill in common_skills:
            if skill.lower() in jd_text.lower():
                # Look at surrounding context to decide if it's "must have"
                idx = jd_text.lower().find(skill.lower())
                start = max(0, idx - 50)
                end = min(len(jd_text), idx + 50)
                context = jd_text.lower()[start:end]
                if any(k in context for k in ['must','required','need','essential']):
                    must.append(skill)
                else:
                    nice.append(skill)

        exp_match = re.search(r'(\d+)\+?\s*years?', jd_text.lower())
        min_exp = float(exp_match.group(1)) if exp_match else 0.0

        edu_match = re.search(r'(bachelor|master|phd|b\.?s\.?|m\.?s\.?|b\.?tech)', jd_text.lower())
        edu_req = edu_match.group(0) if edu_match else ""

        return JobDescription(
            must_have_skills=must,
            nice_to_have_skills=nice,
            min_experience_years=min_exp,
            education_requirement=edu_req,
            requirements_text=[]
        )

    def semantic_search_candidates(self, jd: JobDescription, top_k: int = 20) -> List[Dict]:
        # Boost query with must-have skills and education
        boosted_query = " ".join(jd.must_have_skills[:5]) + " " + jd.education_requirement
        return self.rag.search(boosted_query, top_k=top_k)

    def rank_candidates(self, candidates: List[Dict], jd: JobDescription) -> List[Dict]:
        ranked = []
        for cand in candidates:
            meta = cand['metadata']
            # Skills are stored as comma-separated string
            cand_skills = meta.get('skills', '').split(', ') if meta.get('skills') else []
            matched_must = [s for s in jd.must_have_skills if s in cand_skills]
            skill_score = (len(matched_must) / max(len(jd.must_have_skills), 1)) * 100

            exp_years = float(meta.get('experience_years', 0))
            required_exp = jd.min_experience_years
            if required_exp > 0:
                exp_score = 100 if exp_years >= required_exp else (exp_years / required_exp) * 100
            else:
                exp_score = 100

            sem_score = (1 - cand.get('distance', 1)) * 100
            final_score = skill_score * 0.6 + exp_score * 0.3 + sem_score * 0.1

            ranked.append({
                "candidate_name": meta['candidate_name'],
                "resume_path": meta.get('resume_path', ''),
                "match_score": round(final_score, 1),
                "matched_skills": matched_must,
                "missing_skills": [s for s in jd.must_have_skills if s not in cand_skills],
                "relevant_excerpts": [cand['content'][:300]],
                "reasoning": f"Skills: {len(matched_must)}/{len(jd.must_have_skills)} | "
                             f"Exp: {exp_years}/{required_exp} yrs | Semantic: {sem_score:.0f}%"
            })

        ranked.sort(key=lambda x: x['match_score'], reverse=True)
        return ranked[:10]

    def match_job(self, jd_text: str, top_k: int = 10) -> Dict:
        jd = self.parse_job_description(jd_text)
        candidates = self.semantic_search_candidates(jd, top_k=20)
        top_matches = self.rank_candidates(candidates, jd)
        return {
            "job_description": jd_text[:500],
            "top_matches": top_matches[:top_k],
            "requirements": {
                "must_have_skills": jd.must_have_skills,
                "nice_to_have_skills": jd.nice_to_have_skills,
                "min_experience_years": jd.min_experience_years,
                "education_requirement": jd.education_requirement
            }
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            job_desc = f.read()
        matcher = JobMatcher()
        result = matcher.match_job(job_desc)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Usage: python job_matcher.py path/to/job_description.txt")