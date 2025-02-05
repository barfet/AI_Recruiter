### User Story: Smart Interview Strategy Generator

````markdown
As a technical recruiter
I want an AI-generated interview strategy based on skill gap analysis
So that I can conduct more efficient and focused technical interviews

Business Value:
- Reduces interview time by focusing on critical skill gaps
- Improves interview effectiveness
- Standardizes the interview process
````

### Acceptance Criteria

````markdown
GIVEN a candidate's skill match analysis
WHEN I request an interview strategy
THEN I receive a structured interview plan with:
- Prioritized interview phases
- Time allocation for each phase
- Specific technical questions for each skill gap
- Focus areas based on match confidence scores

The strategy should:
1. Prioritize missing critical skills first
2. Include validation of weak skill matches (similarity < 0.7)
3. Suggest time allocations based on skill importance
4. Provide 2-3 technical questions per skill
5. Return results in structured JSON format
````

### Example Input/Output

Input:
````json
{
    "match_analysis": {
        "missing_skills": ["Kubernetes", "AWS Lambda"],
        "semantic_matches": [
            ["Python", "Python Programming", 0.65],
            ["Docker", "Container Development", 0.85]
        ],
        "job_requirements": {
            "title": "Senior Backend Engineer",
            "required_skills": ["Python", "Kubernetes", "AWS Lambda", "Docker"]
        }
    }
}
````

Expected Output:
````json
{
    "interview_strategy": {
        "total_time": 60,
        "phases": [
            {
                "priority": 1,
                "skill_focus": "Kubernetes",
                "reason": "Critical missing skill",
                "time_allocation": 20,
                "questions": [
                    {
                        "text": "Explain Kubernetes pod lifecycle and how it handles container failures",
                        "difficulty": "medium",
                        "expected_signals": ["Understanding of k8s concepts", "Experience with container orchestration"]
                    },
                    {
                        "text": "How would you design a highly available application using Kubernetes?",
                        "difficulty": "hard",
                        "expected_signals": ["Architecture knowledge", "Production experience"]
                    }
                ]
            },
            {
                "priority": 2,
                "skill_focus": "AWS Lambda",
                "reason": "Critical missing skill",
                "time_allocation": 15,
                "questions": [
                    {
                        "text": "Describe your experience with serverless architectures and AWS Lambda",
                        "difficulty": "medium",
                        "expected_signals": ["Serverless understanding", "AWS knowledge"]
                    }
                ]
            },
            {
                "priority": 3,
                "skill_focus": "Python",
                "reason": "Weak semantic match (0.65)",
                "time_allocation": 15,
                "questions": [
                    {
                        "text": "Explain Python's async/await and when you'd use it",
                        "difficulty": "medium",
                        "expected_signals": ["Advanced Python knowledge", "Async programming"]
                    }
                ]
            }
        ],
        "notes": "Focus on practical experience validation. Candidate shows weak match in Python - verify depth of knowledge."
    }
}
````

### Test Cases

````python
@pytest.mark.integration
class TestInterviewStrategy:
    """Test suite for interview strategy generation."""
    
    async def test_strategy_generation(self):
        """Test generation of prioritized interview strategy."""
        # Setup test data
        match_analysis = {
            "match_analysis": {
                "missing_skills": ["Kubernetes", "AWS Lambda"],
                "semantic_matches": [
                    ("Python", "Python Programming", 0.65),
                    ("Docker", "Container Development", 0.85)
                ],
                "job_requirements": {
                    "title": "Senior Backend Engineer",
                    "required_skills": ["Python", "Kubernetes", "AWS Lambda", "Docker"]
                }
            }
        }
        
        # Generate strategy
        strategy_generator = InterviewStrategyGenerator()
        strategy = await strategy_generator.create_strategy(match_analysis)
        
        # Basic structure tests
        assert "interview_strategy" in strategy
        assert "phases" in strategy["interview_strategy"]
        assert "total_time" in strategy["interview_strategy"]
        
        # Verify phases
        phases = strategy["interview_strategy"]["phases"]
        assert len(phases) > 0
        
        # Test first phase (highest priority)
        first_phase = phases[0]
        assert first_phase["priority"] == 1
        assert first_phase["skill_focus"] in match_analysis["match_analysis"]["missing_skills"]
        assert "time_allocation" in first_phase
        assert "questions" in first_phase
        assert len(first_phase["questions"]) >= 1
        
        # Test question structure
        for phase in phases:
            for question in phase["questions"]:
                assert "text" in question
                assert "difficulty" in question
                assert "expected_signals" in question
                
        # Test time allocations
        total_time = sum(phase["time_allocation"] for phase in phases)
        assert total_time == strategy["interview_strategy"]["total_time"]
        
        # Test prioritization
        priorities = [phase["priority"] for phase in phases]
        assert priorities == sorted(priorities)  # Verify ordered by priority
````

### Implementation Outline

````python
from typing import Dict, List
from dataclasses import dataclass
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class InterviewStrategyGenerator:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name)
        self.question_generator = self.agent.generate_interview_questions
        
    async def create_strategy(self, match_analysis: Dict) -> Dict:
        """Create a prioritized interview strategy."""
        # Implementation during live demo
        pass
````

### Technical Context & Available Components

```markdown
Existing Core Components:

1. Job/Candidate Matching System
   - JobDiscoveryService
     - search_jobs(query: str) -> List[Dict]
     - find_matching_candidates(job_id: str) -> List[Dict]
     - get_match_analysis(job_id: str, candidate_id: str) -> Dict
     - _are_skills_semantically_similar(skill1: str, skill2: str) -> bool

2. Interview Management
   - RecruitingAgent
     - generate_interview_questions(job_id: str, question_types: List[str]) -> List[Dict]
     - evaluate_response(job_id: str, question: str, response: str) -> Dict
     - generate_interview_feedback(interview_data: Dict) -> Dict

3. Vector Storage
   - ChromaStore
     - get_job_by_id(job_id: str) -> Dict
     - get_candidate_by_id(candidate_id: str) -> Dict
     - similarity_search(query: str, k: int) -> List[Dict]

4. Skill Analysis
   - SkillAnalysisTool
     - calculate_similarity(skill1: str, skill2: str) -> float
     - get_skill_matches(required_skills: List[str], candidate_skills: List[str]) -> Dict
```

### User Story: Smart Interview Strategy Generator

```markdown
As a technical recruiter
I want an AI-generated interview strategy based on skill gap analysis
So that I can conduct more efficient and focused technical interviews

Context:
- We already have skill gap analysis from JobDiscoveryService.get_match_analysis()
- We can generate questions using RecruitingAgent.generate_interview_questions()
- We have skill similarity checking via SkillAnalysisTool.calculate_similarity()
- We store all data in ChromaStore with vector search capabilities

Technical Requirements:
1. Use existing match_analysis output from JobDiscoveryService
2. Leverage existing question generation from RecruitingAgent
3. Utilize skill similarity scores from SkillAnalysisTool
4. Return prioritized interview strategy
```

### Integration Points

```python
# Existing match analysis output structure:
match_analysis = {
    "match_analysis": {
        "skill_match_score": float,
        "semantic_match_score": float,
        "combined_score": float,
        "matching_skills": List[str],
        "semantic_matches": List[Tuple[str, str, float]],
        "missing_skills": List[str],
        "additional_skills": List[str]
    },
    "candidate_info": Dict,
    "job_info": Dict
}

# Existing question generation output:
questions = [
    {
        "type": str,  # "technical" or "behavioral"
        "question": str,
        "expected_signals": List[str],
        "difficulty": str
    }
]

# Existing response evaluation output:
evaluation = {
    "score": float,
    "strengths": List[str],
    "improvements": List[str],
    "technical_accuracy": float,
    "communication": float
}
```

[Previous test cases and implementation outline remain the same...]

### Development Approach

1. **Use Existing Data Flow**:
   ```python
   # 1. Get match analysis
   match_analysis = await job_service.get_match_analysis(job_id, candidate_id)
   
   # 2. For each skill gap, generate questions
   questions = await agent.generate_interview_questions(
       job_id=job_id,
       question_types=["technical"]
   )
   
   # 3. Use skill similarity for prioritization
   similarity_score = skill_tool.calculate_similarity(skill1, skill2)
   ```

2. **New Components Needed**:
   - Priority calculation logic
   - Time allocation algorithm
   - Strategy organization logic

3. **Integration Points**:
   - Input: Use existing match_analysis structure
   - Processing: Use existing question generation
   - Output: New strategy format

