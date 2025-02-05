"""Interview tools for the recruiting agent."""
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from pydantic import Field
import json

from src.agent.tools.base import BaseRecruitingTool, BaseChainTool
from src.agent.models.inputs import (
    InterviewQuestionInput,
    ResponseEvaluationInput,
    InterviewFeedbackInput
)
from src.agent.models.outputs import (
    StandardizedOutput,
    InterviewQuestionOutput,
    InterviewFeedbackOutput
)
from src.agent.tools.analysis import SkillAnalysisTool
from src.agent.tools.search import JobSearchTool
from src.data.models.interview import (
    InterviewQuestion,
    ResponseEvaluation,
    InterviewFeedback
)
from src.core.config import settings


# Templates
QUESTION_TEMPLATE = """Generate {difficulty} level interview questions for assessing {skill_focus} expertise.
Focus on practical, real-world scenarios that demonstrate both theoretical knowledge and hands-on experience.

Job Context:
{job_context}

Generate {question_count} questions that:
1. Are specific to the job requirements
2. Test both theoretical understanding and practical application
3. Allow for follow-up discussion
4. Have clear evaluation criteria

For each question, provide:
1. The main question text
2. Expected signals in the answer (key points that demonstrate understanding)
3. Question type (technical or behavioral)
4. Difficulty level

Format as JSON array with fields:
- question: string
- type: "technical" or "behavioral"
- difficulty: "easy", "medium", or "hard"
- expected_signals: array of strings"""

EVALUATION_TEMPLATE = """Evaluate the candidate's response based on the following context:

Job Requirements:
{job_requirements}

Skill Focus: {skill_focus}

Question:
{question}

Candidate Response:
{response}

Evaluate the response considering:
1. Technical accuracy and depth
2. Problem-solving approach
3. Communication clarity
4. Practical experience demonstrated
5. Understanding of best practices

Provide a detailed evaluation with:
1. Score (0-100)
2. Key strengths demonstrated
3. Areas for improvement
4. Technical accuracy assessment
5. Communication effectiveness

Format the evaluation as a JSON object with these exact fields:
- score: Numerical score 0-100
- strengths: List of key strengths
- improvements: List of areas to improve
- technical_accuracy: Score 0-100
- communication_score: Score 0-100
- notes: Additional observations
"""

FEEDBACK_TEMPLATE = """Generate comprehensive interview feedback based on:

Job Requirements:
{job_requirements}

Interview Responses:
{responses}

Individual Evaluations:
{evaluations}

Additional Notes:
{interview_notes}

Provide detailed feedback including:
1. Overall assessment
2. Technical competency evaluation
3. Communication skills assessment
4. Key strengths demonstrated
5. Areas for development
6. Hiring recommendation

Consider:
- Match with job requirements
- Technical proficiency
- Problem-solving ability
- Communication effectiveness
- Cultural fit indicators

Format the feedback as a JSON object with these exact fields:
- overall_score: Overall interview score (0-100)
- technical_score: Technical competency score (0-100)
- communication_score: Communication effectiveness score (0-100)
- strengths: List of key strengths
- improvements: List of areas for improvement
- recommendation: Clear hiring recommendation
- notes: Additional observations
"""


class InterviewQuestionTool(BaseChainTool):
    """Tool for generating interview questions."""
    
    name: str = "generate_questions"
    description: str = "Generate interview questions based on job requirements"
    job_search: JobSearchTool = Field(default_factory=JobSearchTool)
    args_schema: type[InterviewQuestionInput] = InterviewQuestionInput
    
    def __init__(self):
        """Initialize the interview question tool."""
        super().__init__(
            chain_template=QUESTION_TEMPLATE,
            input_variables=[
                "difficulty",
                "skill_focus",
                "job_context",
                "question_count"
            ],
            llm=ChatOpenAI(**settings.LLM_CONFIG)
        )
    
    async def _arun(
        self,
        job_id: str,
        skill_focus: str,
        difficulty: str = "medium",
        question_count: int = 3,
        question_type: str = "technical",
        **kwargs
    ) -> str:
        """Generate interview questions."""
        try:
            # Get job data
            job_result = await self.job_search._arun({"query": f"id:{job_id}", "limit": 1})
            job_data = StandardizedOutput.parse_raw(job_result)
            
            if job_data.status == "error" or not job_data.data:
                return StandardizedOutput(
                    status="error",
                    error=f"Job not found: {job_id}"
                ).json()
            
            # Format job context
            job = job_data.data[0] if isinstance(job_data.data, list) else job_data.data
            job_context = f"""
            Title: {job.get('title', '')}
            Description: {job.get('description', '')}
            Required Skills: {', '.join(job.get('skills', []))}
            Focus Skill: {skill_focus}
            """
            
            # Generate questions
            result = await self.chain.ainvoke({
                "job_context": job_context,
                "skill_focus": skill_focus,
                "difficulty": difficulty,
                "question_count": question_count
            })
            
            # Parse and validate questions
            try:
                raw_questions = json.loads(result["text"])
                if not isinstance(raw_questions, list):
                    raw_questions = [raw_questions]
                
                # Validate each question using our model
                validated_questions = []
                for q in raw_questions:
                    try:
                        # Create InterviewQuestion model instance
                        question = InterviewQuestion(
                            text=q.get("question", ""),
                            difficulty=q.get("difficulty", difficulty),
                            expected_signals=q.get("expected_signals", [])
                        )
                        
                        # Convert to dict and add type
                        question_dict = question.model_dump()
                        question_dict.update({
                            "type": q.get("type", question_type),
                            "question": question_dict.pop("text")  # Rename text to question for API consistency
                        })
                        validated_questions.append(question_dict)
                        
                    except Exception as e:
                        self.logger.warning(f"Invalid question format: {str(e)}")
                        continue
                
                if not validated_questions:
                    return StandardizedOutput(
                        status="error",
                        error="No valid questions were generated"
                    ).json()
                
                return StandardizedOutput(
                    status="success",
                    data=validated_questions
                ).json()
                
            except json.JSONDecodeError as e:
                return StandardizedOutput(
                    status="error",
                    error=f"Invalid JSON response from LLM: {str(e)}"
                ).json()
                
        except Exception as e:
            self.logger.error(f"Error generating questions: {str(e)}")
            return StandardizedOutput(
                status="error",
                error=str(e)
            ).json()


class ResponseEvaluationTool(BaseChainTool):
    """Tool for evaluating interview responses."""
    
    name: str = "evaluate_response"
    description: str = "Evaluate candidate responses to interview questions"
    args_schema: type[ResponseEvaluationInput] = ResponseEvaluationInput
    job_search: JobSearchTool = Field(default_factory=JobSearchTool)
    
    def __init__(self):
        """Initialize the response evaluation tool."""
        super().__init__(
            chain_template=EVALUATION_TEMPLATE,
            input_variables=[
                "job_requirements",
                "skill_focus",
                "question",
                "response"
            ],
            llm=ChatOpenAI(**settings.LLM_CONFIG)
        )

    async def _arun(
        self,
        job_id: str,
        resume_id: str,
        question: str,
        response: str,
        skill_focus: str,
        **kwargs
    ) -> str:
        """Evaluate an interview response."""
        try:
            # Get job requirements
            job_result = await self.job_search._arun(f"job_id:{job_id}", limit=1)
            job_data = StandardizedOutput.parse_raw(job_result).data[0]
            
            # Run evaluation
            result = await self._run_chain(
                job_requirements=job_data.get("requirements", ""),
                skill_focus=skill_focus,
                question=question,
                response=response
            )
            
            evaluation_data = StandardizedOutput.parse_raw(result).data
            
            return StandardizedOutput(
                status="success",
                data=evaluation_data,
                metadata={
                    "job_id": job_id,
                    "resume_id": resume_id,
                    "skill_focus": skill_focus
                }
            ).to_json()
            
        except Exception as e:
            self.logger.error(f"Error evaluating response: {str(e)}")
            return StandardizedOutput(
                status="error",
                error=str(e)
            ).to_json()


class FeedbackGenerationTool(BaseChainTool):
    """Tool for generating comprehensive interview feedback."""
    
    name: str = "generate_feedback"
    description: str = "Generate comprehensive interview feedback"
    args_schema: type[InterviewFeedbackInput] = InterviewFeedbackInput
    job_search: JobSearchTool = Field(default_factory=JobSearchTool)
    
    def __init__(self):
        """Initialize the feedback generation tool."""
        super().__init__(
            chain_template=FEEDBACK_TEMPLATE,
            input_variables=[
                "job_requirements",
                "responses",
                "evaluations",
                "interview_notes"
            ],
            llm=ChatOpenAI(**settings.LLM_CONFIG)
        )

    async def _arun(
        self,
        job_id: str,
        resume_id: str,
        responses: Dict[str, str],
        evaluations: Dict[str, Dict],
        interview_notes: str = None,
        **kwargs
    ) -> str:
        """Generate interview feedback."""
        try:
            # Get job requirements
            job_result = await self.job_search._arun(f"job_id:{job_id}", limit=1)
            job_data = StandardizedOutput.parse_raw(job_result).data[0]
            
            # Generate feedback
            result = await self._run_chain(
                job_requirements=job_data.get("requirements", ""),
                responses=responses,
                evaluations=evaluations,
                interview_notes=interview_notes or ""
            )
            
            feedback_data = StandardizedOutput.parse_raw(result).data
            feedback = InterviewFeedbackOutput(
                overall_score=feedback_data["overall_score"],
                technical_score=feedback_data["technical_score"],
                communication_score=feedback_data["communication_score"],
                strengths=feedback_data["strengths"],
                improvements=feedback_data["improvements"],
                recommendation=feedback_data["recommendation"],
                notes=feedback_data.get("notes")
            )
            
            return StandardizedOutput(
                status="success",
                data=feedback.model_dump(),
                metadata={
                    "job_id": job_id,
                    "resume_id": resume_id,
                    "num_questions": len(responses)
                }
            ).to_json()
            
        except Exception as e:
            self.logger.error(f"Error generating feedback: {str(e)}")
            return StandardizedOutput(
                status="error",
                error=str(e)
            ).to_json() 