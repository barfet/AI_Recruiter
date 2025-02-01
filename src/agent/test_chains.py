"""Tests for multi-step chains"""

import asyncio
from src.agent.chains import CandidateJobMatchChain, InterviewWorkflowChain
from src.agent.prompts import create_skill_analysis_prompt
from src.core.logging import setup_logger

logger = setup_logger(__name__)


async def test_match_chain():
    """Test the candidate-job match chain with dynamic prompts"""
    chain = CandidateJobMatchChain()
    
    # Test data
    job_info = """
    Title: Senior Software Engineer
    Company: TechCorp
    Location: San Francisco, CA
    Description: We are looking for a senior software engineer with strong experience in Python, AWS, and microservices.
    Requirements:
    - 5+ years of Python development
    - Experience with AWS services
    - Strong understanding of microservices architecture
    - Experience with CI/CD pipelines
    Skills: Python, AWS, Docker, Kubernetes, CI/CD
    """
    
    candidate_info = """
    Name: John Doe
    Experience:
    - Software Engineer at ABC Corp (3 years)
      - Developed microservices using Python and AWS
      - Implemented CI/CD pipelines
    - Junior Developer at XYZ Inc (2 years)
      - Python development
      - Docker containerization
    Skills: Python, AWS, Docker, Git, REST APIs
    """
    
    try:
        # Test with technical role context
        logger.info("\n=== Testing Match Chain (Technical Role) ===")
        result = await chain.run(candidate_info=candidate_info, job_info=job_info)
        logger.info("1. Candidate Summary:")
        logger.info(result["candidate_summary"])
        logger.info("\n2. Job Analysis:")
        logger.info(result["job_analysis"])
        logger.info("\n3. Skills Gap Analysis:")
        logger.info(result["skills_gap_analysis"])
        logger.info("\n4. Interview Strategy:")
        logger.info(result["interview_strategy"])
        
        # Test with different focus areas
        logger.info("\n=== Testing Match Chain (Leadership Focus) ===")
        chain.job_analysis_chain = create_skill_analysis_prompt(
            focus_areas=["Leadership", "Team Fit", "Technical Skills"]
        ) | chain.llm
        result = await chain.run(candidate_info=candidate_info, job_info=job_info)
        logger.info(result["job_analysis"])
    except Exception as e:
        logger.error(f"Error in match chain test: {str(e)}")


async def test_interview_chain():
    """Test the interview workflow chain with dynamic difficulty"""
    chain = InterviewWorkflowChain()
    
    # Test data
    job_info = """
    Title: Senior Software Engineer
    Requirements:
    - Strong Python development skills
    - AWS cloud services experience
    - System design and architecture
    - Team leadership experience
    """
    
    candidate_info = """
    Experience: 5 years in software development
    Skills: Python, AWS, Docker
    Leadership: Led a team of 3 developers
    """
    
    focus_areas = [
        "Technical Skills",
        "System Design",
        "Leadership",
        "Problem Solving"
    ]
    
    try:
        # Test question generation with different difficulties
        logger.info("\n=== Testing Interview Questions Generation ===")
        for difficulty in ["basic", "intermediate", "advanced"]:
            questions = await chain.generate_questions(
                job_info=job_info,
                candidate_info=candidate_info,
                focus_areas=focus_areas,
                difficulty=difficulty
            )
            logger.info(f"\nQuestions ({difficulty} difficulty):")
            logger.info(questions["interview_questions"])
        
        # Test response evaluation with custom focus areas
        logger.info("\n=== Testing Response Evaluation ===")
        sample_qa = {
            "question": "Describe a complex system you designed using AWS services.",
            "answer": """I designed a scalable microservices architecture using AWS ECS, 
            API Gateway, and Lambda. The system handled 1M+ daily requests and included 
            automatic scaling and fault tolerance.""",
            "job_requirements": "Experience with AWS and system design",
            "focus_areas": ["Technical Depth", "Architecture Skills", "Scale Handling"]
        }
        
        evaluation = await chain.evaluate_response(**sample_qa)
        logger.info("\nResponse Evaluation:")
        logger.info(evaluation["response_evaluation"])
        
        # Test feedback generation with different focus areas
        logger.info("\n=== Testing Feedback Generation ===")
        feedback = await chain.generate_feedback(
            [evaluation],
            focus_areas=["Technical Excellence", "System Design", "Communication"]
        )
        logger.info("\nGenerated Feedback:")
        logger.info(feedback["feedback"])
    except Exception as e:
        logger.error(f"Error in interview chain test: {str(e)}")


async def main():
    """Run all chain tests"""
    logger.info("Starting chain tests with advanced prompts...")
    
    try:
        await test_match_chain()
        await test_interview_chain()
    except Exception as e:
        logger.error(f"Error in tests: {str(e)}")
    
    logger.info("Chain tests completed")


if __name__ == "__main__":
    asyncio.run(main()) 