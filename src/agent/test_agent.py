import asyncio
import json
import re
from src.agent.agent import RecruitingAgent
from src.core.logging import setup_logger

logger = setup_logger(__name__)


def extract_id_from_text(text: str, id_type: str) -> str:
    """Extract job_id or resume_id from text using regex"""
    if id_type == "job":
        pattern = r'\[Job ID: (\d+)\]'
    else:  # resume
        pattern = r'\[Resume ID: (\d+)\]'
    
    match = re.search(pattern, text)
    return match.group(1) if match else None


def parse_response(response: str) -> dict:
    """Parse agent response, handling both JSON and text formats"""
    try:
        # Try parsing as JSON first
        return json.loads(response)
    except json.JSONDecodeError:
        # If not JSON, try extracting IDs from text
        result = {}
        
        # Try to find job_id
        job_id = extract_id_from_text(response, "job")
        if job_id:
            result["job_id"] = job_id
            
        # Try to find resume_id
        resume_id = extract_id_from_text(response, "resume")
        if resume_id:
            result["resume_id"] = resume_id
            
        if not result:
            logger.warning(f"Could not parse response: {response}")
            # Try to extract any numeric ID as fallback
            fallback_match = re.search(r'(?:job|resume).*?(\d+)', response.lower())
            if fallback_match:
                id_val = fallback_match.group(1)
                if 'job' in response.lower():
                    result["job_id"] = id_val
                else:
                    result["resume_id"] = id_val
            
        return result


async def test_skill_analysis():
    """Test the skill analysis functionality"""
    agent = RecruitingAgent(temperature=0.3)
    
    # First, find a job and a candidate
    job_response = await agent.run("Find me a senior software engineer position")
    job_data = parse_response(job_response)
    job_id = job_data.get("job_id", "3904095574")  # Fallback to a known job ID
    
    candidate_response = await agent.run("Find a candidate with Python and AWS experience")
    candidate_data = parse_response(candidate_response)
    resume_id = candidate_data.get("resume_id", "23408537")  # Fallback to a known resume ID
    
    # Test skill analysis
    query = f"Analyze the skill match between job {job_id} and candidate {resume_id}"
    response = await agent.run(query)
    logger.info("\nSkill Analysis Test:")
    logger.info(response)


async def test_interview_questions():
    """Test the interview question generation"""
    agent = RecruitingAgent(temperature=0.3)
    
    # Find a job
    job_response = await agent.run("Find me a data scientist position")
    job_data = parse_response(job_response)
    job_id = job_data.get("job_id", "3899540393")  # Fallback to a known job ID
    
    # Generate questions
    query = f"Generate interview questions for job {job_id}"
    response = await agent.run(query)
    logger.info("\nInterview Questions Test:")
    logger.info(response)


async def main():
    """Run all tests"""
    logger.info("Starting agent tests...")
    
    try:
        await test_skill_analysis()
        await test_interview_questions()
    except Exception as e:
        logger.error(f"Error in tests: {str(e)}")
        logger.exception("Full traceback:")
    
    logger.info("Tests completed")


if __name__ == "__main__":
    asyncio.run(main()) 