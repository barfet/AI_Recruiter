import asyncio
import typer
from typing import Optional

from src.core.logging import setup_logger
from src.agent.agent import RecruitingAgent
from src.data.managers.job import JobManager
from src.data.managers.candidate import CandidateManager
from src.embeddings.manager import EmbeddingManager
from src.embeddings.test_search import test_job_search, test_candidate_search, test_cross_matching

logger = setup_logger(__name__)
app = typer.Typer()

@app.command()
def process_data(
    jobs_dir: Optional[str] = typer.Option(None, help="Directory containing job posting files"),
    resumes_dir: Optional[str] = typer.Option(None, help="Directory containing resume files"),
    force: bool = typer.Option(False, help="Force reprocessing of data")
):
    """Process job postings and resumes"""
    try:
        if jobs_dir:
            logger.info("Processing job postings...")
            job_manager = JobManager()
            job_manager.process_jobs(jobs_dir, force=force)
            
        if resumes_dir:
            logger.info("Processing resumes...")
            candidate_manager = CandidateManager()
            candidate_manager.process_candidates(resumes_dir, force=force)
            
        logger.info("Data processing complete")
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise typer.Exit(1)

@app.command()
def create_embeddings(
    data_type: str = typer.Option(..., help="Type of data to create embeddings for (jobs/candidates)"),
    force: bool = typer.Option(False, help="Force recreation of embeddings"),
    batch_size: int = typer.Option(50, help="Number of items to process in each batch")
):
    """Create embeddings for processed data"""
    try:
        logger.info(f"Creating embeddings for {data_type}...")
        embedding_manager = EmbeddingManager()
        
        if data_type == "jobs":
            embedding_manager.create_job_embeddings(force=force, batch_size=batch_size)
        elif data_type == "candidates":
            embedding_manager.create_candidate_embeddings(force=force, batch_size=batch_size)
        else:
            logger.error(f"Invalid data type: {data_type}")
            raise typer.Exit(1)
            
        logger.info("Embeddings creation complete")
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise typer.Exit(1)

@app.command()
async def chat(
    temperature: float = typer.Option(0.7, help="Temperature for the LLM")
):
    """Start an interactive chat session with the recruiting agent"""
    try:
        agent = RecruitingAgent(temperature=temperature)
        logger.info("Starting chat session...")
        
        while True:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit", "bye"]:
                break
                
            response = await agent.run(query)
            print(f"\nAssistant: {response}")
            
    except KeyboardInterrupt:
        logger.info("\nChat session ended by user")
    except Exception as e:
        logger.error(f"Error in chat session: {str(e)}")
        raise typer.Exit(1)

@app.command()
def test_search(
    test_type: str = typer.Option(..., help="Type of test to run (jobs/candidates/matching/all)")
):
    """Run semantic search tests"""
    try:
        if test_type == "jobs" or test_type == "all":
            test_job_search()
        if test_type == "candidates" or test_type == "all":
            test_candidate_search()
        if test_type == "matching" or test_type == "all":
            test_cross_matching()
            
        logger.info("Search tests completed")
    except Exception as e:
        logger.error(f"Error running search tests: {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 