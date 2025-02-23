"""Main CLI interface for the AI Recruiter system."""

import typer
from src.core.logging import setup_logger
from src.agent.agent import RecruitingAgent
from src.data.process import (
    process_jobs,
    process_candidates,
    create_embeddings,
    process_all
)

logger = setup_logger(__name__)
app = typer.Typer()

@app.command()
def process(
    data_type: str = typer.Argument(
        ...,
        help="Type of data to process (jobs/candidates/all)"
    ),
    force: bool = typer.Option(
        False,
        help="Force reprocessing of data"
    ),
    with_embeddings: bool = typer.Option(
        False,
        help="Also create embeddings after processing"
    ),
    batch_size: int = typer.Option(
        50,
        help="Batch size for embedding creation"
    )
):
    """Process raw data files into structured format and optionally create embeddings."""
    try:
        if data_type == "all":
            process_all(
                force_clean=force,
                force_embeddings=with_embeddings,
                batch_size=batch_size
            )
        elif data_type == "jobs":
            processed = process_jobs(force=force)
            if with_embeddings:
                create_embeddings("jobs", force=True, batch_size=batch_size)
        elif data_type == "candidates":
            processed = process_candidates(force=force)
            if with_embeddings:
                create_embeddings("candidates", force=True, batch_size=batch_size)
        else:
            logger.error(f"Unknown data type: {data_type}")
            raise typer.Exit(1)
            
        logger.info("Processing complete")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise typer.Exit(1)

@app.command()
def embeddings(
    data_type: str = typer.Argument(
        ...,
        help="Type of data to create embeddings for (jobs/candidates/all)"
    ),
    force: bool = typer.Option(
        False,
        help="Force recreation of embeddings"
    ),
    batch_size: int = typer.Option(
        50,
        help="Number of items to process in each batch"
    )
):
    """Create embeddings for processed data."""
    try:
        if data_type == "all":
            create_embeddings("jobs", force=force, batch_size=batch_size)
            create_embeddings("candidates", force=force, batch_size=batch_size)
        elif data_type in ["jobs", "candidates"]:
            create_embeddings(data_type, force=force, batch_size=batch_size)
        else:
            logger.error(f"Invalid data type: {data_type}")
            raise typer.Exit(1)
            
        logger.info("Embeddings creation complete")
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise typer.Exit(1)

@app.command()
async def chat(
    temperature: float = typer.Option(
        0.7,
        help="Temperature for the LLM"
    )
):
    """Start an interactive chat session with the recruiting agent."""
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
    