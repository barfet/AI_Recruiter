from pathlib import Path
import os
from dotenv import load_dotenv
from embeddings import EmbeddingManager
from agent_flow import RecruitingAgent

# Load environment variables
load_dotenv()

def setup_embeddings():
    """Set up or load vector embeddings"""
    print("Setting up embeddings...")
    manager = EmbeddingManager()
    
    # Check if embeddings already exist
    index_dir = Path(__file__).parent.parent / "data/indexes"
    jobs_exist = (index_dir / "jobs").exists()
    candidates_exist = (index_dir / "candidates").exists()
    
    if not jobs_exist or not candidates_exist:
        print("Creating new embeddings...")
        if not jobs_exist:
            print("Creating job embeddings...")
            manager.create_job_embeddings()
        if not candidates_exist:
            print("Creating candidate embeddings...")
            manager.create_candidate_embeddings()
    else:
        print("Loading existing embeddings...")
        manager.load_embeddings()
    
    return manager

def initialize_agent() -> RecruitingAgent:
    """Initialize the recruiting agent"""
    print("Initializing recruiting agent...")
    return RecruitingAgent()

def main():
    """Main entry point"""
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Set up embeddings
    embedding_manager = setup_embeddings()
    
    # Initialize agent
    agent = initialize_agent()
    
    # Example queries
    queries = [
        "Find software engineering jobs in San Francisco",
        "Find candidates with experience in machine learning",
        "Match candidates with software engineering jobs in San Francisco"
    ]
    
    # Run example queries
    print("\nRunning example queries:")
    for query in queries:
        print(f"\nQuery: {query}")
        try:
            result = agent.run(query)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main() 