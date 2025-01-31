import uvicorn
from src.core.config import settings
from src.core.logging import setup_logger

logger = setup_logger(__name__)

def main():
    """Run the FastAPI application"""
    try:
        uvicorn.run(
            "src.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )
    except Exception as e:
        logger.error(f"Error running application: {str(e)}")
        raise

if __name__ == "__main__":
    main() 