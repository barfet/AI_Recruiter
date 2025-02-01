from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.core.logging import setup_logger
from src.data import init_data_directories

logger = setup_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    # Initialize data directories
    init_data_directories()

    # Create FastAPI app
    app = FastAPI(
        title="AI Recruiter API",
        description="API for the AI Recruiter application",
        version="1.0.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router, prefix="/api")

    return app


app = create_app()
