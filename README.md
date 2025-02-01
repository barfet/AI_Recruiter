# AI Recruiter

An intelligent recruiting assistant that helps match jobs with candidates using natural language processing and semantic search.

## Features

- Search job postings using natural language queries
- Search candidate profiles using natural language queries
- Match candidates with job postings based on skills, experience, and requirements
- RESTful API for integration with other applications
- Conversation memory for contextual interactions
- Semantic search using OpenAI embeddings and FAISS vector store

## Project Structure

```
ai_recruiter/
├── data/                    # Data storage
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed data files
│   └── indexes/            # Vector store indexes
├── logs/                   # Application logs
├── src/                    # Source code
│   ├── agent/             # Agent components
│   │   ├── agent.py       # Main agent class
│   │   ├── prompts.py     # Agent prompts
│   │   └── tools.py       # Agent tools
│   ├── api/               # API components
│   │   ├── app.py         # FastAPI application
│   │   ├── models.py      # API models
│   │   └── routes.py      # API routes
│   ├── core/              # Core components
│   │   ├── config.py      # Configuration
│   │   ├── exceptions.py  # Custom exceptions
│   │   └── logging.py     # Logging setup
│   ├── data/              # Data handling
│   │   ├── models/        # Data models
│   │   └── managers/      # Data managers
│   ├── embeddings/        # Embedding components
│   │   ├── manager.py     # Embedding manager
│   │   └── create_embeddings.py
│   └── search/            # Search components
│       └── search_manager.py
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
├── run.py                # Application entry point
└── README.md             # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai_recruiter.git
cd ai_recruiter
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

4. Initialize the data directories and process the data:
```bash
python -m src.data.ingestion
```

5. Create embeddings for jobs and candidates:
```bash
python -m src.embeddings.create_embeddings
```

## Running the Application

Start the FastAPI application:
```bash
python run.py
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### POST /api/query
Process a query using the recruiting agent.

Request body:
```json
{
  "query": "Find software engineering jobs in San Francisco",
  "filters": {
    "experience_level": "Senior",
    "remote_allowed": true
  }
}
```

### POST /api/reset
Reset the agent's conversation memory.

## Development

1. Data Processing:
   - Raw data is stored in `data/raw/`
   - Data is processed and validated using Pydantic models
   - Processed data is stored in `data/processed/`

2. Vector Embeddings:
   - OpenAI embeddings are used for semantic search
   - FAISS is used for efficient similarity search
   - Indexes are stored in `data/indexes/`

3. Agent:
   - Uses LangChain for the agent framework
   - Custom tools for searching jobs and candidates
   - Conversation memory for context

4. API:
   - FastAPI for the REST API
   - Pydantic models for request/response validation
   - CORS middleware for frontend integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License.
