# AI-Powered Recruitment Platform

An intelligent recruitment platform that leverages LLMs and vector search for advanced candidate-job matching and skill analysis.

## 🎯 Core Features

- **Intelligent Matching**: Advanced semantic matching between job requirements and candidate profiles
- **Skill Analysis**: Deep skill gap analysis and recommendations
- **Interview Support**: Dynamic interview question generation and evaluation
- **Vector Search**: Efficient similarity search for candidates and jobs
- **Compliance Management**: Automated compliance checking and reporting

## 🏗 Architecture

```
src/
├── agent/                 # AI agent components
│   ├── chains.py         # LangChain processing chains
│   ├── tools.py          # Specialized AI tools
│   └── prompts.py        # Prompt templates
├── api/                   # API endpoints
│   └── routes/           # Route definitions
├── core/                 # Core functionality
│   ├── config.py         # Configuration management
│   └── logging.py        # Logging setup
├── data/                 # Data management
│   ├── managers/         # Data access layer
│   └── cleaning/         # Data cleaning utilities
├── services/            # Business logic
│   └── job_discovery.py  # Job matching service
└── vector_store/        # Vector database interface
    └── chroma_store.py   # ChromaDB implementation
```

## 🛠 Tech Stack

- **Python 3.9+**: Core language
- **LangChain**: LLM orchestration
- **OpenAI GPT-4**: Language model
- **ChromaDB**: Vector storage
- **FastAPI**: API framework
- **Pydantic**: Data validation
- **pytest**: Testing framework

## 🚀 Getting Started

1. **Environment Setup**
```bash
python -m venv venv
source venv/bin/activate  # Unix
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. **Configuration**
```bash
cp .env.example .env
# Edit .env with your settings
```

3. **Run Tests**
```bash
pytest
```

4. **Start Server**
```bash
uvicorn src.api.main:app --reload
```

## 📋 Development Guidelines

### Code Structure

- **Agents**: Inherit from `BaseAgent`, implement `_run` method
- **Chains**: Follow `CandidateJobMatchChain` pattern
- **Tools**: Inherit from `BaseTool`, implement required methods
- **Managers**: Follow `BaseManager` pattern with logging

### AI Components

- **Prompts**: 
  - Include examples and context
  - Validate outputs
  - Handle edge cases
- **LLM Calls**:
  - Set appropriate temperature
  - Implement timeout handling
  - Include error recovery
- **Embeddings**:
  - Specify model and dimensions
  - Implement batch processing
  - Handle OOV tokens

### Testing

- Unit tests for all components
- Integration tests for chains
- Mock external services
- Test async functionality
- Validate outputs

## 🔍 Key Patterns

### Chain Pattern
```python
class CustomChain:
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(temperature=0.0)
        self._init_chains()

    def _init_chains(self):
        # Initialize component chains
        pass

    async def run(self, **kwargs):
        # Chain execution logic
        pass
```

### Manager Pattern
```python
class CustomManager(BaseManager):
    def __init__(self):
        super().__init__()
        self.logger = setup_logger(__name__)

    async def process_data(self):
        try:
            # Processing logic
            pass
        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
            raise
```

## 🤝 Contributing

1. Follow the `.cursorrules` configuration
2. Ensure comprehensive documentation
3. Add tests for new features
4. Update relevant documentation

## 📚 Documentation Standards

- Google-style docstrings
- Include Args, Returns, Raises sections
- Provide usage examples
- Document class attributes and methods

## 🔐 Security

- Implement rate limiting
- Validate all inputs
- Sanitize LLM outputs
- Handle sensitive data appropriately

## 📈 Performance

- Use async/await for I/O operations
- Implement caching where appropriate
- Batch process embeddings
- Monitor LLM token usage

## 📝 License

MIT License - see LICENSE file for details

## 🔗 Links

- [Technical Architecture](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Development Guide](docs/development.md)
