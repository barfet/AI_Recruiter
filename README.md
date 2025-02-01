# Aridmi - AI-Powered Recruitment Platform

## Overview
Aridmi is an advanced AI-powered recruitment platform that streamlines the hiring process through intelligent candidate matching, automated skill assessment, and comprehensive interview management.

## Current Status
We have completed Phase 1 of development, focusing on core AI capabilities:
- ✅ Candidate-Job matching using LangChain and OpenAI
- ✅ Skill gap analysis
- ✅ Interview question generation
- ✅ Response evaluation and feedback

## Key Features

### 1. Intelligent Matching
- Semantic understanding of job requirements
- Advanced candidate skill analysis
- Multi-dimensional scoring system
- Contextual matching algorithms

### 2. Interview Management
- Automated question generation
- Technical and behavioral assessment
- Real-time evaluation
- Structured feedback system

### 3. Workflow Automation
- End-to-end process management
- Status tracking
- Notification system
- Calendar integration

### 4. Analytics & Insights
- Performance metrics
- Process optimization
- Predictive analytics
- Compliance monitoring

## Technical Stack

### Core Technologies
- Python 3.9+
- LangChain
- OpenAI GPT-4
- Vector Search (Coming Soon)
- PostgreSQL (Coming Soon)

### Architecture
- Microservices-based
- Event-driven
- Scalable infrastructure
- Security-first design

## Getting Started

### Prerequisites
```bash
python -m venv venv
source venv/bin/activate  # Unix
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Configuration
```python
# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key'
```

### Running Tests
```bash
PYTHONPATH=$PYTHONPATH:. python src/agent/test_chains.py
```

## Documentation
- [Technical Architecture](domain/architecture.md)
- [Product Vision](domain/vision.md)
- [Use Cases](domain/use_cases.md)
- [Technical Roadmap](domain/roadmap.md)

## Next Steps
We are currently moving into Phase 2, which focuses on:
1. Vector search implementation
2. Data model development
3. Caching layer setup
4. API development

See our [Technical Roadmap](domain/roadmap.md) for detailed plans.

## Contributing
We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or feedback, please open an issue or contact the maintainers.
