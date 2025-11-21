# Resume Tailor Crew

An enterprise-grade CrewAI system for intelligent resume tailoring based on job descriptions. This system uses specialized AI agents to analyze job requirements, extract resume data, and generate tailored resumes optimized for both Applicant Tracking Systems (ATS) and human reviewers.

## Overview

Resume Tailor Crew automates the labor-intensive process of customizing resumes for different job applications. By leveraging multi-agent AI:

- **Job Analysis Agent**: Extracts requirements, skills, and context from job descriptions
- **Resume Parsing Agent**: Structures existing resume data for comparison
- **Gap Analysis Agent**: Identifies alignment between resume and job requirements
- **Content Generation Agents**: Creates tailored professional summary, experience bullets, and skills
- **Quality Assurance Agents**: Validates ATS compatibility and professional standards

## Features

- Multi-agent workflow for comprehensive resume analysis and tailoring
- ATS-optimized formatting and keyword integration
- Professional summary generation with strategic keyword placement
- Experience section optimization with achievement metrics
- Skills section reordering and prioritization
- Quality assurance and consistency validation
- Enterprise-grade logging and error handling
- Type-safe configurations with Pydantic
- Comprehensive testing framework

## Quick Start

### Prerequisites

- Python 3.10 - 3.12
- UV package manager (or pip)
- OpenAI API key (or alternative LLM provider)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/resume-tailor-crew.git
cd resume-tailor-crew
```

2. Install dependencies using UV:
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp env_example .env
# Edit .env with your API keys and configuration
```

### Basic Usage

```python
from resume_tailor_crew.flows.resume_tailoring_flow import ResumeTailoringFlow

# Initialize the flow
flow = ResumeTailoringFlow()

# Execute resume tailoring
result = flow.kickoff(
    resume_file_path="./path/to/resume.md",
    job_description_file_path="./path/to/job_description.txt"
)

# Result contains tailored resume and analysis
print(result.tailored_resume)
print(result.analysis_report)
```

## Project Structure

```
resume-tailor-crew/
├── src/resume_tailor_crew/
│   ├── core/                 # Core infrastructure
│   │   ├── config.py        # Configuration management
│   │   └── logger.py        # Logging setup
│   ├── models/              # Data models and schemas
│   │   ├── resume.py
│   │   ├── job_description.py
│   │   └── strategy.py
│   ├── tools/               # Custom tools for agents
│   │   ├── resume_parser.py
│   │   ├── job_analyzer.py
│   │   └── document_formatter.py
│   ├── agents/              # CrewAI agents and tasks
│   │   ├── config/
│   │   │   ├── agents.yaml
│   │   │   └── tasks.yaml
│   │   └── crew.py
│   ├── flows/               # Business workflows
│   │   └── resume_tailoring_flow.py
│   ├── utils/               # Utility functions
│   │   ├── file_handlers.py
│   │   └── validators.py
│   └── main.py              # Entry point
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── fixtures/           # Test data
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── examples/               # Example usage
├── pyproject.toml          # Project configuration
└── README.md
```

## Configuration

### Environment Variables

Key environment variables (see `env_example` for complete list):

- `OPENAI_API_KEY` - OpenAI API key for GPT models
- `ANTHROPIC_API_KEY` - Anthropic API key for Claude models
- `ENVIRONMENT` - Deployment environment (development/staging/production)
- `LOG_LEVEL` - Logging verbosity (DEBUG/INFO/WARNING/ERROR)

### Application Config

Configure application behavior via `config.py`:

```python
from resume_tailor_crew.core.config import AppConfig

config = AppConfig()
config.max_resume_iterations  # Max refinement iterations
config.quality_threshold      # Minimum quality score required
config.ats_keyword_density   # Optimal keyword density for ATS
```

## Development

### Setup Development Environment

```bash
# Install with development dependencies
uv sync --extra dev

# Install pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/
mypy src/

# Run tests
pytest tests/ -v --cov=src/resume_tailor_crew

# Full quality check
make quality  # if using Makefile
```

### Testing

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run with coverage
pytest --cov=src/resume_tailor_crew --cov-report=html
```

## Architecture

### Agent Workflow

```
1. Job Description Analyzer
   └─> Structured job requirements

2. Resume Content Extractor  
   └─> Structured resume data

3. Gap & Alignment Strategist
   └─> Strategy document

4. Professional Summary Writer, Experience Optimizer, Skills Strategist (parallel)
   └─> Tailored content

5. ATS Optimizer
   └─> ATS-compliant version

6. Quality Assurance Reviewer
   └─> Final validated resume
```

### Process Flow

- **Sequential Processing**: Tasks execute sequentially with context passing
- **Type-Safe Models**: Pydantic models for all data structures
- **Error Handling**: Comprehensive error handling with recovery mechanisms
- **Logging**: Structured logging for debugging and monitoring

## Documentation

See `docs/` directory for detailed documentation:

- [Architecture](docs/ARCHITECTURE.md) - System design and principles
- [Setup Guide](docs/SETUP.md) - Detailed installation instructions
- [Agent Documentation](docs/AGENTS.md) - Agent descriptions and capabilities
- [API Reference](docs/API.md) - Complete API documentation
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## Usage Examples

See `examples/` directory for complete examples:

- [Basic Usage](examples/basic_usage.py) - Simple resume tailoring
- [Advanced Workflow](examples/advanced_workflow.py) - Complex scenarios

## Best Practices

1. **Keep Resume Data Updated**: Maintain accurate employment history and achievements
2. **Use Specific Job Descriptions**: More detailed job descriptions yield better results
3. **Review Generated Content**: Always review AI-generated content for accuracy
4. **Test Before Using**: Validate tailored resumes before submission
5. **Version Control**: Keep versions of resumes for different roles

## Troubleshooting

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues and solutions.

## Performance Considerations

- Average processing time: 2-5 minutes per resume
- Token usage: ~3000-5000 tokens (depends on resume/job description length)
- Estimated cost: $0.10-0.30 per resume (using GPT-4)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes with clear messages
4. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For issues, questions, or suggestions:

- Open an issue on GitHub
- Check [Troubleshooting](docs/TROUBLESHOOTING.md)
- Review [Architecture](docs/ARCHITECTURE.md) for design decisions

## Roadmap

- [ ] Cover letter generation
- [ ] Multiple language support
- [ ] Persistent storage with database
- [ ] Web UI interface
- [ ] API service deployment
- [ ] Advanced analytics and feedback

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

**Note**: This tool generates resume content based on AI analysis. Always review and verify accuracy before submitting to employers.

