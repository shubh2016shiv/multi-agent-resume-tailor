# Weave Observability Integration

This module provides comprehensive observability for the Resume Tailor multi-agent system using Weave from Weights & Biases.

## Overview

The observability module enables complete visibility into your multi-agent workflow, tracking:

- **Agent execution flows** - Visualize how agents interact and pass data
- **Iterative improvement loops** - Track critique cycles and quality progression
- **Token usage and costs** - Monitor LLM API usage across all agents
- **Quality metrics** - Log scores, evaluations, and performance indicators
- **Timing and bottlenecks** - Identify slow operations

## Quick Start

### 1. Installation

```bash
pip install weave wandb
```

### 2. Get Your API Key

1. Sign up at [wandb.ai](https://wandb.ai)
2. Go to [wandb.ai/authorize](https://wandb.ai/authorize)
3. Copy your API key

### 3. Set Environment Variable

**Windows (PowerShell):**
```powershell
$env:WANDB_API_KEY="your_api_key_here"
```

**Linux/Mac:**
```bash
export WANDB_API_KEY="your_api_key_here"
```

**Or add to your `.env` file:**
```
WANDB_API_KEY=your_api_key_here
```

### 4. Run Your Workflow

Observability is automatically initialized when you run the orchestrator:

```python
from src.agent_orchestrator import ResumeTailorOrchestrator

orchestrator = ResumeTailorOrchestrator()
result = orchestrator.orchestrate(resume_text, job_description_text)
```

### 5. View Traces in Dashboard

After running, view your traces at:
```
https://wandb.ai/your-username/resume-tailor-agents
```

## Features

### Automatic Agent Tracing

All agent execution methods are automatically traced:

- Resume Extraction
- Job Analysis
- Gap Analysis
- Summary Writing
- Experience Optimization (with iterative loops)
- Skills Optimization
- ATS Assembly
- Quality Assurance

### Iterative Improvement Tracking

The Experience Optimizer agent's critique loops are fully traced:

**Iteration 1:**
- Agent generates initial bullets
- Tool evaluates quality → Score: 65/100
- Agent reads critique → "Add metrics, specify method..."
- Agent decides to regenerate

**Iteration 2:**
- Agent regenerates improved bullets
- Tool evaluates → Score: 78/100
- Agent reads critique → "Add scope indicators..."
- Agent regenerates again

**Iteration 3:**
- Agent regenerates with all improvements
- Tool evaluates → Score: 88/100
- Quality threshold met ✓
- Agent stops iterating

All of this is visible in the Weave dashboard with:
- Quality scores per iteration
- Token usage per iteration
- Issues identified
- Improvement delta

### Quality Metrics Logging

Quality Assurance agent logs comprehensive metrics:

```python
{
    "overall_quality_score": 87.5,
    "accuracy_score": 92.0,
    "relevance_score": 85.0,
    "ats_score": 86.0,
    "passed_threshold": true,
    "exaggerated_claims_count": 0,
    "missed_opportunities_count": 2
}
```

## Architecture

### Module Structure

```
src/observability/
├── __init__.py           # Public API exports
├── weave_setup.py        # Core implementation
└── README.md             # This file
```

### Key Components

**1. `init_observability(project_name, entity, enabled)`**
- One-time initialization
- Configures Weave tracking
- Called automatically by orchestrator

**2. `@trace_agent` Decorator**
- Traces high-level agent execution
- Captures inputs/outputs
- Creates parent traces in hierarchy

**3. `@trace_tool` Decorator**
- Traces tool and helper functions
- Creates child traces under agents
- Captures detailed execution flow

**4. `log_iteration_metrics(agent_name, iteration, metrics)`**
- Logs structured metrics
- Tracks quality progression
- Monitors token usage

## Usage Examples

### Manual Initialization

```python
from src.observability import init_observability

# Initialize with custom project name
init_observability(
    project_name="my-resume-project",
    entity="my-team",
    enabled=True
)
```

### Tracing Custom Functions

```python
from src.observability import trace_agent, trace_tool

@trace_agent
def my_custom_agent_workflow(input_data):
    """This will be traced as a parent span."""
    result = process_with_tool(input_data)
    return result

@trace_tool
def process_with_tool(data):
    """This will be traced as a child span."""
    return data.upper()
```

### Logging Custom Metrics

```python
from src.observability import log_iteration_metrics

log_iteration_metrics(
    agent_name="my_custom_agent",
    iteration=1,
    metrics={
        "quality_score": 85.5,
        "tokens_used": 1200,
        "processing_time_ms": 3400,
        "custom_metric": 42
    }
)
```

### Checking Observability Status

```python
from src.observability import is_weave_enabled, get_weave_project_url

if is_weave_enabled():
    print("✓ Observability is active")
    print(f"Dashboard: {get_weave_project_url()}")
else:
    print("✗ Observability is disabled")
```

## Dashboard Features

### Trace View

The Weave dashboard provides:

**Timeline View:**
- See execution flow chronologically
- Identify parallel vs sequential execution
- Spot timing bottlenecks

**Tree View:**
- Hierarchical agent → tool relationships
- Drill down into nested function calls
- See complete call stack

**Details Panel:**
- Input/output for each function
- Execution time
- Token usage
- Custom metrics

### Iteration Analysis

For iterative agents (Experience Optimizer):

**Score Progression:**
- Chart showing quality improvement
- Iteration-by-iteration breakdown
- Threshold visualization

**Issue Tracking:**
- Issues identified per iteration
- Resolution tracking
- Critique effectiveness

**Token Usage:**
- Per-iteration token consumption
- Total usage across iterations
- Cost estimation

### Quality Metrics

For QA agent:

**Score Dashboard:**
- Overall quality score
- Accuracy, relevance, ATS scores
- Pass/fail status

**Detailed Breakdown:**
- Exaggerated claims count
- Missed opportunities
- Keyword coverage

## Troubleshooting

### Observability Not Working

**Issue:** No traces appearing in dashboard

**Solutions:**
1. Check API key is set:
   ```python
   import os
   print(os.getenv("WANDB_API_KEY"))  # Should not be None
   ```

2. Check initialization succeeded:
   ```python
   from src.observability import is_weave_enabled
   print(is_weave_enabled())  # Should be True
   ```

3. Check logs for errors:
   ```
   grep "weave" logs/resume_tailor.log
   ```

### Missing Weave Package

**Issue:** `ImportError: No module named 'weave'`

**Solution:**
```bash
pip install weave wandb
```

### API Key Not Found

**Issue:** Warning about WANDB_API_KEY not found

**Solution:**
Set the environment variable before running:
```bash
export WANDB_API_KEY="your_key_here"
python your_script.py
```

## Configuration

### Disabling Observability

To disable Weave tracking without removing code:

```python
orchestrator = ResumeTailorOrchestrator(enable_observability=False)
```

Or set environment variable:
```bash
export WEAVE_ENABLED=false
```

### Custom Project Names

Change the project name in initialization:

```python
from src.observability import init_observability

init_observability(project_name="resume-prod-v2")
```

### Controlling Log Verbosity

Weave respects your logging configuration in `src/core/logger.py`.

To reduce Weave-specific logs:
```python
import logging
logging.getLogger("weave").setLevel(logging.WARNING)
```

## Best Practices

### 1. Descriptive Agent Names

Use clear names when logging metrics:
```python
# Good
log_iteration_metrics("experience_optimizer", iteration=1, metrics={...})

# Bad
log_iteration_metrics("agent1", iteration=1, metrics={...})
```

### 2. Consistent Metric Keys

Use the same metric keys across iterations for trend analysis:
```python
# Good - consistent keys
metrics = {"quality_score": 85, "token_count": 1200}

# Bad - inconsistent keys
metrics = {"score": 85, "tokens": 1200}  # Different keys in next iteration
```

### 3. Granular Tool Tracing

Decorate important helper functions to see detailed execution:
```python
@trace_tool
def evaluate_single_bullet(bullet):
    # This will show up as a child trace
    return score
```

### 4. Error Handling

Observability functions fail gracefully:
```python
# If Weave fails, your code still runs
@trace_agent
def my_function():
    # Will execute even if tracing fails
    return result
```

## Performance Impact

Weave observability has minimal performance impact:

- **Overhead:** < 5% execution time
- **Network:** Asynchronous logging (non-blocking)
- **Memory:** Negligible (< 50MB for typical workflow)

## Security & Privacy

### Data Handling

- Traces are sent to W&B cloud (or self-hosted)
- Data is encrypted in transit (HTTPS)
- API key required for authentication

### Sensitive Data

If your resumes contain sensitive information:

1. **Option 1:** Use environment-specific projects
   ```python
   init_observability(project_name="resume-tailor-dev")  # Dev data only
   ```

2. **Option 2:** Disable for production
   ```python
   is_prod = os.getenv("ENV") == "production"
   orchestrator = ResumeTailorOrchestrator(enable_observability=not is_prod)
   ```

3. **Option 3:** Self-host W&B
   - Deploy W&B server on-premises
   - Configure Weave to use your server

## Advanced Usage

### Custom Trace Attributes

Add custom metadata to traces:

```python
import weave

@trace_agent
def my_agent():
    weave.log({"custom_attribute": "value"})
    return result
```

### Conditional Tracing

Enable tracing only in certain conditions:

```python
from src.observability import init_observability

# Only enable in development
import os
if os.getenv("ENV") == "development":
    init_observability("resume-tailor-dev")
```

### Integration with Other Tools

Weave works alongside other observability tools:

- **structlog:** Local logging (already integrated)
- **Prometheus:** Metrics export (custom integration needed)
- **Sentry:** Error tracking (compatible)

## Support

### Documentation

- **Weave Docs:** [docs.wandb.ai/weave](https://docs.wandb.ai/weave)
- **W&B Docs:** [docs.wandb.ai](https://docs.wandb.ai)

### Community

- **W&B Community:** [community.wandb.ai](https://community.wandb.ai)
- **GitHub:** [github.com/wandb/weave](https://github.com/wandb/weave)

## Next Steps

1. **Run your first workflow** with observability enabled
2. **Explore the dashboard** at wandb.ai
3. **Review trace details** for each agent
4. **Analyze iteration metrics** for quality improvement
5. **Optimize based on insights** from token usage and timing

---

**Questions?** Check the logs in `logs/resume_tailor.log` for observability-related messages.

