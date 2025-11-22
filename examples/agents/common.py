import json
import re
from typing import Any, Callable, Dict, Optional

# Low quality resume for a Backend/Cloud developer
# This resume is intentionally vague and poorly formatted to demonstrate the agent's ability to extract and structure data.
RESUME_MD = """
# John Doe
Email: john.doe@email.com | Phone: 123-456-7890
Location: New York

## Summary
I am a developer with experience in coding. I like cloud and backend.
Looking for a job where I can code.

## Experience
### Software Engineer
Tech Corp
2020 - Present
- Wrote code for backend
- Used AWS
- Fixed bugs
- Helped team

### Junior Developer
StartUp Inc
2018 - 2020
- Learned Python
- Worked on database
- Made API

## Skills
- Python, Java, C++
- AWS, Azure
- SQL, NoSQL
- Git, Docker

## Education
BS Computer Science
University of Tech
2018
"""

# High quality Job Description for Backend/Cloud Engineer
JOB_DESC_MD = """
# Senior Backend Cloud Engineer

**Company:** CloudScale Solutions
**Location:** Remote / New York, NY

## About Us
CloudScale Solutions is a leading provider of scalable cloud infrastructure for enterprise clients. We are looking for a Senior Backend Cloud Engineer to join our core platform team.

## Role Overview
As a Senior Backend Cloud Engineer, you will design and build robust, scalable microservices using Python and Go. You will work heavily with AWS services (Lambda, DynamoDB, ECS) and ensure our infrastructure is reliable and efficient.

## Key Responsibilities
- Design, develop, and deploy serverless microservices on AWS.
- Optimize database performance (DynamoDB, PostgreSQL).
- Implement CI/CD pipelines using GitHub Actions and Terraform.
- Collaborate with frontend teams to define API specifications.
- Mentor junior engineers and conduct code reviews.

## Requirements
- **Must Have:**
  - 5+ years of experience in backend development (Python or Go).
  - Strong experience with AWS (Lambda, API Gateway, DynamoDB, IAM).
  - Proficiency in Infrastructure as Code (Terraform or CloudFormation).
  - Experience with containerization (Docker, Kubernetes/ECS).
- **Should Have:**
  - Experience with event-driven architecture.
  - Knowledge of NoSQL data modeling.
- **Nice to Have:**
  - AWS Certifications.
  - Experience with GraphQL.

## Benefits
- Competitive salary and equity.
- Remote-first culture.
- Comprehensive health insurance.
"""

def get_resume_md() -> str:
    """Returns the sample resume markdown string."""
    return RESUME_MD

def get_job_desc_md() -> str:
    """Returns the sample job description markdown string."""
    return JOB_DESC_MD

def parse_json_output(output: str) -> Dict[str, Any]:
    """
    Parses JSON from agent output, handling markdown code blocks.
    """
    json_str = output
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise

def validate_output(data: Dict[str, Any], validator_func: Callable[[Dict[str, Any]], Any]) -> bool:
    """
    Validates data using the provided validator function.
    Returns True if valid, False otherwise.
    """
    try:
        result = validator_func(data)
        return result is not None
    except Exception as e:
        print(f"Validation error: {e}")
        return False
