"""
Shared agent config loader.

All agent factories call this to pull their config block from agents.yaml
and validate that required fields are present before building a CrewAI Agent.
"""

# get_agents_config parses and caches src/config/agents.yaml.
from src.core.settings import get_agents_config

# Every agent persona block in agents.yaml must define these 4 keys.
REQUIRED_AGENT_CONFIG_FIELDS = ["role", "goal", "backstory", "llm"]


def load_agent_config(name: str) -> dict:
    """Load and validate an agent config block from agents.yaml.

    Expects: agents.yaml has a key matching `name` with role, goal, backstory, llm.
    Returns: the config dict.
    Raises: RuntimeError if any required field is missing.
    """
    ####################################################
    # STEP 1: PULL THE CONFIG BLOCK FOR THIS AGENT NAME
    ####################################################
    agents_config = get_agents_config()
    config = agents_config.get(name, {})

    ####################################################
    # STEP 2: VERIFY ALL REQUIRED FIELDS ARE PRESENT
    ####################################################
    missing_fields = [
        field_name
        for field_name in REQUIRED_AGENT_CONFIG_FIELDS
        if not config.get(field_name)
    ]
    if missing_fields:
        raise RuntimeError(
            f"FATAL: Missing required field(s) in '{name}' agent config: {missing_fields}\n"
            f"Add all required fields to src/config/agents.yaml."
        )

    return config
