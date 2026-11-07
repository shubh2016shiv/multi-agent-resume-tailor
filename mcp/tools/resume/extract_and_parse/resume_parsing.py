"""Tool: parse resume text into structured data using the CLIENT's LLM (sampling).

The server never calls an LLM itself; it asks the client's model to do the parsing.
"""

import json

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import (
    CreateMessageRequest,
    CreateMessageRequestParams,
    CreateMessageResult,
    InputRequiredResult,
    SamplingMessage,
    TextContent,
)
from pydantic import ValidationError

from tools.resume.extract_and_parse.schema import ParsedResume

MAX_OUTPUT_TOKENS_FOR_RESUME_PARSING = 4000
RESUME_PARSING_SYSTEM_PROMPT = (
    "You convert resume text into JSON. Copy every value word for word from the "
    "resume; never rewrite or invent anything. Use null for anything absent. "
    "Reply with only a JSON object that matches this JSON schema:\n"
)


def build_resume_parsing_sampling_request(resume_markdown: str) -> InputRequiredResult:
    """Build the request that asks the client's model to parse the resume."""
    parsed_resume_json_schema = json.dumps(ParsedResume.model_json_schema())
    sampling_request = CreateMessageRequest(
        method="sampling/createMessage",
        params=CreateMessageRequestParams(
            messages=[
                SamplingMessage(role="user", content=TextContent(type="text", text=resume_markdown))
            ],
            system_prompt=RESUME_PARSING_SYSTEM_PROMPT + parsed_resume_json_schema,
            max_tokens=MAX_OUTPUT_TOKENS_FOR_RESUME_PARSING,
        ),
    )
    return InputRequiredResult(
        result_type="input_required",
        input_requests={"resume_parsing_completion": sampling_request},
    )


def parse_resume_from_client_model_reply(client_model_reply: object) -> dict:
    """Turn the client model's reply into the parsed resume."""
    if not (
        isinstance(client_model_reply, CreateMessageResult)
        and isinstance(client_model_reply.content, TextContent)
    ):
        raise ToolError("The client returned no text completion")
    try:
        return ParsedResume.model_validate_json(client_model_reply.content.text).model_dump()
    except ValidationError as validation_error:
        raise ToolError(
            f"The model's JSON did not match the schema: {validation_error}"
        ) from validation_error


async def parse_resume(resume_markdown: str, tool_context: Context) -> dict | InputRequiredResult:
    """Parse resume Markdown into structured data, using the caller's LLM.

    Expects the Markdown returned by extract_resume_text. The first call asks
    the client's model to do the parsing; the client then calls this tool
    again with the answer. Returns the parsed resume as a dict. The client
    must support sampling.
    """
    client_input_responses = tool_context.input_responses
    if client_input_responses is None:
        return build_resume_parsing_sampling_request(resume_markdown)
    return parse_resume_from_client_model_reply(client_input_responses["resume_parsing_completion"])


def register_resume_parsing_tool(mcp: FastMCP) -> None:
    """Add this module's tool to the server."""
    mcp.tool(annotations={"readOnlyHint": True})(parse_resume)
