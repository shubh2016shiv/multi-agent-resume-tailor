import os

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from tools.resume.extract_and_parse.resume_extraction import register_resume_extraction_tool
from tools.resume.extract_and_parse.resume_parsing import register_resume_parsing_tool

mcp = FastMCP(
    name="resume-tailor",
    instructions="A tool to tailor resumes for specific job descriptions.",
    version="0.1.0",
)

register_resume_extraction_tool(mcp)
register_resume_parsing_tool(mcp)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Liveness probe for Docker and load balancers; not an MCP tool."""
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    # Default to localhost; the Docker image sets RESUME_MCP_HOST=0.0.0.0.
    host = os.environ.get("RESUME_MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("RESUME_MCP_PORT", "8010"))
    mcp.run(transport="http", host=host, port=port, stateless_http=True)
