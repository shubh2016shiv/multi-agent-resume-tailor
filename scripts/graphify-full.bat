@echo off
REM graphify-full.bat — Full graphify build with LLM (community naming + report + visualization)
REM Usage: scripts\graphify-full.bat
REM        set OPENAI_API_KEY=sk-... && scripts\graphify-full.bat
REM        scripts\graphify-full.bat --model gpt-4o
setlocal enabledelayedexpansion

set MODEL=gpt-4o

REM Parse optional --model flag
:parse_args
if "%~1"=="" goto :check_key
if "%~1"=="--model" (
    set MODEL=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    echo Usage: graphify-full.bat [--model MODEL_NAME]
    echo.
    echo Set OPENAI_API_KEY in environment or .env file before running.
    exit /b 0
)
echo Unknown option: %~1
exit /b 1

:check_key
REM Resolve API key: explicit env > .env file > prompt
if not defined OPENAI_API_KEY (
    if exist .env (
        for /f "tokens=1,2 delims==" %%a in ('findstr /b "OPENAI_API_KEY=" .env 2^>nul') do (
            set "OPENAI_API_KEY=%%b"
        )
    )
)

if not defined OPENAI_API_KEY (
    set /p "OPENAI_API_KEY=Enter your OpenAI API key: "
    if "!OPENAI_API_KEY!"=="" (
        echo ERROR: No API key provided. Set OPENAI_API_KEY env var or create a .env file.
        exit /b 1
    )
)

echo Running full graphify build (AST + LLM report + visualization)...
echo Backend: openai ^| Model: !MODEL!
echo.

uv run graphify extract . --backend openai --model !MODEL!

echo.
echo Done. Output in graphify-out\:
echo   graph.json         -- queryable knowledge graph
echo   GRAPH_REPORT.md    -- human-readable architecture summary
echo   graph.html         -- interactive visualization

endlocal
