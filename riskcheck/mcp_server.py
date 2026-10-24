"""The RiskCheck engine as MCP tools (MCP Python SDK 2.x).

Run as a stdio server:  python -m riskcheck.mcp_server
Requires ``make forecasts`` first (the tools read reports/forecasts.parquet and
reports/attribution.parquet).
"""

from __future__ import annotations

import asyncio
import functools
import json

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from riskcheck.copilot import engine

_P = "Portfolio id: one of 60_40, equal_sector, tech_heavy, defensive, risk_parity."
_M = "Model id: one of historical, ewma, garch, factor."
_D = "Dates are YYYY-MM-DD trading days; data covers 2006 to 2025-12-31."


def _expected_errors(fn):
    """Turn engine errors (bad name, no data, future date) into MCP tool errors.

    Over the protocol a ToolError reaches the client as an is_error result with its message.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except engine.EngineError as e:
            raise ToolError(str(e)) from e

    return wrapper


server = MCPServer(
    name="riskcheck",
    instructions="Portfolio risk engine: VaR/ES forecasts, factor risk attribution, VaR "
    "backtest statistics and realized returns for five ETF portfolios, 2006 to 2025. "
    "Tools return an error for unknown names, non-trading days and dates after the data.",
)


@server.tool(
    description=f"1-day volatility, 95%/99% VaR and 97.5% ES forecast for a "
    f"portfolio on a date, from one model, plus that day's realized return. "
    f"{_P} {_M} {_D}"
)
@_expected_errors
def get_var(portfolio: str, date: str, model: str) -> dict:
    return engine.get_var(portfolio, date, model)


@server.tool(
    description=f"Average share of a portfolio's risk from each source (market, "
    f"size, value, profitability, investment, momentum, rates, gold, "
    f"idiosyncratic) over a date range, from the 8-factor model. {_P} {_D}"
)
@_expected_errors
def get_attribution(portfolio: str, start_date: str, end_date: str) -> dict:
    return engine.get_attribution(portfolio, start_date, end_date)


@server.tool(
    description=f"VaR backtest over a date range: trading days, number of days "
    f"the loss exceeded the model's VaR, and the breach rate. level is 95 or 99. "
    f"{_P} {_M} {_D}"
)
@_expected_errors
def get_backtest_stats(
    portfolio: str, model: str, start_date: str, end_date: str, level: int
) -> dict:
    return engine.get_backtest_stats(portfolio, model, start_date, end_date, level)


@server.tool(
    description=f"Realized daily returns of a portfolio over a date range: total "
    f"return, worst and best day, annualized volatility (and each day's return "
    f"for ranges up to 31 trading days). {_P} {_D}"
)
@_expected_errors
def get_returns(portfolio: str, start_date: str, end_date: str) -> dict:
    return engine.get_returns(portfolio, start_date, end_date)


async def call_tool(name: str, arguments: dict) -> tuple[str, bool]:
    """Call a tool through the MCP server; returns (text, is_error) like an MCP client sees."""
    try:
        res = await server.call_tool(name, arguments)
        text = "\n".join(c.text for c in res.content if getattr(c, "text", None))
        return text, bool(res.is_error)
    except ToolError as e:
        return str(e), True
    except Exception as e:  # malformed arguments from the model, unknown tool, etc.
        return f"Error executing tool {name}: {type(e).__name__}: {e}", True


async def tool_schemas() -> list[dict]:
    """Tool definitions in the OpenAI/Groq function-calling format, taken from the server."""
    out = []
    for t in await server.list_tools():
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
        )
    return out


def call_tool_sync(name: str, arguments: dict) -> tuple[str, bool]:
    return asyncio.run(call_tool(name, arguments))


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


if __name__ == "__main__":
    server.run("stdio")
