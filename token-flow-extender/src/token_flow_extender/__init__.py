"""token-flow-extender: lossless unlimited-input processing for any LLM.

Public API:
    TokenFlowExtender   direct SDK (Pattern 3)
    token_flow          decorator (Pattern 2)
    TokenFlowMiddleware ASGI middleware (Pattern 1)
    TokenFlowConfig     Pydantic configuration
"""

from .api.decorator import token_flow
from .api.middleware import TokenFlowMiddleware, TokenFlowMiddlewareSettings
from .api.sdk import OpenAICompatLLM, TokenFlowExtender
from .config import ResolvedLimits, TokenFlowConfig, WindowBudget
from .core.coverage_ledger import CoverageLedger, CoverageReport
from .core.flow_engine import FlowEngine, FlowResult
from .core.limit_resolver import LimitResolver
from .core.merge_engine import MergeEngine
from .core.token_counter import (
    EndpointTokenCounter,
    HeuristicTokenCounter,
    TokenCounter,
    get_counter,
)
from .core.window_planner import Window, WindowPlan, WindowPlanner
from .exceptions import (
    CheckpointError,
    CoverageError,
    LimitResolutionError,
    LLMCallError,
    TokenCountError,
    TokenFlowError,
    WindowPlanError,
)

__version__ = "1.0.0"

__all__ = [
    "TokenFlowExtender",
    "token_flow",
    "TokenFlowMiddleware",
    "TokenFlowMiddlewareSettings",
    "OpenAICompatLLM",
    "TokenFlowConfig",
    "ResolvedLimits",
    "WindowBudget",
    "CoverageLedger",
    "CoverageReport",
    "FlowEngine",
    "FlowResult",
    "LimitResolver",
    "MergeEngine",
    "TokenCounter",
    "EndpointTokenCounter",
    "HeuristicTokenCounter",
    "get_counter",
    "Window",
    "WindowPlan",
    "WindowPlanner",
    "TokenFlowError",
    "CoverageError",
    "CheckpointError",
    "LimitResolutionError",
    "LLMCallError",
    "TokenCountError",
    "WindowPlanError",
    "__version__",
]
