"""Exception hierarchy for token-flow-extender.

Every exception derives from TokenFlowError so integrators can catch a single
base class. The integration layer converts ALL of these into fail-open
pass-through behavior; they are never allowed to crash a host application.
"""

from __future__ import annotations


class TokenFlowError(Exception):
    """Base class for all token-flow-extender errors."""


class LimitResolutionError(TokenFlowError):
    """No source in the limit resolution chain produced a usable limit."""


class TokenCountError(TokenFlowError):
    """Token counting failed for the requested model."""


class WindowPlanError(TokenFlowError):
    """The window planner could not produce a valid lossless plan."""


class CoverageError(TokenFlowError):
    """Coverage verification failed: the union of window spans does not
    equal the full input span. This is the lossless-guarantee violation
    and always aborts the job rather than returning a partial result."""


class CheckpointError(TokenFlowError):
    """Checkpoint storage could not be read or written."""


class JobStateError(TokenFlowError):
    """An operation was attempted on a job in an incompatible state."""


class LLMCallError(TokenFlowError):
    """The downstream LLM call failed after retries."""
