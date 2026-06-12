"""Optional FastAPI router (extra: 'fastapi').

Mounts:
  POST /token-flow/jobs            submit a processing job
  GET  /token-flow/jobs/{job_id}   job status + result
  WS   /token-flow/jobs/{job_id}/progress   live window progress events

Usage:
    from token_flow_extender.api.routes import build_router
    app.include_router(build_router(extender, llm_factory=...))
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

from ..utils.logger import get_logger
from .sdk import TokenFlowExtender
from .websocket import ProgressBroker, broker as default_broker

logger = get_logger(__name__)


def build_router(
    extender: Optional[TokenFlowExtender] = None,
    llm_factory: Optional[Callable[[str], Callable[[str], Awaitable[str]]]] = None,
    base_url_factory: Optional[Callable[[str], Optional[str]]] = None,
    broker: Optional[ProgressBroker] = None,
):
    """Build the router. Imported lazily so FastAPI stays optional."""
    from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
    from pydantic import BaseModel

    active_broker = broker or default_broker
    tfe = extender or TokenFlowExtender(progress=active_broker.callback)

    router = APIRouter(prefix="/token-flow", tags=["token-flow-extender"])

    class JobRequest(BaseModel):
        input: str
        task: str = "Answer the user's request using the full input."
        model: str
        strategy: Optional[str] = None
        max_input_tokens: Optional[int] = None

    @router.post("/jobs")
    async def submit_job(request: JobRequest):
        llm = llm_factory(request.model) if llm_factory else None
        base_url = base_url_factory(request.model) if base_url_factory else None
        if llm is None and base_url is None:
            raise HTTPException(status_code=422,
                                detail="No llm_factory or base_url_factory configured")
        result = await tfe.process(
            request.input, task=request.task, model=request.model,
            llm=llm, base_url=base_url, strategy=request.strategy,
            max_input_tokens=request.max_input_tokens,
        )
        return result.as_dict()

    @router.get("/jobs/{job_id}")
    async def job_status(job_id: str):
        record = tfe.store.get_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}")
        return {
            "job_id": record.job_id,
            "status": record.status,
            "model": record.model,
            "total_windows": record.total_windows,
            "completed_windows": len(tfe.store.get_window_results(job_id)),
            "result": record.result,
            "error": record.error,
        }

    @router.websocket("/jobs/{job_id}/progress")
    async def job_progress(websocket: WebSocket, job_id: str):
        await websocket.accept()
        try:
            async for event in active_broker.subscribe(job_id):
                await websocket.send_json(event)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass

    return router
