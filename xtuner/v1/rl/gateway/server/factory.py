from __future__ import annotations

from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from transformers import AutoTokenizer
from xtuner.v1.rl.rollout.controller import RolloutController
from xtuner.v1.rl.rollout.reasoning_parser import ThinkTagReasoningParser
from xtuner.v1.rl.rollout.tool_call_parser import CompositeToolCallParser, JsonToolCallParser, RegexToolCallParser

from ..adapters import AnthropicChatAdapter, OpenAIChatAdapter
from ..adapters.responses import OpenAIResponsesAdapter
from ..backend.local_backend import LocalRolloutBackend
from ..backend.protocol import GatewayBackend
from ..core.exceptions import ContextLengthExceededError, GatewayError, GatewayStateError
from .routes import build_anthropic_router, build_openai_router, build_responses_router, build_runtime_router


# ---------------------------------------------------------------------------
# Dependency functions (used via FastAPI Depends() in route handlers)
# ---------------------------------------------------------------------------


def get_openai_adapter(request: Request) -> OpenAIChatAdapter:
    adapter = getattr(request.app.state, "gateway_openai_adapter", None)
    if adapter is None:
        raise GatewayStateError("Gateway OpenAI adapter is not configured.")
    return cast(OpenAIChatAdapter, adapter)


def get_anthropic_adapter(request: Request) -> AnthropicChatAdapter:
    adapter = getattr(request.app.state, "gateway_anthropic_adapter", None)
    if adapter is None:
        raise GatewayStateError("Gateway Anthropic adapter is not configured.")
    return cast(AnthropicChatAdapter, adapter)


def get_responses_adapter(request: Request) -> OpenAIResponsesAdapter:
    adapter = getattr(request.app.state, "gateway_responses_adapter", None)
    if adapter is None:
        raise GatewayStateError("Gateway Responses adapter is not configured.")
    return cast(OpenAIResponsesAdapter, adapter)


# ---------------------------------------------------------------------------
# App factories
# ---------------------------------------------------------------------------


def _create_base_gateway_app(
    backend: GatewayBackend,
    *,
    title: str = "XTuner Gateway",
    version: str = "0.1.0",
) -> FastAPI:
    """Create the base FastAPI app with runtime routes and global error
    handlers.

    This is an internal builder used by higher-level factory functions. The returned app exposes /livez, /readyz, and
    /capabilities but no protocol-specific endpoints.
    """
    app = FastAPI(title=title, version=version)
    app.state.gateway_backend = backend
    app.include_router(build_runtime_router())

    @app.exception_handler(ContextLengthExceededError)
    async def context_length_error_handler(request: Request, exc: ContextLengthExceededError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(exc), "type": "context_length_exceeded", "code": "context_too_long"}},
        )

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": type(exc).__name__, "code": "gateway_error"}},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "internal_error", "code": "internal_server_error"}},
        )

    return app


def build_local_gateway_app(
    controller: RolloutController,
    *,
    title: str = "XTuner Gateway",
    version: str = "0.1.0",
    capture_path: str | None = None,
) -> FastAPI:
    tokenizer = AutoTokenizer.from_pretrained(controller.config.tokenizer_path, trust_remote_code=True)

    strip_tokens: list[str] = []
    if tokenizer.eos_token:
        strip_tokens.append(tokenizer.eos_token)
    for tok in getattr(tokenizer, "additional_special_tokens", []):
        if any(marker in tok.lower() for marker in ("im_end", "eot", "end_of_turn", "turn_end")):
            strip_tokens.append(tok)

    controller.configure_output_parsers(
        tool_call_parser=CompositeToolCallParser(parsers=[JsonToolCallParser(), RegexToolCallParser()]),
        reasoning_parser=ThinkTagReasoningParser(strip_tokens=strip_tokens),
    )

    backend = LocalRolloutBackend(controller, tokenizer=tokenizer)
    app = _create_base_gateway_app(backend=backend, title=title, version=version)
    adapter_kwargs = {
        "generate_handler": backend.generate,
        "tokenizer": tokenizer,
        "default_model_name": controller.config.model_name,
        "context_length": controller.config.context_length,
        "capture_path": capture_path,
    }
    app.state.gateway_openai_adapter = OpenAIChatAdapter(**adapter_kwargs)
    app.state.gateway_anthropic_adapter = AnthropicChatAdapter(**adapter_kwargs)
    app.state.gateway_responses_adapter = OpenAIResponsesAdapter(**adapter_kwargs)
    app.include_router(build_openai_router())
    app.include_router(build_anthropic_router())
    app.include_router(build_responses_router())
    return app
