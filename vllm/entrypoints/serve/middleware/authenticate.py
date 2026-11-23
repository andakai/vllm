# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import secrets
from collections.abc import Awaitable, Sequence

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Match
from starlette.types import ASGIApp, Receive, Scope, Send

GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")

# Modules whose routes control the server: weights, generation state, memory,
# caches, RPC, LoRA adapters, profiling, checkpoints and scaling.
ADMIN_ROUTE_MODULES = (
    "vllm.entrypoints.rl.",
    "vllm.entrypoints.serve.dev.",
    "vllm.entrypoints.serve.lora.",
    "vllm.entrypoints.serve.profile.",
    "vllm.entrypoints.serve.checkpoint.",
    "vllm.entrypoints.serve.elastic_ep.",
)


def admin_routes(routes: Sequence[BaseRoute]) -> list[BaseRoute]:
    return [
        route
        for route in routes
        if getattr(route, "endpoint", None) is not None
        and route.endpoint.__module__.startswith(ADMIN_ROUTE_MODULES)  # type: ignore[attr-defined]
    ]


def _hash(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


class AuthenticationMiddleware:
    """Pure ASGI middleware that authenticates each request by checking
    if the Authorization Bearer token exists and equals anyof "{api_key}".

    With `admin_tokens`, requests matching `admin_routes` must present one of
    those instead, whatever their path prefix.

    Note:
        There are two cases in which authentication is skipped:

        1. The HTTP method is OPTIONS.
        2. The request path doesn't start with GUARDED_PREFIX (e.g. /health)
           and doesn't match an admin route.

    """

    def __init__(
        self,
        app: ASGIApp,
        tokens: list[str],
        admin_tokens: Sequence[str] = (),
        admin_routes: Sequence[BaseRoute] = (),
    ) -> None:
        self.app = app
        self.api_tokens = [_hash(t) for t in tokens]
        self.admin_tokens = [_hash(t) for t in admin_tokens]
        self.admin_routes = admin_routes if admin_tokens else ()

    @staticmethod
    def verify_token(headers: Headers, token_hashes: list[bytes]) -> bool:
        authorization_header_value = headers.get("Authorization")
        if not authorization_header_value:
            return False

        scheme, _, param = authorization_header_value.partition(" ")
        if scheme.lower() != "bearer":
            return False

        param_hash = _hash(param)

        token_match = False
        for token_hash in token_hashes:
            token_match |= secrets.compare_digest(param_hash, token_hash)

        return token_match

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if (
            scope["type"] not in ("http", "websocket")
            or scope.get("method") == "OPTIONS"
        ):
            # scope["type"] can be "lifespan" or "startup" for example,
            # in which case we don't need to do anything
            return self.app(scope, receive, send)
        headers = Headers(scope=scope)
        if any(route.matches(scope)[0] != Match.NONE for route in self.admin_routes):
            token_hashes = self.admin_tokens
        else:
            root_path = scope.get("root_path", "")
            url_path = scope["path"].removeprefix(root_path)
            if not (self.api_tokens and url_path.startswith(GUARDED_PREFIX)):
                return self.app(scope, receive, send)
            token_hashes = self.api_tokens
        if not self.verify_token(headers, token_hashes):
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)
