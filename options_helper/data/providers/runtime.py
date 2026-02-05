from __future__ import annotations

from contextvars import ContextVar, Token

_DEFAULT_PROVIDER_NAME: ContextVar[str] = ContextVar("options_helper_default_provider_name", default="alpaca")


def get_default_provider_name() -> str:
    return _DEFAULT_PROVIDER_NAME.get()


def set_default_provider_name(name: str | None) -> Token[str]:
    cleaned = (name or "alpaca").strip().lower()
    if not cleaned:
        cleaned = "alpaca"
    return _DEFAULT_PROVIDER_NAME.set(cleaned)


def reset_default_provider_name(token: Token[str]) -> None:
    _DEFAULT_PROVIDER_NAME.reset(token)
