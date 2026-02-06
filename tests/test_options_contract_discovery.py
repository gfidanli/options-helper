from __future__ import annotations

from datetime import date
import time

from options_helper.data.ingestion.options_bars import discover_option_contracts


def _raw_contract(underlying: str) -> dict[str, object]:
    return {
        "contractSymbol": f"{underlying}260117C00100000",
        "underlying": underlying,
        "expiration_date": "2026-01-17",
        "option_type": "call",
        "strike_price": 100.0,
        "multiplier": 100,
    }


class _ClientWithContractsRps:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def list_option_contracts(
        self,
        underlying: str,
        *,
        exp_gte: date | None = None,  # noqa: ARG002
        exp_lte: date | None = None,  # noqa: ARG002
        limit: int | None = None,  # noqa: ARG002
        page_limit: int | None = None,  # noqa: ARG002
        max_requests_per_second: float | None = None,
    ) -> list[dict[str, object]]:
        self.calls.append({"underlying": underlying, "max_rps": max_requests_per_second, "limit": limit})
        return [_raw_contract(underlying)]


class _ClientWithoutContractsRps:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def list_option_contracts(
        self,
        underlying: str,
        *,
        exp_gte: date | None = None,  # noqa: ARG002
        exp_lte: date | None = None,  # noqa: ARG002
        limit: int | None = None,  # noqa: ARG002
        page_limit: int | None = None,  # noqa: ARG002
    ) -> list[dict[str, object]]:
        self.calls.append(underlying)
        return [_raw_contract(underlying)]


class _ClientTimingProbe:
    def __init__(self) -> None:
        self.call_starts: list[float] = []

    def list_option_contracts(
        self,
        underlying: str,  # noqa: ARG002
        *,
        exp_gte: date | None = None,  # noqa: ARG002
        exp_lte: date | None = None,  # noqa: ARG002
        limit: int | None = None,  # noqa: ARG002
        page_limit: int | None = None,  # noqa: ARG002
        max_requests_per_second: float | None = None,  # noqa: ARG002
    ) -> list[dict[str, object]]:
        self.call_starts.append(time.monotonic())
        return []


class _ClientTimingProbeNoRps:
    def __init__(self) -> None:
        self.call_starts: list[float] = []

    def list_option_contracts(
        self,
        underlying: str,  # noqa: ARG002
        *,
        exp_gte: date | None = None,  # noqa: ARG002
        exp_lte: date | None = None,  # noqa: ARG002
        limit: int | None = None,  # noqa: ARG002
        page_limit: int | None = None,  # noqa: ARG002
    ) -> list[dict[str, object]]:
        self.call_starts.append(time.monotonic())
        return []


def test_discover_option_contracts_passes_max_rps_when_supported() -> None:
    client = _ClientWithContractsRps()

    result = discover_option_contracts(
        client,  # type: ignore[arg-type]
        underlyings=["AAA"],
        exp_start=date(2026, 1, 1),
        exp_end=date(2026, 12, 31),
        page_limit=10,
        max_requests_per_second=2.5,
    )

    assert len(client.calls) == 1
    assert client.calls[0]["max_rps"] == 2.5
    assert client.calls[0]["limit"] is None
    assert not result.contracts.empty


def test_discover_option_contracts_fallback_for_clients_without_max_rps_kw() -> None:
    client = _ClientWithoutContractsRps()

    result = discover_option_contracts(
        client,  # type: ignore[arg-type]
        underlyings=["BBB"],
        exp_start=date(2026, 1, 1),
        exp_end=date(2026, 12, 31),
        max_requests_per_second=2.5,
    )

    assert client.calls == ["BBB"]
    assert len(result.summaries) == 1
    assert result.summaries[0].status == "ok"
    assert result.summaries[0].contracts == 1


def test_discover_option_contracts_throttle_limits_underlying_scan_rate_for_clients_without_max_rps_kw() -> None:
    client = _ClientTimingProbeNoRps()

    discover_option_contracts(
        client,  # type: ignore[arg-type]
        underlyings=["AAA", "BBB", "CCC"],
        exp_start=date(2026, 1, 1),
        exp_end=date(2026, 12, 31),
        max_requests_per_second=5.0,
    )

    assert len(client.call_starts) == 3
    elapsed = max(client.call_starts) - min(client.call_starts)
    assert elapsed >= 0.35


def test_discover_option_contracts_does_not_double_throttle_when_client_supports_max_rps_kw() -> None:
    client = _ClientTimingProbe()

    discover_option_contracts(
        client,  # type: ignore[arg-type]
        underlyings=["AAA", "BBB", "CCC"],
        exp_start=date(2026, 1, 1),
        exp_end=date(2026, 12, 31),
        max_requests_per_second=5.0,
    )

    assert len(client.call_starts) == 3
    elapsed = max(client.call_starts) - min(client.call_starts)
    assert elapsed < 0.25


def test_discover_option_contracts_passes_limit_page_size() -> None:
    client = _ClientWithContractsRps()

    result = discover_option_contracts(
        client,  # type: ignore[arg-type]
        underlyings=["AAA"],
        exp_start=date(2026, 1, 1),
        exp_end=date(2026, 12, 31),
        limit=10000,
        max_requests_per_second=2.5,
    )

    assert not result.contracts.empty
    assert client.calls[0]["limit"] == 10000
