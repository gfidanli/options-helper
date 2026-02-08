from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import pandas as pd

from options_helper.schemas.zero_dte_put_study import ExitMode


_SIMULATION_COLUMNS: tuple[str, ...] = (
    "session_date",
    "decision_ts",
    "entry_anchor_ts",
    "close_label_ts",
    "exit_ts",
    "risk_tier",
    "decision_mode",
    "time_of_day_bucket",
    "iv_regime",
    "strike_return",
    "strike_price",
    "quote_quality_status",
    "policy_status",
    "policy_reason",
    "status",
    "skip_reason",
    "exit_mode",
    "exit_reason",
    "sizing_mode",
    "quantity",
    "equity_before",
    "equity_after",
    "entry_premium_gross",
    "entry_premium_net",
    "exit_premium_gross",
    "exit_premium_net",
    "entry_fee_total",
    "exit_fee_total",
    "pnl_per_contract",
    "pnl_total",
    "close_intrinsic",
    "max_loss_proxy_per_contract",
    "max_loss_proxy_total",
)

_DEFAULT_EXIT_MODES: tuple[str, ...] = (
    ExitMode.HOLD_TO_CLOSE.value,
    ExitMode.ADAPTIVE_EXIT.value,
)


@dataclass(frozen=True)
class ZeroDTEPutSimulatorConfig:
    contract_multiplier: int = 100
    initial_equity: float = 100000.0
    sizing_mode: str = "fixed_contracts"
    fixed_contracts: int = 1
    risk_pct_of_equity: float = 0.01
    target_notional: float = 10000.0
    entry_slippage_bps: float = 0.0
    exit_slippage_bps: float = 0.0
    entry_fee_per_contract: float = 0.0
    exit_fee_per_contract: float = 0.0
    max_concurrent_positions: int = 1
    max_concurrent_positions_per_day: int = 1
    enforce_cash_constraint: bool = True
    adaptive_take_profit_return_threshold: float = 0.004
    adaptive_take_profit_capture: float = 0.6
    adaptive_stop_buffer_return: float = 0.0
    adaptive_stop_loss_multiple: float = 2.0
    adaptive_min_exit_premium: float = 0.01

    def __post_init__(self) -> None:
        if self.contract_multiplier < 1:
            raise ValueError("contract_multiplier must be >= 1")
        if self.initial_equity <= 0.0:
            raise ValueError("initial_equity must be > 0")
        if self.sizing_mode not in {"fixed_contracts", "risk_pct_of_equity", "target_notional"}:
            raise ValueError(
                "sizing_mode must be one of fixed_contracts, risk_pct_of_equity, target_notional"
            )
        if self.fixed_contracts < 1:
            raise ValueError("fixed_contracts must be >= 1")
        if self.risk_pct_of_equity <= 0.0 or self.risk_pct_of_equity > 1.0:
            raise ValueError("risk_pct_of_equity must be in (0, 1]")
        if self.target_notional <= 0.0:
            raise ValueError("target_notional must be > 0")
        if self.entry_slippage_bps < 0.0 or self.exit_slippage_bps < 0.0:
            raise ValueError("slippage bps must be >= 0")
        if self.entry_fee_per_contract < 0.0 or self.exit_fee_per_contract < 0.0:
            raise ValueError("fees must be >= 0")
        if self.max_concurrent_positions < 1:
            raise ValueError("max_concurrent_positions must be >= 1")
        if self.max_concurrent_positions_per_day < 1:
            raise ValueError("max_concurrent_positions_per_day must be >= 1")
        if self.adaptive_take_profit_capture <= 0.0 or self.adaptive_take_profit_capture >= 1.0:
            raise ValueError("adaptive_take_profit_capture must be in (0, 1)")
        if self.adaptive_stop_loss_multiple < 1.0:
            raise ValueError("adaptive_stop_loss_multiple must be >= 1")
        if self.adaptive_min_exit_premium < 0.0:
            raise ValueError("adaptive_min_exit_premium must be >= 0")


def simulate_zero_dte_put_outcomes(
    candidates: pd.DataFrame,
    *,
    config: ZeroDTEPutSimulatorConfig | None = None,
    exit_modes: Iterable[str] = _DEFAULT_EXIT_MODES,
) -> pd.DataFrame:
    cfg = config or ZeroDTEPutSimulatorConfig()
    normalized = _normalize_candidates(candidates)
    if normalized.empty:
        return pd.DataFrame(columns=list(_SIMULATION_COLUMNS))

    modes = _normalize_exit_modes(exit_modes)
    all_rows: list[dict[str, object]] = []
    for exit_mode in modes:
        all_rows.extend(_simulate_exit_track(normalized, exit_mode=exit_mode, config=cfg))

    if not all_rows:
        return pd.DataFrame(columns=list(_SIMULATION_COLUMNS))

    out = pd.DataFrame(all_rows, columns=list(_SIMULATION_COLUMNS))
    out = out.sort_values(
        by=["decision_ts", "risk_tier", "exit_mode"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def _simulate_exit_track(
    frame: pd.DataFrame,
    *,
    exit_mode: str,
    config: ZeroDTEPutSimulatorConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    equity = float(config.initial_equity)
    active_positions: list[dict[str, object]] = []

    ordered = frame.sort_values(
        by=["entry_anchor_ts", "decision_ts", "risk_tier"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    for _, candidate in ordered.iterrows():
        entry_ts = candidate["entry_anchor_ts"]
        session_date = candidate["session_date"]
        active_positions = [
            item for item in active_positions if pd.Timestamp(item["exit_ts"]) > pd.Timestamp(entry_ts)
        ]

        base_row = _build_base_row(candidate, exit_mode=exit_mode, sizing_mode=config.sizing_mode)
        base_row["equity_before"] = equity
        base_row["equity_after"] = equity
        if _has_policy_skip(candidate):
            base_row["skip_reason"] = _skip_reason_from_candidate(candidate)
            base_row["status"] = "skipped"
            rows.append(base_row)
            continue

        if len(active_positions) >= int(config.max_concurrent_positions):
            base_row["skip_reason"] = "concurrency_cap_total"
            base_row["status"] = "skipped"
            rows.append(base_row)
            continue

        same_day_open = sum(1 for item in active_positions if item["session_date"] == session_date)
        if same_day_open >= int(config.max_concurrent_positions_per_day):
            base_row["skip_reason"] = "concurrency_cap_same_day"
            base_row["status"] = "skipped"
            rows.append(base_row)
            continue

        price_inputs = _resolve_price_inputs(candidate, config=config, exit_mode=exit_mode)
        if price_inputs is None:
            base_row["skip_reason"] = "invalid_pricing_inputs"
            base_row["status"] = "skipped"
            rows.append(base_row)
            continue

        quantity = _resolve_quantity(
            candidate,
            equity=equity,
            max_loss_proxy_per_contract=price_inputs["max_loss_proxy_per_contract"],
            config=config,
        )
        if quantity < 1:
            base_row["skip_reason"] = "sizing_zero_quantity"
            base_row["status"] = "skipped"
            rows.append(base_row)
            continue

        pnl_per_contract = (
            (price_inputs["entry_premium_net"] - price_inputs["exit_premium_net"])
            * float(config.contract_multiplier)
            - float(config.entry_fee_per_contract)
            - float(config.exit_fee_per_contract)
        )
        pnl_total = pnl_per_contract * float(quantity)
        equity_after = equity + pnl_total

        filled_row = base_row.copy()
        filled_row["status"] = "filled"
        filled_row["quantity"] = int(quantity)
        filled_row["exit_ts"] = price_inputs["exit_ts"]
        filled_row["exit_reason"] = price_inputs["exit_reason"]
        filled_row["entry_premium_gross"] = price_inputs["entry_premium_gross"]
        filled_row["entry_premium_net"] = price_inputs["entry_premium_net"]
        filled_row["exit_premium_gross"] = price_inputs["exit_premium_gross"]
        filled_row["exit_premium_net"] = price_inputs["exit_premium_net"]
        filled_row["entry_fee_total"] = float(config.entry_fee_per_contract) * float(quantity)
        filled_row["exit_fee_total"] = float(config.exit_fee_per_contract) * float(quantity)
        filled_row["pnl_per_contract"] = pnl_per_contract
        filled_row["pnl_total"] = pnl_total
        filled_row["close_intrinsic"] = price_inputs["close_intrinsic"]
        filled_row["max_loss_proxy_per_contract"] = price_inputs["max_loss_proxy_per_contract"]
        filled_row["max_loss_proxy_total"] = price_inputs["max_loss_proxy_per_contract"] * float(quantity)
        filled_row["equity_after"] = equity_after
        rows.append(filled_row)

        equity = equity_after
        active_positions.append(
            {
                "session_date": session_date,
                "exit_ts": price_inputs["exit_ts"],
            }
        )

    return rows


def _resolve_price_inputs(
    candidate: pd.Series,
    *,
    config: ZeroDTEPutSimulatorConfig,
    exit_mode: str,
) -> dict[str, object] | None:
    entry_gross = _first_finite(candidate, "premium_estimate", "entry_premium")
    strike_price = _first_finite(candidate, "strike_price", "target_strike_price")
    if entry_gross is None or entry_gross <= 0.0:
        return None
    if strike_price is None or strike_price <= 0.0:
        return None

    close_intrinsic = _resolve_close_intrinsic(candidate, strike_price=strike_price)
    if close_intrinsic is None:
        return None

    entry_net = entry_gross * (1.0 - (float(config.entry_slippage_bps) / 10000.0))
    hold_exit_gross = close_intrinsic

    exit_gross = hold_exit_gross
    exit_ts = candidate.get("close_label_ts")
    exit_reason = "hold_to_close"

    if exit_mode == ExitMode.ADAPTIVE_EXIT.value:
        adaptive = _resolve_adaptive_exit(
            candidate,
            entry_credit=entry_gross,
            hold_exit_gross=hold_exit_gross,
            strike_price=strike_price,
            config=config,
        )
        exit_gross = adaptive["exit_premium_gross"]
        exit_reason = adaptive["exit_reason"]
        exit_ts = adaptive["exit_ts"]

    exit_net = float(exit_gross) * (1.0 + (float(config.exit_slippage_bps) / 10000.0))
    max_loss_proxy = max(
        (strike_price * float(config.contract_multiplier))
        - (entry_net * float(config.contract_multiplier)),
        0.0,
    )

    resolved_exit_ts = pd.to_datetime(exit_ts, errors="coerce", utc=True)
    if pd.isna(resolved_exit_ts):
        return None

    return {
        "entry_premium_gross": entry_gross,
        "entry_premium_net": entry_net,
        "exit_premium_gross": float(exit_gross),
        "exit_premium_net": exit_net,
        "exit_ts": resolved_exit_ts,
        "exit_reason": exit_reason,
        "close_intrinsic": close_intrinsic,
        "max_loss_proxy_per_contract": max_loss_proxy,
    }


def _resolve_adaptive_exit(
    candidate: pd.Series,
    *,
    entry_credit: float,
    hold_exit_gross: float,
    strike_price: float,
    config: ZeroDTEPutSimulatorConfig,
) -> dict[str, object]:
    provided_exit = _to_float(candidate.get("adaptive_exit_premium"))
    provided_exit_ts = pd.to_datetime(candidate.get("adaptive_exit_ts"), errors="coerce", utc=True)
    close_ts = pd.to_datetime(candidate.get("close_label_ts"), errors="coerce", utc=True)
    if provided_exit is not None and provided_exit >= 0.0:
        exit_ts = provided_exit_ts if pd.notna(provided_exit_ts) else close_ts
        return {
            "exit_premium_gross": provided_exit,
            "exit_ts": exit_ts,
            "exit_reason": "adaptive_exit_provided",
        }

    min_return = _to_float(candidate.get("intraday_min_return_from_entry"))
    max_return = _to_float(candidate.get("intraday_max_return_from_entry"))
    strike_return = _to_float(candidate.get("strike_return"))
    entry_underlying = _to_float(candidate.get("entry_anchor_price"))
    stop_exit_ts = pd.to_datetime(candidate.get("adaptive_stop_exit_ts"), errors="coerce", utc=True)
    take_profit_exit_ts = pd.to_datetime(
        candidate.get("adaptive_take_profit_exit_ts"), errors="coerce", utc=True
    )

    if (
        min_return is not None
        and strike_return is not None
        and min_return <= (strike_return + float(config.adaptive_stop_buffer_return))
    ):
        stop_intrinsic = hold_exit_gross
        if entry_underlying is not None and entry_underlying > 0.0:
            breached_underlying = entry_underlying * (1.0 + float(min_return))
            stop_intrinsic = max(strike_price - breached_underlying, 0.0)
        stop_exit = max(
            hold_exit_gross,
            stop_intrinsic,
            entry_credit * float(config.adaptive_stop_loss_multiple),
        )
        return {
            "exit_premium_gross": float(stop_exit),
            "exit_ts": stop_exit_ts if pd.notna(stop_exit_ts) else close_ts,
            "exit_reason": "adaptive_stop_loss",
        }

    if (
        max_return is not None
        and max_return >= float(config.adaptive_take_profit_return_threshold)
    ):
        take_profit_exit = max(
            float(config.adaptive_min_exit_premium),
            entry_credit * (1.0 - float(config.adaptive_take_profit_capture)),
        )
        return {
            "exit_premium_gross": float(take_profit_exit),
            "exit_ts": take_profit_exit_ts if pd.notna(take_profit_exit_ts) else close_ts,
            "exit_reason": "adaptive_take_profit",
        }

    return {
        "exit_premium_gross": hold_exit_gross,
        "exit_ts": close_ts,
        "exit_reason": "adaptive_hold_fallback",
    }


def _resolve_close_intrinsic(candidate: pd.Series, *, strike_price: float) -> float | None:
    close_price = _to_float(candidate.get("close_price"))
    if close_price is not None:
        return max(strike_price - close_price, 0.0)

    close_return = _to_float(candidate.get("close_return_from_entry"))
    entry_underlying = _to_float(candidate.get("entry_anchor_price"))
    if close_return is not None and entry_underlying is not None and entry_underlying > 0.0:
        close_underlying = entry_underlying * (1.0 + close_return)
        return max(strike_price - close_underlying, 0.0)

    strike_return = _to_float(candidate.get("strike_return"))
    if strike_return is not None and close_return is not None and entry_underlying is not None:
        return max((strike_return - close_return) * entry_underlying, 0.0)
    return None


def _resolve_quantity(
    candidate: pd.Series,
    *,
    equity: float,
    max_loss_proxy_per_contract: float,
    config: ZeroDTEPutSimulatorConfig,
) -> int:
    strike_price = _first_finite(candidate, "strike_price", "target_strike_price")
    if strike_price is None or strike_price <= 0.0:
        return 0

    quantity: int
    if config.sizing_mode == "fixed_contracts":
        quantity = int(config.fixed_contracts)
    elif config.sizing_mode == "risk_pct_of_equity":
        if max_loss_proxy_per_contract <= 0.0:
            return 0
        risk_budget = float(equity) * float(config.risk_pct_of_equity)
        quantity = int(math.floor(risk_budget / max_loss_proxy_per_contract))
    else:
        notional_per_contract = strike_price * float(config.contract_multiplier)
        if notional_per_contract <= 0.0:
            return 0
        quantity = int(math.floor(float(config.target_notional) / notional_per_contract))

    if quantity < 1:
        return 0

    if not config.enforce_cash_constraint:
        return quantity
    if max_loss_proxy_per_contract <= 0.0:
        return 0
    cash_cap_qty = int(math.floor(float(equity) / max_loss_proxy_per_contract))
    return max(min(quantity, cash_cap_qty), 0)


def _build_base_row(candidate: pd.Series, *, exit_mode: str, sizing_mode: str) -> dict[str, object]:
    return {
        "session_date": candidate.get("session_date"),
        "decision_ts": candidate.get("decision_ts"),
        "entry_anchor_ts": candidate.get("entry_anchor_ts"),
        "close_label_ts": candidate.get("close_label_ts"),
        "exit_ts": pd.NaT,
        "risk_tier": _to_float(candidate.get("risk_tier")),
        "decision_mode": candidate.get("decision_mode"),
        "time_of_day_bucket": candidate.get("time_of_day_bucket"),
        "iv_regime": candidate.get("iv_regime"),
        "strike_return": _to_float(candidate.get("strike_return")),
        "strike_price": _to_float(candidate.get("strike_price") or candidate.get("target_strike_price")),
        "quote_quality_status": _normalize_text(candidate.get("quote_quality_status"), default="unknown"),
        "policy_status": _normalize_text(candidate.get("policy_status"), default="ok"),
        "policy_reason": candidate.get("policy_reason"),
        "status": "pending",
        "skip_reason": None,
        "exit_mode": exit_mode,
        "exit_reason": None,
        "sizing_mode": sizing_mode,
        "quantity": 0,
        "equity_before": float("nan"),
        "equity_after": float("nan"),
        "entry_premium_gross": float("nan"),
        "entry_premium_net": float("nan"),
        "exit_premium_gross": float("nan"),
        "exit_premium_net": float("nan"),
        "entry_fee_total": float("nan"),
        "exit_fee_total": float("nan"),
        "pnl_per_contract": float("nan"),
        "pnl_total": float("nan"),
        "close_intrinsic": float("nan"),
        "max_loss_proxy_per_contract": float("nan"),
        "max_loss_proxy_total": float("nan"),
    }


def _normalize_candidates(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(_SIMULATION_COLUMNS))
    out = frame.copy()
    out.columns = [str(col) for col in out.columns]

    for ts_col in ("decision_ts", "entry_anchor_ts", "close_label_ts", "adaptive_exit_ts"):
        if ts_col in out.columns:
            out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
        else:
            out[ts_col] = pd.NaT

    if "session_date" not in out.columns:
        out["session_date"] = None
    if "risk_tier" not in out.columns:
        out["risk_tier"] = float("nan")
    if "policy_status" not in out.columns:
        out["policy_status"] = "ok"
    if "policy_reason" not in out.columns:
        out["policy_reason"] = None
    if "quote_quality_status" not in out.columns:
        out["quote_quality_status"] = "unknown"
    if "premium_estimate" not in out.columns and "entry_premium" in out.columns:
        out["premium_estimate"] = pd.to_numeric(out["entry_premium"], errors="coerce")
    out["premium_estimate"] = pd.to_numeric(out.get("premium_estimate"), errors="coerce")
    if "strike_price" in out.columns:
        out["strike_price"] = pd.to_numeric(out["strike_price"], errors="coerce")
    elif "target_strike_price" in out.columns:
        out["strike_price"] = pd.to_numeric(out["target_strike_price"], errors="coerce")
    else:
        out["strike_price"] = float("nan")
    out["strike_return"] = pd.to_numeric(out.get("strike_return"), errors="coerce")
    out["close_price"] = pd.to_numeric(out.get("close_price"), errors="coerce")
    out["close_return_from_entry"] = pd.to_numeric(out.get("close_return_from_entry"), errors="coerce")
    out["entry_anchor_price"] = pd.to_numeric(out.get("entry_anchor_price"), errors="coerce")
    out["intraday_min_return_from_entry"] = pd.to_numeric(
        out.get("intraday_min_return_from_entry"), errors="coerce"
    )
    out["intraday_max_return_from_entry"] = pd.to_numeric(
        out.get("intraday_max_return_from_entry"), errors="coerce"
    )
    out["adaptive_exit_premium"] = pd.to_numeric(out.get("adaptive_exit_premium"), errors="coerce")
    return out


def _normalize_exit_modes(exit_modes: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in exit_modes:
        mode = _normalize_text(raw, default="")
        if mode in {ExitMode.HOLD_TO_CLOSE.value, ExitMode.ADAPTIVE_EXIT.value} and mode not in seen:
            out.append(mode)
            seen.add(mode)
    if not out:
        return _DEFAULT_EXIT_MODES
    return tuple(out)


def _has_policy_skip(candidate: pd.Series) -> bool:
    status = _normalize_text(candidate.get("policy_status"), default="ok")
    if status in {"skip", "skipped"}:
        return True
    return False


def _skip_reason_from_candidate(candidate: pd.Series) -> str:
    reason = _normalize_text(candidate.get("policy_reason"), default="")
    if reason:
        return reason
    explicit = _normalize_text(candidate.get("skip_reason"), default="")
    if explicit:
        return explicit
    return "policy_skip"


def _normalize_text(value: object, *, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    return text if text else default


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _first_finite(candidate: pd.Series, *keys: str) -> float | None:
    for key in keys:
        value = _to_float(candidate.get(key))
        if value is not None:
            return value
    return None


__all__ = [
    "ZeroDTEPutSimulatorConfig",
    "simulate_zero_dte_put_outcomes",
]
