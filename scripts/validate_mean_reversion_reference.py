#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_PATH = Path(
    "/Users/sergio/Library/Mobile Documents/iCloud~md~obsidian/Documents/Personal/Clippings/"
    "Found a simple mean reversion setup with 70% win rate but only invested 20% of the time.md"
)
DEFAULT_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "technicals" / "reddit_mean_reversion_reference.json"

REFERENCE_SCHEMA_VERSION = 1
PERCENT_METRIC_KEYS = (
    "total_return_pct",
    "cagr_pct",
    "win_rate_pct",
    "max_drawdown_pct",
    "time_invested_pct",
)
DEFAULT_TOLERANCES = {
    "percent_metrics_abs_pct_points": 1.0,
    "profit_factor_abs": 0.10,
    "total_trades_abs": 3,
    "final_capital_rel_pct": 1.50,
}


def _search(pattern: str, text: str, *, field: str) -> re.Match[str]:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not parse {field} from reference markdown.")
    return match


def _parse_float_token(raw: str) -> float:
    return float(raw.replace(",", "").replace("$", "").replace("%", "").strip())


def _parse_int_token(raw: str) -> int:
    return int(float(raw.replace(",", "").replace("$", "").replace("%", "").strip()))


def _find_symbol_section(markdown: str, *, symbol: str) -> str:
    if symbol.upper() == "SPY":
        anchor = _search(
            r"^\s*-\s*Ticker\s*-\s*\*{0,2}\s*SPY\s*\*{0,2}\s*$",
            markdown,
            field="SPY section anchor",
        )
    else:
        anchor = _search(
            rf"^\s*###\s*Testing\s+with\s+ticker\s+{re.escape(symbol)}\b.*$",
            markdown,
            field=f"{symbol} section anchor",
        )
    start = anchor.start()
    scan_start = anchor.end()
    trailing = markdown[scan_start:]
    next_heading = re.search(r"^\s*#{2,6}\s+", trailing, flags=re.MULTILINE)
    end = len(markdown) if next_heading is None else scan_start + next_heading.start()
    return markdown[start:end]


def _parse_symbol_metrics(section: str, *, symbol: str) -> dict[str, float | int]:
    return {
        "total_return_pct": _parse_float_token(
            _search(r"Total\s+Return\s*:\s*([-$0-9,.\s]+%)", section, field=f"{symbol} total return").group(1)
        ),
        "cagr_pct": _parse_float_token(
            _search(r"CAGR\s*:\s*([-$0-9,.\s]+%)", section, field=f"{symbol} cagr").group(1)
        ),
        "profit_factor": _parse_float_token(
            _search(r"Profit\s+Factor\s*:\s*([0-9,.\s]+)", section, field=f"{symbol} profit factor").group(1)
        ),
        "win_rate_pct": _parse_float_token(
            _search(r"Win\s+Rate\s*:\s*([-$0-9,.\s]+%)", section, field=f"{symbol} win rate").group(1)
        ),
        "max_drawdown_pct": _parse_float_token(
            _search(r"Max\s+Drawdown\s*:\s*([-$0-9,.\s]+%)", section, field=f"{symbol} max drawdown").group(1)
        ),
        "time_invested_pct": _parse_float_token(
            _search(r"Time\s+Invested\s*:\s*([-$0-9,.\s]+%)", section, field=f"{symbol} time invested").group(1)
        ),
        "total_trades": _parse_int_token(
            _search(r"Total\s+Trades\s*:\s*([0-9,\s]+)", section, field=f"{symbol} total trades").group(1)
        ),
        "final_capital": _parse_float_token(
            _search(r"Final\s+Capital\s*:\s*([$0-9,.\s]+)", section, field=f"{symbol} final capital").group(1)
        ),
    }


def parse_reference_markdown(markdown: str, *, source_path: str | None = None) -> dict[str, Any]:
    entry_pattern = (
        r"close\s*<\s*\(\s*(\d+)\s*days\s*high\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*(?:\\?\*)\s*"
        r"\(\s*(\d+)\s*days\s*average\s*high\s*-\s*(\d+)\s*days\s*average\s*low"
    )
    entry_match = _search(entry_pattern, markdown, field="entry formula")
    high_window_days = int(entry_match.group(1))
    range_mult = float(entry_match.group(2))
    avg_high_days = int(entry_match.group(3))
    avg_low_days = int(entry_match.group(4))
    if avg_high_days != avg_low_days:
        raise ValueError("Reference markdown has mismatched average high/low windows.")

    ibs_formula_line = _search(r"IBS.*=\s*`?([^`\n]+)`?", markdown, field="IBS formula").group(1).lower()
    if "close" not in ibs_formula_line or "low" not in ibs_formula_line or "high" not in ibs_formula_line:
        raise ValueError("Reference markdown IBS formula is missing expected OHLC terms.")

    spy_section = _find_symbol_section(markdown, symbol="SPY")
    qqq_section = _find_symbol_section(markdown, symbol="QQQ")
    payload = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "source_path": source_path,
        "rules": {
            "entry": {
                "lookback_high_days": high_window_days,
                "range_window_days": avg_high_days,
                "range_multiplier": range_mult,
                "ibs_threshold": _parse_float_token(
                    _search(r"ibs\s*<\s*([0-9]+(?:\.[0-9]+)?)", markdown, field="IBS threshold").group(1)
                ),
                "ibs_formula": "(close - low) / (high - low)",
            },
            "exit": {
                "close_gt_yesterday_high": bool(
                    re.search(
                        r"close\s*>\s*yesterday(?:'s)?\s*high",
                        markdown,
                        flags=re.IGNORECASE | re.MULTILINE,
                    )
                ),
            },
            "backtest_defaults": {
                "timeframe": _search(
                    r"^\s*-\s*Timeframe\s*-\s*([A-Za-z0-9 ]+)$",
                    markdown,
                    field="timeframe",
                )
                .group(1)
                .strip(),
                "ticker": "SPY",
                "slippage": _parse_float_token(
                    _search(r"^\s*-\s*Slippage\s*-\s*([0-9.,]+)$", markdown, field="slippage").group(1)
                ),
                "commission": _parse_float_token(
                    _search(r"^\s*-\s*commission\s*-\s*([0-9.,]+)$", markdown, field="commission").group(1)
                ),
                "initial_capital": _parse_float_token(
                    _search(r"^\s*-\s*Capital\s*-\s*([$0-9,.\s]+)$", markdown, field="capital").group(1)
                ),
            },
        },
        "metrics": {
            "SPY": _parse_symbol_metrics(spy_section, symbol="SPY"),
            "QQQ": _parse_symbol_metrics(qqq_section, symbol="QQQ"),
        },
        "tolerances": dict(DEFAULT_TOLERANCES),
    }
    return payload


def load_reference_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Fixture payload must be a JSON object.")
    return payload


def compare_reference_payload_to_fixture(
    payload: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    if int(payload.get("schema_version", 0)) != int(fixture.get("schema_version", 0)):
        mismatches.append(
            f"schema_version mismatch: parsed={payload.get('schema_version')} fixture={fixture.get('schema_version')}"
        )
    for symbol in ("SPY", "QQQ"):
        parsed_metrics = payload.get("metrics", {}).get(symbol)
        fixture_metrics = fixture.get("metrics", {}).get(symbol)
        if not isinstance(parsed_metrics, Mapping) or not isinstance(fixture_metrics, Mapping):
            mismatches.append(f"missing metrics section for symbol {symbol}")
            continue
        for key, expected in fixture_metrics.items():
            actual = parsed_metrics.get(key)
            if isinstance(expected, (int, float)):
                if abs(float(actual) - float(expected)) > 1e-9:
                    mismatches.append(f"{symbol}.{key} mismatch: parsed={actual} fixture={expected}")
                continue
            if actual != expected:
                mismatches.append(f"{symbol}.{key} mismatch: parsed={actual} fixture={expected}")
    if payload.get("rules") != fixture.get("rules"):
        mismatches.append("rules mismatch between parsed markdown and fixture")
    if payload.get("tolerances") != fixture.get("tolerances"):
        mismatches.append("tolerances mismatch between parsed markdown and fixture")
    return mismatches


def _to_float(value: object) -> float:
    if isinstance(value, str):
        return _parse_float_token(value)
    return float(value)


def _normalize_artifact_symbol_metrics(mapping: Mapping[str, Any]) -> dict[str, float | int]:
    return {
        "total_return_pct": _to_float(mapping.get("total_return_pct", 0.0)) * 100.0,
        "cagr_pct": _to_float(mapping.get("annualized_return_pct", 0.0)) * 100.0,
        "profit_factor": _to_float(mapping.get("profit_factor", 0.0)),
        "win_rate_pct": _to_float(mapping.get("win_rate", 0.0)) * 100.0,
        "max_drawdown_pct": abs(_to_float(mapping.get("max_drawdown_pct", 0.0)) * 100.0),
        "time_invested_pct": _to_float(mapping.get("invested_pct", 0.0)) * 100.0,
        "total_trades": int(round(_to_float(mapping.get("trade_count", 0)))),
        "final_capital": _to_float(mapping.get("ending_equity", 0.0)),
    }


def _normalize_stats_symbol_metrics(mapping: Mapping[str, Any]) -> dict[str, float | int]:
    return {
        "total_return_pct": _to_float(mapping.get("Return [%]", 0.0)),
        "cagr_pct": _to_float(mapping.get("CAGR [%]", 0.0)),
        "profit_factor": _to_float(mapping.get("Profit Factor", 0.0)),
        "win_rate_pct": _to_float(mapping.get("Win Rate [%]", 0.0)),
        "max_drawdown_pct": abs(_to_float(mapping.get("Max. Drawdown [%]", 0.0))),
        "time_invested_pct": _to_float(mapping.get("Exposure Time [%]", 0.0)),
        "total_trades": int(round(_to_float(mapping.get("# Trades", 0)))),
        "final_capital": _to_float(mapping.get("Equity Final [$]", 0.0)),
    }


def normalize_run_metrics_payload(run_payload: Mapping[str, Any]) -> dict[str, dict[str, float | int]]:
    per_symbol = run_payload.get("per_symbol_metrics")
    if isinstance(per_symbol, list):
        normalized: dict[str, dict[str, float | int]] = {}
        for item in per_symbol:
            if not isinstance(item, Mapping):
                continue
            symbol = str(item.get("symbol", "")).upper().strip()
            metrics = item.get("metrics")
            if not symbol or not isinstance(metrics, Mapping):
                continue
            normalized[symbol] = _normalize_artifact_symbol_metrics(metrics)
        return normalized

    normalized = {}
    for symbol, metrics in run_payload.items():
        if not isinstance(metrics, Mapping):
            continue
        normalized[str(symbol).upper()] = _normalize_stats_symbol_metrics(metrics)
    return normalized


def validate_run_metrics_against_fixture(
    run_metrics: Mapping[str, Mapping[str, float | int]],
    fixture: Mapping[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    tolerances = fixture.get("tolerances", {})
    percent_tol = float(tolerances.get("percent_metrics_abs_pct_points", DEFAULT_TOLERANCES["percent_metrics_abs_pct_points"]))
    pf_tol = float(tolerances.get("profit_factor_abs", DEFAULT_TOLERANCES["profit_factor_abs"]))
    trades_tol = int(tolerances.get("total_trades_abs", DEFAULT_TOLERANCES["total_trades_abs"]))
    final_cap_rel_tol = float(tolerances.get("final_capital_rel_pct", DEFAULT_TOLERANCES["final_capital_rel_pct"]))

    targets = fixture.get("metrics", {})
    for symbol, target_metrics in targets.items():
        if symbol not in run_metrics:
            mismatches.append(f"missing run metrics for {symbol}")
            continue
        actual_metrics = run_metrics[symbol]
        for key, target_value in target_metrics.items():
            actual_value = actual_metrics.get(key)
            if actual_value is None:
                mismatches.append(f"{symbol}.{key} missing from run metrics")
                continue
            if key in PERCENT_METRIC_KEYS:
                delta = abs(float(actual_value) - float(target_value))
                if delta > percent_tol:
                    mismatches.append(f"{symbol}.{key} delta={delta:.4f} exceeds {percent_tol:.4f}")
                continue
            if key == "profit_factor":
                delta = abs(float(actual_value) - float(target_value))
                if delta > pf_tol:
                    mismatches.append(f"{symbol}.{key} delta={delta:.4f} exceeds {pf_tol:.4f}")
                continue
            if key == "total_trades":
                delta = abs(int(actual_value) - int(target_value))
                if delta > trades_tol:
                    mismatches.append(f"{symbol}.{key} delta={delta} exceeds {trades_tol}")
                continue
            if key == "final_capital":
                base = abs(float(target_value))
                rel_delta = 0.0 if base == 0.0 else abs(float(actual_value) - float(target_value)) / base * 100.0
                if rel_delta > final_cap_rel_tol:
                    mismatches.append(f"{symbol}.{key} rel_delta={rel_delta:.4f}% exceeds {final_cap_rel_tol:.4f}%")
    return mismatches


def _write_fixture(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate MeanReversionIBS markdown reference parity.")
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Path to reference markdown file.",
    )
    parser.add_argument(
        "--fixture-path",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="Fixture JSON path for deterministic parity.",
    )
    parser.add_argument(
        "--run-summary-path",
        type=Path,
        default=None,
        help="Optional summary JSON path from a strategy run to compare against reference tolerances.",
    )
    parser.add_argument(
        "--write-fixture",
        action="store_true",
        help="Parse the reference markdown and refresh the normalized fixture JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    fixture_payload = load_reference_fixture(args.fixture_path) if args.fixture_path.exists() else None
    if args.write_fixture:
        if not args.reference_path.exists():
            print(f"ERROR: reference markdown path does not exist: {args.reference_path}")
            return 2
        parsed = parse_reference_markdown(
            args.reference_path.read_text(encoding="utf-8"),
            source_path=str(args.reference_path),
        )
        _write_fixture(args.fixture_path, parsed)
        fixture_payload = parsed
        print(f"Wrote normalized fixture: {args.fixture_path}")

    if fixture_payload is None:
        print(f"ERROR: fixture JSON does not exist: {args.fixture_path}")
        return 2

    ok = True
    if args.reference_path.exists():
        parsed = parse_reference_markdown(
            args.reference_path.read_text(encoding="utf-8"),
            source_path=str(args.reference_path),
        )
        mismatches = compare_reference_payload_to_fixture(parsed, fixture_payload)
        if mismatches:
            ok = False
            print("FAIL: reference markdown does not match normalized fixture.")
            for item in mismatches:
                print(f"  - {item}")
        else:
            print("PASS: reference markdown matches normalized fixture.")
    else:
        print(f"SKIP: reference markdown path not found: {args.reference_path}")

    if args.run_summary_path is not None:
        run_payload = json.loads(args.run_summary_path.read_text(encoding="utf-8"))
        run_metrics = normalize_run_metrics_payload(run_payload)
        mismatches = validate_run_metrics_against_fixture(run_metrics, fixture_payload)
        if mismatches:
            ok = False
            print("FAIL: run summary metrics are outside reference tolerances.")
            for item in mismatches:
                print(f"  - {item}")
        else:
            print("PASS: run summary metrics are within reference tolerances.")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
