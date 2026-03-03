from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "validate_mean_reversion_reference.py"
    spec = importlib.util.spec_from_file_location("validate_mean_reversion_reference", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "technicals" / "reddit_mean_reversion_reference.json"


def _reference_markdown_sample() -> str:
    return """## Entry condition
close < (10 days high - 2.5* (25 days average high - 25 days average low) and
IBS < 0.3

## Exit
close > yesterday's high

## Backtest
- Timeframe - Daily
- Ticker - **SPY**
- Slippage - 0.01
- commission - 0.01
- Duration - 2006 march till 2026 march
- Capital - 100,000

**Core Returns**
- Total Return: 334.84%
- CAGR: 7.75%
- Profit Factor: 2.02
- Win Rate: 75.00% (180 Wins / 60 Losses)

**Risk Metrics**
- Max Drawdown: 15.26%

**Position & Efficiency**
- Time Invested: 21.02%

**Execution & Friction**
- Total Trades: 240
- Final Capital: $434,835.64

IBS (Internal Bar Strength) = `(close - low) / (high - low)`

### Testing with ticker QQQ (2011 - 2026)
**Core Returns**
- Total Return: 265.74%
- CAGR: 9.18%
- Profit Factor: 2.15
- Win Rate: 70.74%
**Risk Metrics**
- Max Drawdown: 11.92%
**Position & Efficiency**
- Time Invested: 16.41%
**Execution & Friction**
- Total Trades: 188
- Final Capital: $365,740.47
"""


def _load_fixture() -> dict:
    return json.loads(_fixture_path().read_text(encoding="utf-8"))


def test_reference_fixture_has_locked_targets_and_tolerances() -> None:
    fixture = _load_fixture()
    assert fixture["schema_version"] == 1
    assert fixture["metrics"]["SPY"]["total_return_pct"] == 334.84
    assert fixture["metrics"]["QQQ"]["total_return_pct"] == 265.74
    assert fixture["tolerances"] == {
        "percent_metrics_abs_pct_points": 1.0,
        "profit_factor_abs": 0.1,
        "total_trades_abs": 3,
        "final_capital_rel_pct": 1.5,
    }


def test_parse_reference_markdown_handles_format_noise() -> None:
    module = _load_module()
    payload = module.parse_reference_markdown(_reference_markdown_sample(), source_path="sample.md")

    assert payload["rules"]["entry"] == {
        "lookback_high_days": 10,
        "range_window_days": 25,
        "range_multiplier": 2.5,
        "ibs_threshold": 0.3,
        "ibs_formula": "(close - low) / (high - low)",
    }
    assert payload["rules"]["exit"] == {"close_gt_yesterday_high": True}
    assert payload["metrics"]["SPY"]["win_rate_pct"] == 75.0
    assert payload["metrics"]["QQQ"]["final_capital"] == 365740.47


def test_fixture_parity_check_is_deterministic() -> None:
    module = _load_module()
    fixture = _load_fixture()
    payload = module.parse_reference_markdown(_reference_markdown_sample(), source_path="sample.md")

    mismatches = module.compare_reference_payload_to_fixture(payload, fixture)
    assert mismatches == []


def test_run_metric_validation_obeys_explicit_tolerances() -> None:
    module = _load_module()
    fixture = _load_fixture()
    run_payload = {
        "per_symbol_metrics": [
            {
                "symbol": "SPY",
                "metrics": {
                    "total_return_pct": 3.3480,
                    "annualized_return_pct": 0.0773,
                    "profit_factor": 2.00,
                    "win_rate": 0.7490,
                    "max_drawdown_pct": -0.1530,
                    "invested_pct": 0.2100,
                    "trade_count": 242,
                    "ending_equity": 433000.0,
                },
            },
            {
                "symbol": "QQQ",
                "metrics": {
                    "total_return_pct": 2.6600,
                    "annualized_return_pct": 0.0920,
                    "profit_factor": 2.10,
                    "win_rate": 0.7080,
                    "max_drawdown_pct": -0.1180,
                    "invested_pct": 0.1650,
                    "trade_count": 186,
                    "ending_equity": 366000.0,
                },
            },
        ]
    }

    run_metrics = module.normalize_run_metrics_payload(run_payload)
    assert module.validate_run_metrics_against_fixture(run_metrics, fixture) == []

    run_payload["per_symbol_metrics"][0]["metrics"]["trade_count"] = 250
    run_metrics = module.normalize_run_metrics_payload(run_payload)
    mismatches = module.validate_run_metrics_against_fixture(run_metrics, fixture)
    assert any("SPY.total_trades" in item for item in mismatches)


def test_main_skips_when_reference_markdown_missing(tmp_path: Path, capsys) -> None:
    module = _load_module()
    fixture_path = _fixture_path()
    missing_reference = tmp_path / "missing_reference.md"

    exit_code = module.main(
        [
            "--fixture-path",
            str(fixture_path),
            "--reference-path",
            str(missing_reference),
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "SKIP: reference markdown path not found" in output
