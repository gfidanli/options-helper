from __future__ import annotations

from itertools import product


def required_feature_columns_for_strategy(
    strategy: str,
    strat_cfg: dict,
) -> list[str]:
    required: set[str] = set()

    def _vals(key: str) -> list:
        values: list = []
        defaults = strat_cfg.get("defaults", {})
        if key in defaults:
            values.append(defaults[key])
        values.extend((strat_cfg.get("search_space", {}) or {}).get(key, []) or [])
        # Preserve order but unique.
        seen = set()
        out = []
        for v in values:
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
        return out

    if strategy == "TrendPullbackATR":
        atr_windows = [int(v) for v in _vals("atr_window")]
        sma_windows = [int(v) for v in _vals("sma_window")]
        z_windows = [int(v) for v in _vals("z_window")]

        for w in atr_windows:
            required.add(f"atr_{w}")
        for w in z_windows:
            required.add(f"zscore_{w}")
        for s, a in product(sma_windows, atr_windows):
            required.add(f"extension_atr_{s}_{a}")
        required.add("weekly_trend_up")

    elif strategy == "MeanReversionBollinger":
        bb_windows = [int(v) for v in _vals("bb_window")]
        bb_devs = [float(v) for v in _vals("bb_dev")]
        atr_windows = [int(v) for v in _vals("atr_window")]

        def _dev_label(dev: float) -> str:
            if float(dev).is_integer():
                return str(int(float(dev)))
            return f"{float(dev):g}".replace(".", "p")

        for w in bb_windows:
            required.add(f"bb_mavg_{w}")
        for w in bb_windows:
            for dev in bb_devs:
                lab = _dev_label(dev)
                required.add(f"bb_lband_{w}_{lab}")
                required.add(f"bb_pband_{w}_{lab}")
        for w in atr_windows:
            required.add(f"atr_{w}")
        required.add("weekly_trend_up")

    return sorted(required)

