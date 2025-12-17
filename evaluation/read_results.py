"""
Summarize multiple evaluation JSON outputs (e.g., different seeds) into mean ± std tables.

Typical usage (3 seeds, separate output files):
  python evaluation/summarize_results.py --inputs "results/cxt_len_ablation_*_*.json"
  python evaluation/summarize_results.py --inputs "results/resident_layer_ablation_*_*.json"

You can pass multiple patterns/paths:
  python evaluation/summarize_results.py --inputs results/run1.json results/run2.json

This script aggregates across *runs/files* (e.g., seeds). Note:
- The per-file mean/std in the JSON are usually over dataset samples inside that run.
- The mean ± std printed by this script is over the provided runs/files.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _std(xs: List[float]) -> float:
    # sample std (ddof=1) when possible; otherwise 0.0
    n = len(xs)
    if n <= 1:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _fmt(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _flatten_inputs(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for item in inputs:
        matched = sorted(glob.glob(item))
        if matched:
            paths.extend(matched)
        else:
            paths.append(item)
    # de-dup while preserving order
    out: List[str] = []
    seen = set()
    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _detect_x_key_from_sweep(sweep: Dict[str, Any]) -> Optional[str]:
    """
    Detect which field is the sweep variable (x-axis).

    Many sweep entries include multiple fields (e.g., prefill_length and resident_layers).
    We choose the key whose values *vary across the sweep entries*.
    """
    candidates = ("prefill_length", "gen_len", "on_chip_layers", "resident_layers")
    # gather values per candidate
    values: Dict[str, set] = {k: set() for k in candidates}
    for entry in sweep.values():
        if not isinstance(entry, dict):
            continue
        for k in candidates:
            if k in entry:
                try:
                    values[k].add(int(entry[k]))
                except Exception:
                    pass
    varying = [k for k in candidates if len(values[k]) > 1]
    if len(varying) == 1:
        return varying[0]
    if len(varying) > 1:
        # Prefer the most common experiment x-axis if multiple vary (rare).
        for k in candidates:
            if k in varying:
                return k
    # If none vary, fall back to a key that exists.
    for k in candidates:
        if len(values[k]) == 1:
            return k
    return None


def _get_metric(block: Dict[str, Any], metric: str) -> float:
    if metric == "throughput":
        if "mean_throughput_tok_per_s" not in block:
            raise KeyError("mean_throughput_tok_per_s")
        return float(block["mean_throughput_tok_per_s"])
    if metric == "latency":
        if "mean_latency_s" not in block:
            raise KeyError("mean_latency_s")
        return float(block["mean_latency_s"])
    if metric == "accepted":
        if "mean_accepted_tokens" not in block:
            raise KeyError("mean_accepted_tokens")
        return float(block["mean_accepted_tokens"])
    raise ValueError(f"Unknown metric: {metric}")


@dataclass(frozen=True)
class Row:
    x: int
    baseline: float
    quant: float
    delta: float
    ratio: Optional[float]


def _load_rows(path: str, metric: str) -> List[Row]:
    with open(path, "r") as f:
        data = json.load(f)

    methods = data.get("methods", {})
    mspeckv = methods.get("mspeckv_kv_quant", {})
    sweep = mspeckv.get("sweep", {})
    if not isinstance(sweep, dict) or len(sweep) == 0:
        raise ValueError(f"{path}: no methods.mspeckv_kv_quant.sweep found")

    x_key = _detect_x_key_from_sweep(sweep)

    rows: List[Row] = []
    # Support both JSON schemas:
    # - Old: methods['mspeckv_kv_quant']['sweep'][x] has nested {'baseline': {...}, 'quant': {...}}
    # - New: methods['mspeckv']['sweep'][x] is baseline, methods['mspeckv_kv_quant']['sweep'][x] is quant (flat)
    is_old_nested = False
    for _k, _entry in sweep.items():
        if isinstance(_entry, dict) and ("baseline" in _entry or "quant" in _entry):
            is_old_nested = True
            break

    mspeckv_baseline_sweep = methods.get("mspeckv", {}).get("sweep", {}) if isinstance(methods, dict) else {}

    for k, entry in sweep.items():
        if not isinstance(entry, dict):
            continue
        x_val = entry.get(x_key) if x_key is not None else None
        if x_val is None:
            # fall back to dict key
            try:
                x_val = int(k)
            except Exception:
                raise ValueError(f"{path}: cannot infer x value for sweep key {k!r}")
        x_int = int(x_val)

        if is_old_nested:
            baseline_block = entry.get("baseline", {})
            quant_block = entry.get("quant", {})
        else:
            # New flat format: kv-quant entry is `entry`, baseline is stored under methods['mspeckv']['sweep'].
            if not isinstance(mspeckv_baseline_sweep, dict):
                raise ValueError(
                    f"{path}: expected methods.mspeckv.sweep for baseline comparison, "
                    f"but found {type(mspeckv_baseline_sweep).__name__}"
                )
            baseline_block = mspeckv_baseline_sweep.get(k, {})
            quant_block = entry

        try:
            b = _get_metric(baseline_block, metric)
        except KeyError as e:
            raise KeyError(
                f"{path}: missing {e!s} in baseline block. "
                f"Schema hint: old format needs sweep[x].baseline.*, new format needs methods.mspeckv.sweep[x].*"
            ) from e
        try:
            q = _get_metric(quant_block, metric)
        except KeyError as e:
            raise KeyError(
                f"{path}: missing {e!s} in quant block. "
                f"Schema hint: old format needs sweep[x].quant.*, new format needs methods.mspeckv_kv_quant.sweep[x].*"
            ) from e
        d = q - b

        # Define a consistent "ratio" where >1 means better for the metric.
        # - throughput: higher is better => ratio = quant / baseline
        # - latency: lower is better => ratio = baseline / quant
        # - accepted: not a speed metric => no ratio
        r: Optional[float]
        if metric == "throughput":
            r = (q / b) if b != 0 else float("nan")
        elif metric == "latency":
            r = (b / q) if q != 0 else float("nan")
        elif metric == "accepted":
            r = None
        else:
            raise ValueError(f"Unknown metric: {metric}")

        rows.append(Row(x=x_int, baseline=b, quant=q, delta=d, ratio=r))

    rows.sort(key=lambda r: r.x)
    return rows


def _group_by_x(all_rows: List[List[Row]]) -> Dict[int, List[Row]]:
    grouped: Dict[int, List[Row]] = {}
    for run_rows in all_rows:
        for r in run_rows:
            grouped.setdefault(r.x, []).append(r)
    return grouped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help='JSON file paths or glob patterns, e.g. "results/cxt_len_ablation_*.json"',
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["throughput"],
        help='Metrics to summarize. Use one or more of: throughput latency accepted, or "all". Default: throughput',
    )
    ap.add_argument("--digits", type=int, default=2, help="Number formatting digits")
    args = ap.parse_args()

    paths = _flatten_inputs(args.inputs)
    paths = [p for p in paths if p.endswith(".json")]
    if len(paths) == 0:
        raise SystemExit("No .json inputs found.")

    allowed = {"throughput", "latency", "accepted"}
    metrics_in = [m.strip().lower() for m in args.metrics]
    if len(metrics_in) == 1 and metrics_in[0] == "all":
        metrics = ["throughput", "latency", "accepted"]
    else:
        unknown = [m for m in metrics_in if m not in allowed]
        if unknown:
            raise SystemExit(f"Unknown metrics: {unknown}. Allowed: {sorted(allowed)} or 'all'.")
        metrics = metrics_in

    print(f"Inputs: {len(paths)} run(s)")
    print(f"Files: {', '.join(paths)}")

    for metric in metrics:
        all_rows: List[List[Row]] = []
        for p in paths:
            all_rows.append(_load_rows(p, metric=metric))

        grouped = _group_by_x(all_rows)
        xs = sorted(grouped.keys())

        print("\n")
        print(f"## Metric: {metric}")
        print("")
        if metric == "accepted":
            print("| x | baseline | quant | delta (quant-baseline) |")
            print("|---:|---:|---:|---:|")
        elif metric == "latency":
            print("| x | baseline | quant | delta (quant-baseline) | speedup (baseline/quant) |")
            print("|---:|---:|---:|---:|---:|")
        else:
            print("| x | baseline | quant | delta (quant-baseline) | speedup (quant/baseline) |")
            print("|---:|---:|---:|---:|---:|")
        for x in xs:
            rs = grouped[x]
            b_list = [r.baseline for r in rs]
            q_list = [r.quant for r in rs]
            d_list = [r.delta for r in rs]
            r_list = [r.ratio for r in rs if r.ratio is not None]

            b_m, b_s = _mean(b_list), _std(b_list)
            q_m, q_s = _mean(q_list), _std(q_list)
            d_m, d_s = _mean(d_list), _std(d_list)

            if metric == "accepted":
                print(
                    f"| {x} | {_fmt(b_m, b_s, args.digits)} | {_fmt(q_m, q_s, args.digits)} | "
                    f"{_fmt(d_m, d_s, args.digits)} |"
                )
            else:
                r_m, r_s = _mean(r_list), _std(r_list)
                print(
                    f"| {x} | {_fmt(b_m, b_s, args.digits)} | {_fmt(q_m, q_s, args.digits)} | "
                    f"{_fmt(d_m, d_s, args.digits)} | {_fmt(r_m, r_s, args.digits)} |"
                )


if __name__ == "__main__":
    main()


