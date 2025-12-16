"""
Aggregate multi-run (e.g., 3 seeds) JSON outputs into mean ± std tables.

Designed for `evaluation/cxt_len_ablation.py` outputs, including the newer schema:
  methods:
    baseline:         sweep[x] -> {mean_latency_s, mean_throughput_tok_per_s, ...}
    vanilla_specdec:  sweep[x] -> {mean_latency_s, mean_throughput_tok_per_s, mean_accepted_tokens, ...}
    triforce:         sweep[x] -> {mean_latency_s, mean_throughput_tok_per_s, mean_accepted_tokens, ...}
    triforce_kv_quant:sweep[x] -> {mean_latency_s, mean_throughput_tok_per_s, mean_accepted_tokens, ...}

Backward compatible with older nested format:
  methods['triforce_kv_quant']['sweep'][x] contains {'baseline': {...}, 'quant': {...}}

Example:
  python evaluation/read_all_results.py --inputs "results/A1/*.json" --metrics all --digits 2
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def _std(xs: Sequence[float]) -> float:
    n = len(xs)
    if n <= 1:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _fmt(mean: float, std: float, digits: int) -> str:
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
    candidates = ("prefill_length", "resident_layers", "gen_len", "on_chip_layers")
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
        for k in candidates:
            if k in varying:
                return k
    for k in candidates:
        if len(values[k]) == 1:
            return k
    return None


def _get_metric(block: Dict[str, Any], metric: str) -> float:
    if metric == "throughput":
        return float(block["mean_throughput_tok_per_s"])
    if metric == "latency":
        return float(block["mean_latency_s"])
    if metric == "accepted":
        return float(block["mean_accepted_tokens"])
    raise ValueError(f"Unknown metric: {metric}")


def _upgrade_old_schema_in_place(data: Dict[str, Any]) -> None:
    """
    Upgrade older nested schema (if present) into the newer per-method flat schema.
    """
    methods = data.get("methods")
    if not isinstance(methods, dict):
        return
    tkv = methods.get("triforce_kv_quant")
    if not isinstance(tkv, dict):
        return
    sweep = tkv.get("sweep")
    if not isinstance(sweep, dict) or len(sweep) == 0:
        return

    # Detect old nested entries
    any_old = False
    for entry in sweep.values():
        if isinstance(entry, dict) and ("baseline" in entry or "quant" in entry):
            any_old = True
            break
    if not any_old:
        return

    # Ensure methods['triforce'] exists
    triforce = methods.setdefault("triforce", {"name": "TriForce", "sweep": {}})
    if not isinstance(triforce, dict):
        methods["triforce"] = {"name": "TriForce", "sweep": {}}
        triforce = methods["triforce"]
    triforce.setdefault("sweep", {})
    if not isinstance(triforce["sweep"], dict):
        triforce["sweep"] = {}

    # Convert nested baseline/quant blocks
    for x, entry in list(sweep.items()):
        if not isinstance(entry, dict):
            continue
        if "baseline" in entry and isinstance(entry["baseline"], dict):
            base = entry["baseline"]
            triforce["sweep"][x] = {
                "prefill_length": int(entry.get("prefill_length", int(x) if str(x).isdigit() else 0) or 0),
                "mean_latency_s": base.get("mean_latency_s"),
                "std_latency_s": base.get("std_latency_s"),
                "mean_throughput_tok_per_s": base.get("mean_throughput_tok_per_s"),
                "mean_accepted_tokens": base.get("mean_accepted_tokens"),
                "mean_latency_s_per_token": base.get("mean_latency_s_per_token"),
                "resident_layers": entry.get("resident_layers"),
            }
        if "quant" in entry and isinstance(entry["quant"], dict):
            quant = entry["quant"]
            sweep[x] = {
                "prefill_length": int(entry.get("prefill_length", int(x) if str(x).isdigit() else 0) or 0),
                "mean_latency_s": quant.get("mean_latency_s"),
                "std_latency_s": quant.get("std_latency_s"),
                "mean_throughput_tok_per_s": quant.get("mean_throughput_tok_per_s"),
                "mean_accepted_tokens": quant.get("mean_accepted_tokens"),
                "mean_latency_s_per_token": quant.get("mean_latency_s_per_token"),
                "resident_layers": entry.get("resident_layers"),
            }


def _load_run(path: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Dict[str, Any]]]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected dict JSON")
    _upgrade_old_schema_in_place(data)

    methods = data.get("methods", {})
    if not isinstance(methods, dict):
        methods = {}

    sweeps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for m, mblock in methods.items():
        if not isinstance(mblock, dict):
            continue
        sweep = mblock.get("sweep", {})
        if isinstance(sweep, dict):
            # normalize: only keep dict entries
            sweeps[m] = {str(k): v for k, v in sweep.items() if isinstance(v, dict)}
    return data, sweeps


def _collect_xs(sweeps: List[Dict[str, Dict[str, Dict[str, Any]]]], methods: List[str]) -> List[str]:
    xs: set = set()
    for run in sweeps:
        for m in methods:
            ms = run.get(m, {})
            xs.update(ms.keys())
    # sort numerically when possible
    def _key(x: str):
        try:
            return (0, int(x))
        except Exception:
            return (1, x)
    return sorted(list(xs), key=_key)


def _print_metric_table(
    *,
    metric: str,
    x_label: str,
    xs: List[str],
    runs: List[Dict[str, Dict[str, Dict[str, Any]]]],
    methods: List[str],
    digits: int,
    baseline_method: str,
) -> None:
    print("\n")
    print(f"## Metric: {metric}")
    print("")

    cols = [x_label] + methods
    if metric in ("throughput", "latency"):
        cols += [f"speedup_vs_{baseline_method}:{m}" for m in methods if m != baseline_method]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---:" for _ in cols]) + " |"
    print(header)
    print(sep)

    for x in xs:
        # per method lists across runs
        vals_by_m: Dict[str, List[float]] = {m: [] for m in methods}
        for run in runs:
            for m in methods:
                block = run.get(m, {}).get(x)
                if not isinstance(block, dict):
                    continue
                try:
                    vals_by_m[m].append(_get_metric(block, metric))
                except Exception:
                    continue

        # baseline values for speedup
        b_vals = vals_by_m.get(baseline_method, [])
        b_m = _mean(b_vals) if b_vals else float("nan")

        # format cells
        row_cells: List[str] = [x]
        for m in methods:
            xs_m = vals_by_m[m]
            if not xs_m:
                row_cells.append("NA")
            else:
                row_cells.append(_fmt(_mean(xs_m), _std(xs_m), digits))

        if metric in ("throughput", "latency"):
            for m in methods:
                if m == baseline_method:
                    continue
                m_vals = vals_by_m.get(m, [])
                if not b_vals or not m_vals:
                    row_cells.append("NA")
                    continue
                m_m = _mean(m_vals)
                if metric == "throughput":
                    sp = (m_m / b_m) if b_m != 0 else float("nan")
                else:
                    sp = (b_m / m_m) if m_m != 0 else float("nan")
                row_cells.append(f"{sp:.{digits}f}x")

        print("| " + " | ".join(row_cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help='JSON file paths or glob patterns, e.g. "results/A1/*.json"',
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["throughput"],
        help='Metrics to summarize. Use one or more of: throughput latency accepted, or "all". Default: throughput',
    )
    ap.add_argument("--digits", type=int, default=2, help="Number formatting digits")
    ap.add_argument(
        "--methods",
        type=str,
        default="baseline,vanilla_specdec,triforce,triforce_kv_quant",
        help="Comma-separated methods to include (in order).",
    )
    ap.add_argument("--baseline_method", type=str, default="baseline", help="Method name to use as baseline for speedups")
    ap.add_argument("--x_key", type=str, default=None, help="Optional x-axis key override (e.g., prefill_length)")
    args = ap.parse_args()

    paths = _flatten_inputs(args.inputs)
    paths = [p for p in paths if p.endswith(".json")]
    if len(paths) == 0:
        raise SystemExit("No .json inputs found.")

    allowed_metrics = {"throughput", "latency", "accepted"}
    metrics_in = [m.strip().lower() for m in args.metrics]
    if len(metrics_in) == 1 and metrics_in[0] == "all":
        metrics = ["throughput", "latency", "accepted"]
    else:
        unknown = [m for m in metrics_in if m not in allowed_metrics]
        if unknown:
            raise SystemExit(f"Unknown metrics: {unknown}. Allowed: {sorted(allowed_metrics)} or 'all'.")
        metrics = metrics_in

    methods = [m.strip() for m in (args.methods or "").split(",") if m.strip() != ""]
    if len(methods) == 0:
        raise SystemExit("--methods resolved to empty list")
    if args.baseline_method not in methods:
        raise SystemExit(f"--baseline_method={args.baseline_method!r} must be included in --methods")

    runs: List[Dict[str, Dict[str, Dict[str, Any]]]] = []
    x_label = "x"
    x_key_override = args.x_key

    # load all runs
    for p in paths:
        data, sweeps = _load_run(p)
        runs.append(sweeps)
        if x_key_override is None:
            # infer from any available sweep
            for m in methods:
                sw = sweeps.get(m, {})
                if isinstance(sw, dict) and len(sw) > 0:
                    k = _detect_x_key_from_sweep(sw)
                    if k is not None:
                        x_label = k
                        break
    if x_key_override is not None:
        x_label = x_key_override

    xs = _collect_xs(runs, methods)

    print(f"Inputs: {len(paths)} run(s)")
    print(f"Files: {', '.join(paths)}")
    print(f"Methods: {', '.join(methods)}")

    for metric in metrics:
        _print_metric_table(
            metric=metric,
            x_label=x_label,
            xs=xs,
            runs=runs,
            methods=methods,
            digits=args.digits,
            baseline_method=args.baseline_method,
        )


if __name__ == "__main__":
    main()


