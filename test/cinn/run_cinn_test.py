#!/usr/bin/env python3
"""Check that the CINN compiler keeps PaddleScience cases accurate and makes them faster.

For every case in test/cinn/cases the parent process launches two workers, one
in plain dynamic graph mode and one with to_static + CINN, because the compiler
FLAGS must be set before paddle is imported. Both workers start from the same
initial weights and the same synthetic batch (the dynamic worker dumps them,
the CINN worker loads them) and measure:

    infer  model forward under no_grad: output diff and per-iteration time
    train  forward + loss (incl. PDE residuals) + backward + Adam step:
           per-step loss diff and per-iteration time

Usage:
    python test/cinn/run_cinn_test.py --list
    python test/cinn/run_cinn_test.py                          # all cases, GPU 0
    python test/cinn/run_cinn_test.py --cases ldc,laplace --devices 0,1,2,3
    python test/cinn/run_cinn_test.py --iters 50 --warmup 10 --stages infer
    python test/cinn/run_cinn_test.py --cases ntopo --output-dir copied --reuse-inputs
"""
import argparse
import concurrent.futures
import importlib
import json
import os
import queue
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent.parent  # PaddleScience root, so `import ppsci` works
for path in (str(REPO_ROOT), str(TEST_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

RESULT_PREFIX = "[CINN_TEST_RESULT] "
SEED = 2024
COMPILER_ENV_KEYS = (
    "FLAGS_prim_enable_dynamic",
    "FLAGS_prim_all",
    "FLAGS_use_cinn",
    "FLAGS_print_ir",
)
CINN_ENV = {
    "FLAGS_prim_enable_dynamic": "true",
    "FLAGS_prim_all": "true",
    "FLAGS_use_cinn": "true",
    "FLAGS_print_ir": "False",
}
STAGES = ("infer", "train")
# Losses that converge to ~0 make a plain relative difference meaningless: both
# runs are at fp32 noise level, yet dividing by the tiny reference inflates the
# error to O(1). Flooring the denominator is the same as allowing an absolute
# difference of train_step_rtol * LOSS_REL_DIFF_FLOOR. Every model we track has
# a window loss either below 1e-8 or above 4e-4, so this value sits in the gap.
LOSS_REL_DIFF_FLOOR = 1e-5


def list_cases():
    return sorted(
        p.stem for p in (TEST_DIR / "cases").glob("*.py") if not p.stem.startswith("_")
    )


# ---------------------------------------------------------------------- worker
def flatten(obj, prefix="out"):
    """Tensor / dict / list output -> {name: Tensor}."""
    import paddle

    if isinstance(obj, paddle.Tensor):
        return {prefix: obj}
    flat = {}
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = ((str(i), v) for i, v in enumerate(obj))
    else:
        return flat
    for key, value in items:
        flat.update(flatten(value, f"{prefix}.{key}"))
    return flat


def save_batch(path, inputs, labels):
    arrays, grads = {}, {}
    for group, tensors in (("inputs", inputs), ("labels", labels)):
        for key, t in tensors.items():
            arrays[f"{group}/{key}"] = t.numpy()
            grads[f"{group}/{key}"] = not t.stop_gradient
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(grads))


def load_batch(path):
    import paddle

    grads = json.loads(path.with_suffix(".json").read_text())
    inputs, labels = {}, {}
    with np.load(path) as data:
        for name in data.files:
            group, key = name.split("/", 1)
            t = paddle.to_tensor(data[name])
            t.stop_gradient = not grads[name]
            (inputs if group == "inputs" else labels)[key] = t
    return inputs, labels


def disable_dropout(model):
    """Dropout draws differ between dynamic and static mode, which would mask
    real numeric differences, so every nn.Dropout* layer is set to p=0."""
    import paddle

    dropout_types = (
        paddle.nn.Dropout,
        paddle.nn.Dropout2D,
        paddle.nn.Dropout3D,
        paddle.nn.AlphaDropout,
    )
    count = 0
    for layer in model.sublayers(include_self=True):
        if isinstance(layer, dropout_types):
            layer.p = 0.0
            count += 1
    return count


def timed_loop(step_fn, total_iters, mode, stage):
    import paddle

    times, values = [], []
    for step in range(total_iters):
        start = time.perf_counter()
        value = step_fn()
        paddle.device.synchronize()
        times.append(time.perf_counter() - start)
        values.append(value)
        loss_text = f" loss {value:.8g}" if stage == "train" else ""
        print(
            f"[{mode}/{stage}] iter {step + 1}/{total_iters}"
            f" time {times[-1]:.5f}s{loss_text}",
            flush=True,
        )
    return times, values


def time_stats(times, warmup):
    steady = times[warmup:]
    return {
        "first_iter_seconds": times[0],
        "steady_mean_seconds": statistics.mean(steady),
        "steady_median_seconds": statistics.median(steady),
        "steady_stdev_seconds": statistics.stdev(steady) if len(steady) > 1 else 0.0,
    }


def worker(case_name, mode, stages, iters, warmup, work_dir, reuse_inputs=False):
    """Run one case in one mode and print its result as a JSON line.

    With reuse_inputs the dynamic worker also loads init.pdparams / batch.npz from
    work_dir instead of regenerating them, e.g. files copied from another machine."""
    import paddle

    if mode == "cinn" and paddle.is_compiled_with_custom_device("iluvatar_gpu"):
        # PirInterpreter waits for the device after every run on a custom place,
        # which keeps the host from queuing the next step while the current one
        # is still running. Dynamic mode has no such barrier, so leaving it on
        # would charge the static graph for a serialisation the baseline never
        # pays. Nothing here copies results into host memory, the case where
        # the wait is load-bearing.
        paddle.set_flags({"FLAGS_custom_device_pir_interpreter_sync": False})

    case = importlib.import_module(f"cases.{case_name}")

    class InferNet(paddle.nn.Layer):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs):
            if hasattr(case, "forward"):
                return case.forward(self.model, inputs)
            return self.model(inputs)

    class LossNet(paddle.nn.Layer):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs, labels):
            return case.compute_loss(self.model, inputs, labels)

    def compile_net(net):
        if mode != "cinn":
            return net
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        return paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)

    paddle.seed(SEED)
    np.random.seed(SEED)
    build_start = time.perf_counter()
    model = case.build_model()
    dropout_layers = disable_dropout(model)
    init_path = work_dir / "init.pdparams"
    batch_path = work_dir / "batch.npz"
    if mode == "dynamic" and not reuse_inputs:
        inputs, labels = case.build_batch()
        paddle.save(model.state_dict(), str(init_path))
        save_batch(batch_path, inputs, labels)
    else:
        model.set_state_dict(paddle.load(str(init_path)))
        inputs, labels = load_batch(batch_path)
    if hasattr(case, "prepare"):
        case.prepare(model)  # e.g. lambdify sympy equations, outside of to_static
    result = {
        "case": case_name,
        "mode": mode,
        "batch_size": case.BATCH_SIZE,
        "num_params": int(sum(int(np.prod(p.shape)) for p in model.parameters())),
        "dropout_layers_disabled": dropout_layers,
        "build_seconds": time.perf_counter() - build_start,
    }
    total_iters = iters + warmup

    if "infer" in stages:
        model.eval()
        net = compile_net(InferNet(model))
        last = {}

        def infer_step():
            with paddle.no_grad():
                last["out"] = net(inputs)

        times, _ = timed_loop(infer_step, total_iters, mode, "infer")
        flat = flatten(last.pop("out"))
        np.savez(
            work_dir / f"infer_{mode}.npz",
            **{k: v.astype("float32").numpy() for k, v in flat.items()},
        )
        result["infer"] = time_stats(times, warmup)

    if "train" in stages:
        model.set_state_dict(paddle.load(str(init_path)))
        model.train()
        net = compile_net(LossNet(model))
        optimizer = paddle.optimizer.Adam(
            getattr(case, "LEARNING_RATE", 1e-3), parameters=model.parameters()
        )
        grad_inputs = [t for t in inputs.values() if not t.stop_gradient]

        def train_step():
            loss = net(inputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            for t in grad_inputs:
                t.clear_gradient()
            return float(loss)

        times, losses = timed_loop(train_step, total_iters, mode, "train")
        result["train"] = {**time_stats(times, warmup), "losses": losses}

    print(RESULT_PREFIX + json.dumps(result), flush=True)


# ---------------------------------------------------------------------- parent
def make_env(mode, device):
    env = os.environ.copy()
    for key in COMPILER_ENV_KEYS:
        env.pop(key, None)
    # NVIDIA / Iluvatar read CUDA_VISIBLE_DEVICES, Hygon DCU reads HIP_VISIBLE_DEVICES
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["HIP_VISIBLE_DEVICES"] = str(device)
    env["PYTHONUNBUFFERED"] = "1"
    if mode == "cinn":
        env.update(CINN_ENV)
    return env


def run_worker(case_name, mode, args, device, work_dir):
    command = [
        args.python,
        str(Path(__file__).resolve()),
        "--worker",
        mode,
        "--cases",
        case_name,
        "--stages",
        ",".join(args.stages),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--work-dir",
        str(work_dir),
    ]
    if args.reuse_inputs:
        command.append("--reuse-inputs")
    log_path = work_dir / f"{mode}.log"
    result = None
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(command) + "\n")
        log_file.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=make_env(mode, device),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            return_code = "timeout"
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX) :])
    if return_code != 0 or result is None:
        return None, f"exit={return_code}: {last_error_line(log_path)}"
    result["wall_seconds"] = time.perf_counter() - started
    return result, None


def last_error_line(log_path):
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        if "Error" in line or "error" in line:
            return line.strip()[:300]
    return lines[-1].strip()[:300] if lines else "no output"


def rel_diff(a, b):
    return abs(a - b) / max(abs(a), LOSS_REL_DIFF_FLOOR)


def train_window_stats(ref, out, start_step, end_step, step_rtol):
    """Compare losses over the inclusive 1-based [start_step, end_step] range."""
    if len(ref) != len(out):
        raise ValueError(f"loss count differs: {len(ref)} vs {len(out)}")
    if start_step < 1 or end_step < start_step or end_step > len(ref):
        raise ValueError(
            f"invalid train loss window {start_step}-{end_step} for {len(ref)} steps"
        )
    ref_window = ref[start_step - 1 : end_step]
    out_window = out[start_step - 1 : end_step]
    diffs = [rel_diff(a, b) for a, b in zip(ref_window, out_window)]
    abs_diffs = [abs(a - b) for a, b in zip(ref_window, out_window)]
    return {
        "dynamic_loss_mean": statistics.mean(ref_window),
        "cinn_loss_mean": statistics.mean(out_window),
        "loss_rel_diff_exceed_count": sum(diff > step_rtol for diff in diffs),
        "loss_abs_diff_exceed_count": sum(diff > step_rtol for diff in abs_diffs),
    }


def plot_train_loss_comparison(work_dir, case_name, ref, out, args):
    """Save loss and error comparisons for the configured step window."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(1, len(ref) + 1)
    window = slice(args.train_window_start - 1, args.train_window_end)
    window_steps = steps[window]
    dynamic_mean = statistics.mean(ref[window])
    cinn_mean = statistics.mean(out[window])

    signed_rel_diffs = [
        (cinn_loss - dynamic_loss) / max(abs(dynamic_loss), LOSS_REL_DIFF_FLOOR)
        for dynamic_loss, cinn_loss in zip(ref, out)
    ]
    signed_diffs = [
        cinn_loss - dynamic_loss for dynamic_loss, cinn_loss in zip(ref, out)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6), constrained_layout=True)
    axes = axes.flat
    axes[0].plot(steps, ref, label="Dynamic graph", linewidth=1.2)
    axes[0].plot(steps, out, label="to_static + CINN", linewidth=1.2)
    axes[0].set_title("Loss, all training steps")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(window_steps, ref[window], label="Dynamic graph", linewidth=1.2)
    axes[1].plot(window_steps, out[window], label="to_static + CINN", linewidth=1.2)
    axes[1].set_title(
        f"Loss, steps {args.train_window_start}-{args.train_window_end}"
    )
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(
        window_steps,
        signed_rel_diffs[window],
        label="(CINN - dynamic) / |dynamic|",
        linewidth=1.2,
    )
    for threshold in (args.train_step_rtol, -args.train_step_rtol):
        axes[2].axhline(threshold, color="tab:red", linestyle="--")
    axes[2].set_title("Signed relative loss difference")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Relative difference")
    axes[2].set_yscale("symlog", linthresh=args.train_step_rtol)
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    axes[3].plot(
        window_steps,
        signed_diffs[window],
        label="CINN - dynamic",
        linewidth=1.2,
    )
    for threshold in (args.train_step_rtol, -args.train_step_rtol):
        axes[3].axhline(threshold, color="tab:red", linestyle="--")
    axes[3].set_title("Signed loss difference")
    axes[3].set_xlabel("Training step")
    axes[3].set_ylabel("Loss difference")
    axes[3].set_yscale("symlog", linthresh=args.train_step_rtol)
    axes[3].grid(alpha=0.25)
    axes[3].legend()

    fig.suptitle(
        f"{case_name}: window mean loss {dynamic_mean:.6g} / {cinn_mean:.6g} "
        f"· threshold {args.train_step_rtol:.0e}"
    )
    fig.savefig(work_dir / "train_loss_comparison.png", dpi=160)
    plt.close(fig)


def compare_infer(work_dir):
    with np.load(work_dir / "infer_dynamic.npz") as ref, np.load(
        work_dir / "infer_cinn.npz"
    ) as out:
        if set(ref.files) != set(out.files):
            return {"error": f"output keys differ: {ref.files} vs {out.files}"}
        worst_abs, worst_rel = 0.0, 0.0
        dynamic_sum, cinn_sum, output_size = 0.0, 0.0, 0
        for key in ref.files:
            a, b = ref[key].astype("float64"), out[key].astype("float64")
            if a.shape != b.shape:
                return {"error": f"{key} shape {a.shape} vs {b.shape}"}
            if not np.isfinite(b).all() and np.isfinite(a).all():
                return {"error": f"{key} has non-finite values under CINN"}
            dynamic_sum += float(a.sum())
            cinn_sum += float(b.sum())
            output_size += a.size
            diff = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
            scale = float(np.nanmax(np.abs(a))) if a.size else 0.0
            worst_abs = max(worst_abs, diff)
            worst_rel = max(worst_rel, diff / max(scale, 1e-12))
    return {
        "max_abs_diff": worst_abs,
        "max_rel_diff": worst_rel,
        "dynamic_final_output_mean": dynamic_sum / output_size if output_size else None,
        "cinn_final_output_mean": cinn_sum / output_size if output_size else None,
    }


def record_infer_means(work_dir, infer):
    """Add available final-inference means to each worker result JSON."""
    for mode, mean in (
        ("dynamic", infer.get("dynamic_final_output_mean")),
        ("cinn", infer.get("cinn_final_output_mean")),
    ):
        if mean is None:
            continue
        log_path = work_dir / f"{mode}.log"
        lines = log_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.startswith(RESULT_PREFIX):
                result = json.loads(line[len(RESULT_PREFIX) :])
                result["infer"]["final_output_mean"] = mean
                lines[index] = RESULT_PREFIX + json.dumps(result)
                break
        else:
            raise ValueError(f"worker result missing from {log_path}")
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_case(dynamic, cinn, work_dir, args):
    """Accuracy and speed verdict per stage."""
    report = {}
    if "infer" in dynamic:
        acc = compare_infer(work_dir)
        acc["pass"] = "error" not in acc and acc["max_rel_diff"] <= args.infer_rtol
        report["infer"] = acc
    if "train" in dynamic:
        ref, out = dynamic["train"]["losses"], cinn["train"]["losses"]
        diffs = [rel_diff(a, b) for a, b in zip(ref, out)]
        finite = all(np.isfinite(out)) or not all(np.isfinite(ref))
        window_stats = train_window_stats(
            ref,
            out,
            args.train_window_start,
            args.train_window_end,
            args.train_step_rtol,
        )
        report["train"] = {
            "dynamic_mean_loss": window_stats["dynamic_loss_mean"],
            "cinn_mean_loss": window_stats["cinn_loss_mean"],
            "first_loss_rel_diff": diffs[0],
            "final_loss_rel_diff": diffs[-1],
            "max_loss_rel_diff": max(diffs),
            "loss_rel_diff_exceed_count": window_stats["loss_rel_diff_exceed_count"],
            "loss_abs_diff_exceed_count": window_stats["loss_abs_diff_exceed_count"],
            "pass": bool(
                finite
                and diffs[0] <= args.train_first_rtol
                and diffs[-1] <= args.train_final_rtol
                and window_stats["loss_rel_diff_exceed_count"]
                < args.train_max_exceed_count
            ),
        }
    for stage in report:
        speedup = (
            dynamic[stage]["steady_mean_seconds"] / cinn[stage]["steady_mean_seconds"]
        )
        report[stage].update(
            {
                "dynamic_seconds": dynamic[stage]["steady_mean_seconds"],
                "cinn_seconds": cinn[stage]["steady_mean_seconds"],
                "cinn_first_iter_seconds": cinn[stage]["first_iter_seconds"],
                "speedup": speedup,
            }
        )
        if stage == "train":
            report[stage].update(
                {
                    "dynamic_fps": dynamic["batch_size"]
                    / dynamic[stage]["steady_mean_seconds"],
                    "cinn_fps": dynamic["batch_size"]
                    / cinn[stage]["steady_mean_seconds"],
                }
            )
    return report


def run_case(case_name, args, devices, out_dir):
    device = devices.get()
    try:
        work_dir = out_dir / case_name
        work_dir.mkdir(parents=True, exist_ok=True)
        summary = {"case": case_name, "device": device}
        dynamic, error = run_worker(case_name, "dynamic", args, device, work_dir)
        if error:
            summary.update(status="CASE_ERROR", error=error)
            return summary
        cinn, error = run_worker(case_name, "cinn", args, device, work_dir)
        if error:
            summary.update(status="CINN_ERROR", error=error)
            return summary
        summary.update(compare_case(dynamic, cinn, work_dir, args))
        if "infer" in args.stages:
            record_infer_means(work_dir, summary["infer"])
        if "train" in args.stages:
            try:
                plot_train_loss_comparison(
                    work_dir,
                    case_name,
                    dynamic["train"]["losses"],
                    cinn["train"]["losses"],
                    args,
                )
            except Exception as exc:
                summary["plot_error"] = f"{type(exc).__name__}: {exc}"
        stages = [summary[s] for s in args.stages]
        if not all(s["pass"] for s in stages):
            summary["status"] = "ACC_FAIL"
        elif all(s["speedup"] >= 1.0 for s in stages):
            summary["status"] = "PASS"
        else:
            summary["status"] = "SLOWER"
        summary["num_params"] = dynamic["num_params"]
        return summary
    finally:
        devices.put(device)


def fmt_stage(summary, stage):
    s = summary.get(stage)
    if not s:
        return ["-"] * (7 if stage == "train" else 4)
    if stage == "infer":
        acc = s.get("error") or f"{s['max_rel_diff']:.1e}"
        before_acc = [
            "-"
            if s.get("dynamic_final_output_mean") is None
            else f"{s['dynamic_final_output_mean']:.8g}/"
            f"{s['cinn_final_output_mean']:.8g}"
        ]
        count, fps = [], []
    else:
        acc = f"{s['first_loss_rel_diff']:.1e}/{s['final_loss_rel_diff']:.1e}"
        before_acc = [
            f"{s['dynamic_mean_loss']:.8g}/{s['cinn_mean_loss']:.8g}",
        ]
        count = [
            str(s["loss_rel_diff_exceed_count"]),
            str(s["loss_abs_diff_exceed_count"]),
        ]
        fps = [f"{s['dynamic_fps']:.2f}/{s['cinn_fps']:.2f}"]
    return [
        *before_acc,
        acc,
        *count,
        f"{s['dynamic_seconds'] * 1e3:.2f}/{s['cinn_seconds'] * 1e3:.2f}",
        *fps,
        f"{s['speedup']:.2f}x",
    ]


def write_summary(summaries, out_dir, args):
    header = ["case", "status"]
    for stage in args.stages:
        start = len(header)
        acc = "infer rel diff" if stage == "infer" else "train loss rel diff first/final"
        header += [acc, f"{stage} ms dyn/cinn", f"{stage} speedup"]
        if stage == "infer":
            header.insert(start, "infer final output mean dyn/cinn")
        else:
            header.insert(
                start,
                (
                    "train loss mean "
                    f"(step {args.train_window_start}-{args.train_window_end}) dyn/cinn"
                ),
            )
            header.insert(
                start + 2,
                f"train loss rel diff > {args.train_step_rtol:.1e} count "
                f"(step {args.train_window_start}-{args.train_window_end})",
            )
            header.insert(
                start + 3,
                f"train loss abs diff > {args.train_step_rtol:.1e} count "
                f"(step {args.train_window_start}-{args.train_window_end})",
            )
            header.insert(start + 5, "train FPS dyn/cinn")
    rows = []
    for s in sorted(summaries, key=lambda x: x["case"]):
        row = [s["case"], s["status"]]
        for stage in args.stages:
            row += fmt_stage(s, stage)
        rows.append(row)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    errors = [s for s in summaries if "error" in s]
    if errors:
        lines += ["", "Errors:"]
        lines += [f"- {s['case']} ({s['status']}): {s['error']}" for s in errors]
    counts = {}
    for s in summaries:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    lines += ["", "Totals: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))]
    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text + "\n", encoding="utf-8")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "cases": summaries}, f, indent=2)
    print("\n" + text)
    print(f"\nresults: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cases", default="all", help="comma separated, or 'all'")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument("--stages", default="infer,train", help="infer,train")
    parser.add_argument("--iters", type=int, default=20, help="timed iters per stage")
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="leading iters excluded from timing (compilation lands here)",
    )
    parser.add_argument("--devices", default="0", help="GPU ids, one case per GPU")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per worker")
    parser.add_argument("--infer-rtol", type=float, default=1e-3)
    parser.add_argument("--train-first-rtol", type=float, default=1e-3)
    parser.add_argument("--train-final-rtol", type=float, default=1e-2)
    parser.add_argument(
        "--train-window-start",
        type=int,
        default=400,
        help="inclusive first step used for train-loss accuracy statistics",
    )
    parser.add_argument(
        "--train-window-end",
        type=int,
        default=500,
        help="inclusive last step used for train-loss accuracy statistics",
    )
    parser.add_argument(
        "--train-step-rtol",
        type=float,
        default=1e-2,
        help="per-step relative-error threshold in the train-loss window",
    )
    parser.add_argument(
        "--train-max-exceed-count",
        type=int,
        default=10,
        help="training passes only if the count above --train-step-rtol is below this",
    )
    parser.add_argument("--output-dir", help="defaults to test/cinn/outputs/<stamp>")
    parser.add_argument(
        "--reuse-inputs",
        action="store_true",
        help="load <output-dir>/<case>/init.pdparams + batch.npz instead of regenerating",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--worker", choices=["dynamic", "cinn"], help=argparse.SUPPRESS)
    parser.add_argument("--work-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.stages = [s for s in args.stages.split(",") if s]
    if not set(args.stages) <= set(STAGES):
        parser.error(f"--stages must be a subset of {STAGES}")
    return args


def main():
    args = parse_args()
    if args.worker:
        worker(
            args.cases,
            args.worker,
            args.stages,
            args.iters,
            args.warmup,
            Path(args.work_dir),
            args.reuse_inputs,
        )
        return
    if args.list:
        print("\n".join(list_cases()))
        return
    if args.warmup < 1 or args.iters < 2:
        raise ValueError("need warmup >= 1 and iters >= 2")
    if args.train_step_rtol <= 0:
        raise ValueError("--train-step-rtol must be positive")
    total_iters = args.iters + args.warmup
    if "train" in args.stages and not (
        1 <= args.train_window_start <= args.train_window_end <= total_iters
    ):
        raise ValueError(
            f"train loss window must be within 1-{total_iters}, got "
            f"{args.train_window_start}-{args.train_window_end}"
        )

    available = list_cases()
    cases = available if args.cases == "all" else args.cases.split(",")
    unknown = sorted(set(cases) - set(available))
    if unknown:
        raise ValueError(f"unknown cases: {unknown}")
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (TEST_DIR / "outputs" / time.strftime("%Y%m%d_%H%M%S"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    devices = queue.Queue()
    device_ids = [d for d in args.devices.split(",") if d]
    for d in device_ids:
        devices.put(d)
    summaries = []
    with concurrent.futures.ThreadPoolExecutor(len(device_ids)) as pool:
        futures = {pool.submit(run_case, c, args, devices, out_dir): c for c in cases}
        for future in concurrent.futures.as_completed(futures):
            s = future.result()
            summaries.append(s)
            print(
                f"[{len(summaries)}/{len(cases)}] {s['case']}: {s['status']}",
                flush=True,
            )
    write_summary(summaries, out_dir, args)


if __name__ == "__main__":
    main()
