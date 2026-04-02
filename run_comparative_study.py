"""Run a comparative study across all makemore model types.

This script trains each model type one-by-one, captures sample outputs,
extracts train/test loss snapshots, and generates a comparison report.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


DEFAULT_MODELS = ["transformer", "bigram", "mlp", "rnn", "gru", "bow"]
LOSS_PATTERN = re.compile(
    r"step\s+(\d+)\s+train loss:\s+([\deE+\-.]+)\s+test loss:\s+([\deE+\-.]+)"
)
SECTION_PATTERN = re.compile(r"^\d+ samples that are (in train|in test|new):$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run makemore comparative study")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="Model types to include in the study",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=datetime.now().strftime("study_%Y%m%d_%H%M%S"),
        help="Subdirectory name for this study run",
    )
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=Path("comparative_runs"),
        help="Base directory where study outputs are written",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="names.txt",
        help="Path to input file (relative paths resolved from project dir)",
    )
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-embd2", type=int, default=64)
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used to launch makemore.py",
    )
    parser.add_argument(
        "--examples-per-group",
        type=int,
        default=5,
        help="How many examples to show per sample group in the markdown report",
    )
    return parser.parse_args()


def resolve_input_file(input_file: str, project_dir: Path) -> Path:
    path = Path(input_file)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path.resolve()
    candidate = project_dir / input_file
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Input file '{input_file}' was not found as absolute, cwd-relative, or project-relative path."
    )


def run_command(cmd: List[str], cwd: Path, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("\n$", " ".join(cmd))
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}. See log: {log_file}"
        )


def parse_loss_records(log_file: Path) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOSS_PATTERN.search(line)
        if not match:
            continue
        step, train_loss, test_loss = match.groups()
        records.append(
            {
                "step": int(step),
                "train_loss": float(train_loss),
                "test_loss": float(test_loss),
            }
        )
    return records


def parse_samples(log_file: Path) -> Dict[str, List[str]]:
    mapping = {"in train": "in_train", "in test": "in_test", "new": "new"}
    samples: Dict[str, List[str]] = {"in_train": [], "in_test": [], "new": []}
    current_group: str | None = None

    for raw_line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        section_match = SECTION_PATTERN.match(line)
        if section_match:
            current_group = mapping[section_match.group(1)]
            continue
        if not line or line.startswith("-"):
            continue
        if current_group is not None:
            samples[current_group].append(line)

    return samples


def write_loss_csv(losses_by_model: Dict[str, List[Dict[str, float]]], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "step", "train_loss", "test_loss"])
        writer.writeheader()
        for model_name, records in losses_by_model.items():
            for record in records:
                writer.writerow({"model": model_name, **record})


def maybe_plot_losses(losses_by_model: Dict[str, List[Dict[str, float]]], out_file: Path) -> str:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return "matplotlib is not installed; skipped plot generation"

    plotted_any = False
    plt.figure(figsize=(11, 7))

    for model_name, records in losses_by_model.items():
        if not records:
            continue
        steps = [r["step"] for r in records]
        test_losses = [r["test_loss"] for r in records]
        plt.plot(steps, test_losses, marker="o", linewidth=2, label=model_name)
        plotted_any = True

    if not plotted_any:
        plt.close()
        return "no evaluation loss points found; skipped plot generation"

    plt.title("Makemore Model Comparison (Test Loss)")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close()
    return "ok"


def write_markdown_report(
    summary: Dict[str, Dict[str, object]], out_file: Path, examples_per_group: int
) -> None:
    lines: List[str] = ["# Makemore Comparative Study", ""]

    for model_name, info in summary.items():
        lines.append(f"## {model_name}")
        lines.append("")

        losses = info.get("loss_records", [])
        if losses:
            final = losses[-1]
            lines.append(
                f"- Final recorded losses at step {final['step']}: train={final['train_loss']:.4f}, test={final['test_loss']:.4f}"
            )
        else:
            lines.append("- No train/test loss snapshots were recorded.")

        samples = info.get("samples", {"in_train": [], "in_test": [], "new": []})
        lines.append("")
        lines.append("Sample outputs:")

        for group_key, group_title in (
            ("in_train", "Seen in train"),
            ("in_test", "Seen in test"),
            ("new", "New"),
        ):
            lines.append(f"- {group_title}:")
            examples = samples.get(group_key, [])[:examples_per_group]
            if examples:
                for ex in examples:
                    lines.append(f"  - {ex}")
            else:
                lines.append("  - (none)")

        lines.append("")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    input_file = resolve_input_file(args.input_file, project_dir)

    run_dir = (args.base_output_dir / args.study_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project dir: {project_dir}")
    print(f"Input file: {input_file}")
    print(f"Study output dir: {run_dir}")

    losses_by_model: Dict[str, List[Dict[str, float]]] = {}
    summary: Dict[str, Dict[str, object]] = {}

    for model_name in args.models:
        print("\n" + "=" * 80)
        print(f"Running model: {model_name}")
        model_dir = run_dir / model_name
        work_dir = model_dir / "work"
        train_log = model_dir / "train.log"
        sample_log = model_dir / "sample.log"

        train_cmd = [
            args.python_executable,
            "makemore.py",
            "--type",
            model_name,
            "--input-file",
            str(input_file),
            "--work-dir",
            str(work_dir),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--num-workers",
            str(args.num_workers),
            "--max-steps",
            str(args.max_steps),
            "--top-k",
            str(args.top_k),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--n-layer",
            str(args.n_layer),
            "--n-head",
            str(args.n_head),
            "--n-embd",
            str(args.n_embd),
            "--n-embd2",
            str(args.n_embd2),
        ]
        run_command(train_cmd, cwd=project_dir, log_file=train_log)

        sample_cmd = [
            args.python_executable,
            "makemore.py",
            "--type",
            model_name,
            "--input-file",
            str(input_file),
            "--work-dir",
            str(work_dir),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--num-workers",
            str(args.num_workers),
            "--top-k",
            str(args.top_k),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--n-layer",
            str(args.n_layer),
            "--n-head",
            str(args.n_head),
            "--n-embd",
            str(args.n_embd),
            "--n-embd2",
            str(args.n_embd2),
            "--sample-only",
        ]
        run_command(sample_cmd, cwd=project_dir, log_file=sample_log)

        loss_records = parse_loss_records(train_log)
        sample_groups = parse_samples(sample_log)

        losses_by_model[model_name] = loss_records
        summary[model_name] = {
            "model": model_name,
            "train_log": str(train_log),
            "sample_log": str(sample_log),
            "work_dir": str(work_dir),
            "loss_records": loss_records,
            "samples": sample_groups,
        }

    summary_file = run_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_file = run_dir / "loss_records.csv"
    write_loss_csv(losses_by_model, csv_file)

    plot_file = run_dir / "test_loss_comparison.png"
    plot_status = maybe_plot_losses(losses_by_model, plot_file)

    report_file = run_dir / "comparison_report.md"
    write_markdown_report(summary, report_file, args.examples_per_group)

    print("\n" + "=" * 80)
    print("Comparative study completed.")
    print(f"Summary JSON: {summary_file}")
    print(f"Loss CSV: {csv_file}")
    print(f"Plot: {plot_file} ({plot_status})")
    print(f"Report: {report_file}")


if __name__ == "__main__":
    main()