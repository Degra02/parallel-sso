from collections import defaultdict
from pathlib import Path
import re

import matplotlib.pyplot as plt


RESULT_PATTERN = re.compile(
    r"(?:(?:procs|threads)\s*=\s*(?P<worker_a>\d+)\s+)?"
    r"(?:(?:procs|threads)\s*=\s*(?P<worker_b>\d+)\s+)?"
    r"Total(?:\s+OpenMP)?\s+time:\s*(?P<time>[0-9]+(?:\.[0-9]+)?)\s*s"
)


def parse_entries(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    entries = []

    for line in text.splitlines():
        match = RESULT_PATTERN.search(line)
        if not match:
            continue

        values = [value for value in (match.group("worker_a"), match.group("worker_b")) if value is not None]
        time_seconds = float(match.group("time"))

        if not values:
            raise ValueError(f"Could not infer worker axis in {file_path}: {line!r}")

        entry = {
            "time": time_seconds,
            "procs": None,
            "threads": None,
        }

        if len(values) == 1:
            if "threads=" in line:
                entry["threads"] = int(values[0])
            else:
                entry["procs"] = int(values[0])
        else:
            entry["procs"] = int(values[0])
            entry["threads"] = int(values[1])

        entries.append(entry)

    if not entries:
        raise ValueError(f"No results found in {file_path}")

    return entries


def infer_axis(entries):
    has_threads = any(entry["threads"] is not None for entry in entries)
    has_procs = any(entry["procs"] is not None for entry in entries)

    if has_threads and has_procs:
        return "threads"
    if has_threads:
        return "threads"
    if has_procs:
        return "procs"
    raise ValueError("Could not infer axis")


def group_by_axis(entries, axis):
    grouped = defaultdict(list)

    for entry in entries:
        axis_value = entry.get(axis)
        if axis_value is None:
            continue
        grouped[axis_value].append(entry["time"])

    if not grouped:
        raise ValueError(f"No usable data for axis {axis}")

    results = []
    for axis_value, times in grouped.items():
        avg_time = sum(times) / len(times)
        results.append((axis_value, avg_time))

    results.sort(key=lambda item: item[0])
    return results


def compute_metrics(results):
    baseline_time = None

    for axis_value, time_seconds in results:
        if axis_value == 1:
            baseline_time = time_seconds
            break

    if baseline_time is None:
        raise ValueError("Baseline with 1 worker not found.")

    metrics = []

    for axis_value, time_seconds in results:
        speedup = baseline_time / time_seconds
        efficiency = speedup / axis_value

        metrics.append(
            {
                "workers": axis_value,
                "time": time_seconds,
                "speedup": speedup,
                "efficiency": efficiency,
            }
        )

    return metrics


def plot_metric(metrics, x_label, y_key, y_label, title, output_path, y_limits=None):
    workers = [row["workers"] for row in metrics]
    values = [row[y_key] for row in metrics]

    plt.figure(figsize=(9, 5))
    plt.plot(workers, values, marker="o", linewidth=1.5, markersize=4)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if y_limits is not None:
        plt.ylim(y_limits)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def label_for_file(file_path: Path):
    if file_path.name == "mpi_dim.txt":
        return "MPI (dimensions)"
    if file_path.name == "mpi_sharks.txt":
        return "MPI (sharks)"
    if file_path.name == "openmp_sharks.txt":
        return "OpenMP (sharks)"
    if file_path.parent.name in {"hybrid_sharks", "hybrid_hybrid"}:
        return f"{file_path.parent.name.replace('_', ' ').title()} ({file_path.stem.split('_')[-1]} procs)"
    return file_path.stem


def family_for_file(file_path: Path):
    if file_path.name in {"mpi_dim.txt", "openmp_dim.txt"}:
        return "dim"
    if file_path.name in {"mpi_sharks.txt", "openmp_sharks.txt"}:
        return "sharks"
    if file_path.parent.name in {"hybrid_sharks", "hybrid_hybrid"}:
        return file_path.parent.name
    return None


def x_label_for_axis(axis: str):
    return "Number of threads" if axis == "threads" else "Number of processes"


def generate_family_plots(family_name: str, series_data, output_dir: Path):
    title_map = {
        "mpi_dim": "MPI dimensions",
        "sharks": "Sharks",
        "hybrid_sharks": "Hybrid sharks",
        "hybrid_hybrid": "Hybrid hybrid",
    }

    for metric_key, y_label, suffix in (
        ("time", "Execution time (seconds)", "time"),
        ("speedup", "Speedup", "speedup"),
        ("efficiency", "Efficiency", "efficiency"),
    ):
        plt.figure(figsize=(10, 5.5))

        x_label = None
        for label, axis, metrics in series_data:
            x_label = x_label or x_label_for_axis(axis)
            x_values = [row["workers"] for row in metrics]
            y_values = [row[metric_key] for row in metrics]
            plt.plot(x_values, y_values, marker="o", linewidth=1.5, markersize=4, label=label)

        plt.xlabel(x_label or "Workers")
        plt.ylabel(y_label)
        plt.title(f"{title_map.get(family_name, family_name.replace('_', ' ').title())}: {y_label}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if metric_key == "efficiency":
            plt.ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(output_dir / f"{family_name.lower()}_{suffix}.png", dpi=300)
        plt.close()


def cleanup_legacy_outputs(output_dir: Path):
    # Remove plots produced by the previous, narrower version of this script.
    for legacy_name in (
        "mpi_time.png",
        "mpi_speedup.png",
        "mpi_efficiency.png",
        "openmp_time.png",
        "openmp_speedup.png",
        "openmp_efficiency.png",
        "mpi_sharks_time.png",
        "mpi_sharks_speedup.png",
        "mpi_sharks_efficiency.png",
        "openmp_sharks_time.png",
        "openmp_sharks_speedup.png",
        "openmp_sharks_efficiency.png",
    ):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()


def main():
    script_dir = Path(__file__).resolve().parent
    raw_dir = script_dir / "raw"
    output_dir = script_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_outputs(output_dir)

    family_series = defaultdict(list)

    for file_path in sorted(raw_dir.rglob("*.txt")):
        if "serial" in file_path.parts or file_path.name.startswith("serial"):
            continue

        family = family_for_file(file_path)
        if family is None:
            continue

        entries = parse_entries(file_path)
        axis = infer_axis(entries)
        results = group_by_axis(entries, axis)
        metrics = compute_metrics(results)
        label = label_for_file(file_path)
        family_series[family].append((label, axis, metrics))

    if not family_series:
        raise ValueError(f"No non-serial benchmark files found under {raw_dir}")

    for family_name, series_data in family_series.items():
        generate_family_plots(family_name, series_data, output_dir)


if __name__ == "__main__":
    main()