from pathlib import Path
import re
import matplotlib.pyplot as plt


def parse_results(file_path: Path, worker_key: str):
    text = file_path.read_text(encoding="utf-8")

    pattern = rf"{worker_key}\s*=\s*(\d+)\s+Total(?:\s+OpenMP)?\s+time:\s*([0-9]+(?:\.[0-9]+)?)\s*s"

    results = []

    for match in re.finditer(pattern, text):
        workers = int(match.group(1))
        time_seconds = float(match.group(2))
        results.append((workers, time_seconds))

    if not results:
        raise ValueError(f"No results found in {file_path}")

    results.sort(key=lambda x: x[0])
    return results


def compute_metrics(results):
    baseline_time = None

    for workers, time_seconds in results:
        if workers == 1:
            baseline_time = time_seconds
            break

    if baseline_time is None:
        raise ValueError("Baseline with 1 worker not found.")

    metrics = []

    for workers, time_seconds in results:
        speedup = baseline_time / time_seconds
        efficiency = speedup / workers

        metrics.append({
            "workers": workers,
            "time": time_seconds,
            "speedup": speedup,
            "efficiency": efficiency,
        })

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


def generate_plots(input_file, worker_key, x_label, label_prefix, output_dir):
    results = parse_results(input_file, worker_key)
    metrics = compute_metrics(results)

    plot_metric(
        metrics,
        x_label,
        "time",
        "Execution time (seconds)",
        f"{label_prefix}: Execution Time",
        output_dir / f"{label_prefix.lower()}_time.png",
    )

    plot_metric(
        metrics,
        x_label,
        "speedup",
        "Speedup",
        f"{label_prefix}: Speedup",
        output_dir / f"{label_prefix.lower()}_speedup.png",
    )

    plot_metric(
        metrics,
        x_label,
        "efficiency",
        "Efficiency",
        f"{label_prefix}: Efficiency",
        output_dir / f"{label_prefix.lower()}_efficiency.png",
        y_limits=(0, 1),
    )


def main():
    script_dir = Path(__file__).resolve().parent

    raw_dir = script_dir / "raw"
    output_dir = script_dir / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)

    generate_plots(
        raw_dir / "openmp_sharks.txt",
        "threads",
        "Number of threads",
        "OpenMP",
        output_dir,
    )

    generate_plots(
        raw_dir / "mpi_sharks.txt",
        "procs",
        "Number of processes",
        "MPI",
        output_dir,
    )


if __name__ == "__main__":
    main()