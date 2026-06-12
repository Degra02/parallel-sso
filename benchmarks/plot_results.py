from collections import defaultdict
from pathlib import Path
import re

import matplotlib.pyplot as plt


RESULT_PATTERN = re.compile(
    r"(?:(?:procs|threads)\s*=\s*(?P<worker_a>\d+)\s+)?"
    r"(?:(?:procs|threads)\s*=\s*(?P<worker_b>\d+)\s+)?"
    r"Total(?:\s+OpenMP)?\s+time:\s*(?P<time>[0-9]+(?:\.[0-9]+)?)\s*s"
)

WEAK_HYBRID_PATTERN = re.compile(
    r"procs=(?P<procs>\d+)\s+"
    r"threads=(?P<threads>\d+)\s+"
    r"workers=(?P<workers>\d+)\s+"
    r"population=(?P<population>\d+)\s+"
    r"Total time:\s*(?P<time>[0-9]+(?:\.[0-9]+)?)\s*s"
)

WEAK_SCALING_VALUES = (1, 2, 4, 8, 16, 32, 64)
IDEAL_LINE_STYLE = {
    "color": "red",
    "linestyle": "--",
    "linewidth": 1.2,
}


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


def compute_hybrid_global_metrics(entries):
    baseline_time = None

    for entry in entries:
        if entry["procs"] == 1 and entry["threads"] == 1:
            baseline_time = entry["time"]
            break

    if baseline_time is None:
        raise ValueError("Hybrid baseline with 1 process and 1 thread not found.")

    metrics_by_procs = defaultdict(list)

    for entry in entries:
        procs = entry["procs"]
        threads = entry["threads"]

        if procs is None or threads is None:
            raise ValueError("Hybrid global metrics require both procs and threads.")

        workers = procs * threads
        speedup = baseline_time / entry["time"]
        efficiency = speedup / workers

        metrics_by_procs[procs].append(
            {
                "procs": procs,
                "threads": threads,
                "workers": workers,
                "time": entry["time"],
                "speedup": speedup,
                "efficiency": efficiency,
            }
        )

    for metrics in metrics_by_procs.values():
        metrics.sort(key=lambda row: row["workers"])

    return dict(sorted(metrics_by_procs.items()))


def compute_weak_scaling_metrics(results, baseline_time):
    metrics = []

    for parallelism, time_seconds in results:
        efficiency = baseline_time / time_seconds
        metrics.append(
            {
                "parallelism": parallelism,
                "time": time_seconds,
                "speedup": parallelism * efficiency,
                "efficiency": efficiency,
            }
        )

    return metrics


def add_ideal_line(kind, x_values, baseline_time=None):
    if kind is None:
        return []

    if kind in {"strong_speedup", "weak_speedup"}:
        label = (
            "Ideal linear speedup (S = N)"
            if kind == "strong_speedup"
            else "Ideal scaled speedup (S = N)"
        )
        plt.plot(x_values, x_values, label=label, **IDEAL_LINE_STYLE)
        return list(x_values)

    if kind in {"strong_efficiency", "weak_efficiency"}:
        label = (
            "Ideal efficiency (E = 1)"
            if kind == "strong_efficiency"
            else "Ideal weak efficiency (E = 1)"
        )
        plt.axhline(1.0, label=label, **IDEAL_LINE_STYLE)
        return [1.0]

    if kind == "weak_time":
        if baseline_time is None:
            raise ValueError("Weak-scaling time ideal requires a baseline time")
        plt.axhline(
            baseline_time,
            label="Ideal weak scaling (T = T1)",
            **IDEAL_LINE_STYLE,
        )
        return [baseline_time]

    raise ValueError(f"Unknown ideal line kind: {kind}")


def apply_y_limits(y_values, y_limits=None):
    if y_limits is None:
        return

    lower, upper = y_limits
    finite_values = [value for value in y_values if value is not None]

    if upper is not None and finite_values:
        upper = max(upper, max(finite_values) * 1.08)

    plt.ylim(lower, upper)


def validate_weak_scaling_values(results, label):
    values = tuple(parallelism for parallelism, _ in results)
    if values != WEAK_SCALING_VALUES:
        raise ValueError(
            f"{label} must contain exactly {WEAK_SCALING_VALUES}, found {values}"
        )


def parse_weak_hybrid_entries(raw_dir: Path):
    entries = []
    seen = set()

    for file_path in sorted(raw_dir.glob("weak_scaling_hybrid_sharks_*.txt")):
        for line in file_path.read_text(encoding="utf-8").splitlines():
            match = WEAK_HYBRID_PATTERN.fullmatch(line.strip())
            if match is None:
                continue

            entry = {
                "procs": int(match.group("procs")),
                "threads": int(match.group("threads")),
                "workers": int(match.group("workers")),
                "population": int(match.group("population")),
                "time": float(match.group("time")),
            }
            key = (entry["procs"], entry["threads"])

            if key in seen:
                raise ValueError(f"Duplicate hybrid weak-scaling result for {key}")
            if entry["workers"] != entry["procs"] * entry["threads"]:
                raise ValueError(f"Invalid worker count in {file_path}: {line!r}")
            if entry["population"] != 1000 * entry["workers"]:
                raise ValueError(f"Invalid population in {file_path}: {line!r}")

            seen.add(key)
            entries.append(entry)

    expected = {
        (procs, threads)
        for procs in WEAK_SCALING_VALUES
        for threads in WEAK_SCALING_VALUES
        if (procs, threads) != (64, 64)
    }
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)

    if missing:
        print(f"Warning: missing hybrid weak-scaling points: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected hybrid weak-scaling points: {unexpected}")
    if (1, 1) not in seen:
        raise ValueError("Hybrid weak scaling requires the (1 process, 1 thread) baseline")

    return sorted(entries, key=lambda row: (row["procs"], row["workers"]))


def plot_weak_scaling_series(
    series,
    metric_key,
    y_label,
    title,
    output_path,
    x_label,
    ideal=None,
    ideal_baseline_time=None,
    y_limits=None,
):
    plt.figure(figsize=(10, 5.5))
    all_x = sorted(
        {
            row["parallelism"]
            for _, metrics in series
            for row in metrics
        }
    )

    all_y = []

    weak_time_baselines = []

    for label, metrics in series:
        y_values = [row[metric_key] for row in metrics]
        all_y.extend(y_values)
        line, = plt.plot(
            [row["parallelism"] for row in metrics],
            y_values,
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=label,
        )

        if ideal == "weak_time_by_series":
            weak_time_baselines.append((label, y_values[0], line.get_color()))

    if ideal == "weak_time_by_series":
        for label, baseline_time, color in weak_time_baselines:
            plt.axhline(
                baseline_time,
                color=color,
                linestyle="--",
                linewidth=1.2,
                label=f"Ideal {label} (T = T1)",
            )
            all_y.append(baseline_time)
    else:
        all_y.extend(add_ideal_line(ideal, all_x, ideal_baseline_time))

    plt.xscale("log", base=2)
    plt.xticks(all_x, [str(value) for value in all_x])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    apply_y_limits(all_y, y_limits)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_weak_scaling_sharks_plots(raw_dir: Path, plots_dir: Path):
    weak_dir = raw_dir / "weak_scaling_sharks"
    hybrid_dir = raw_dir / "weak_scaling_hybrid_sharks"
    output_dir = plots_dir / "weak_scaling_sharks"
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_plot in output_dir.glob("*.png"):
        old_plot.unlink()

    single_variants = (
        (
            "openmp",
            weak_dir / "weak_scaling_openmp_sharks.txt",
            "threads",
            "OpenMP",
        ),
        (
            "mpi",
            weak_dir / "weak_scaling_mpi_sharks.txt",
            "procs",
            "MPI",
        ),
    )
    single_metrics = {}

    for slug, file_path, axis, label in single_variants:
        entries = parse_entries(file_path)
        results = group_by_axis(entries, axis)
        validate_weak_scaling_values(results, label)
        metrics = compute_weak_scaling_metrics(results, results[0][1])
        single_metrics[slug] = metrics
    openmp_mpi_series = [
        ("OpenMP threads", single_metrics["openmp"]),
        ("MPI processes", single_metrics["mpi"]),
    ]
    openmp_mpi_x_label = "Number of processes/threads"

    plot_weak_scaling_series(
        openmp_mpi_series,
        "time",
        "Execution time (seconds)",
        "Weak scaling - OpenMP vs MPI sharks: Execution time",
        output_dir / "weak_scaling_openmp_mpi_sharks_time.png",
        openmp_mpi_x_label,
        ideal="weak_time_by_series",
    )
    plot_weak_scaling_series(
        openmp_mpi_series,
        "speedup",
        "Scaled speedup",
        "Weak scaling - OpenMP vs MPI sharks: Scaled speedup",
        output_dir / "weak_scaling_openmp_mpi_sharks_speedup.png",
        openmp_mpi_x_label,
        ideal="weak_speedup",
    )
    plot_weak_scaling_series(
        openmp_mpi_series,
        "efficiency",
        "Weak-scaling efficiency",
        "Weak scaling - OpenMP vs MPI sharks: Efficiency",
        output_dir / "weak_scaling_openmp_mpi_sharks_efficiency.png",
        openmp_mpi_x_label,
        ideal="weak_efficiency",
        y_limits=(0, 1.05),
    )

    hybrid_entries = parse_weak_hybrid_entries(hybrid_dir)
    baseline_time = next(
        row["time"]
        for row in hybrid_entries
        if row["procs"] == 1 and row["threads"] == 1
    )
    hybrid_series = []

    for procs in WEAK_SCALING_VALUES:
        rows = [row for row in hybrid_entries if row["procs"] == procs]
        results = [(row["workers"], row["time"]) for row in rows]
        metrics = compute_weak_scaling_metrics(results, baseline_time)
        process_label = "process" if procs == 1 else "processes"
        hybrid_series.append((f"{procs} {process_label}", metrics))

    hybrid_specs = (
        ("time", "Execution time (seconds)", "time", "weak_time", None),
        ("speedup", "Scaled speedup", "speedup", "weak_speedup", None),
        ("efficiency", "Weak-scaling efficiency", "efficiency", "weak_efficiency", (0, 1.05)),
    )

    for metric_key, y_label, suffix, ideal, y_limits in hybrid_specs:
        plot_weak_scaling_series(
            hybrid_series,
            metric_key,
            y_label,
            f"Weak scaling - Hybrid sharks: {y_label}",
            output_dir / f"weak_scaling_hybrid_sharks_{suffix}.png",
            "Total processes x threads",
            ideal=ideal,
            ideal_baseline_time=baseline_time,
            y_limits=y_limits,
        )


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
    name_map = {
        "mpi_dim.txt": "MPI (dimensions)",
        "openmp_dim.txt": "OpenMP (dimensions)",
        "mpi_sharks.txt": "MPI (sharks)",
        "openmp_sharks.txt": "OpenMP (sharks)",
        "mpi_rot.txt": "MPI (rotations)",
        "openmp_rot.txt": "OpenMP (rotations)",
        "mpi_rot_huge.txt": "MPI (rotations, huge)",
        "openmp_rot_huge.txt": "OpenMP (rotations, huge)",
        "mpi_sharks_huge.txt": "MPI (sharks, huge)",
        "openmp_sharks_huge.txt": "OpenMP (sharks, huge)",
    }

    if file_path.name in name_map:
        return name_map[file_path.name]
    return file_path.stem


def family_for_file(file_path: Path):
    if file_path.name in {"mpi_dim.txt", "openmp_dim.txt"}:
        return "dim"
    if file_path.name in {"mpi_sharks.txt", "openmp_sharks.txt"}:
        return "sharks"
    if file_path.name in {"mpi_rot.txt", "openmp_rot.txt"}:
        return "rot"
    if file_path.name in {"mpi_rot_huge.txt", "openmp_rot_huge.txt", "mpi_sharks_huge.txt", "openmp_sharks_huge.txt"}:
        return "rot_huge"
    return None


def x_label_for_axis(axis: str):
    return "Number of threads" if axis == "threads" else "Number of processes"


def x_label_for_family(family_name: str, series_data):
    axes = {axis for _, axis, _ in series_data}

    if axes == {"threads"}:
        return "Number of threads"
    if axes == {"procs"}:
        return "Number of processes"
    return "Number of processes/threads"


def generate_family_plots(family_name: str, series_data, output_dir: Path):
    title_map = {
        "mpi_dim": "MPI dimensions",
        "dim": "Dimensions",
        "sharks": "Sharks",
        "rot": "Rotations",
        "rot_huge": "Rotations (huge)",
    }

    for metric_key, y_label, suffix in (
        ("time", "Execution time (seconds)", "time"),
        ("speedup", "Speedup", "speedup"),
        ("efficiency", "Efficiency", "efficiency"),
    ):
        plt.figure(figsize=(10, 5.5))

        x_label = x_label_for_family(family_name, series_data)
        all_y = []

        for label, axis, metrics in series_data:
            x_values = [row["workers"] for row in metrics]
            y_values = [row[metric_key] for row in metrics]
            all_y.extend(y_values)
            plt.plot(x_values, y_values, marker="o", linewidth=1.5, markersize=4, label=label)

        all_x = sorted({row["workers"] for _, _, metrics in series_data for row in metrics})
        if metric_key == "speedup":
            all_y.extend(add_ideal_line("strong_speedup", all_x))
        elif metric_key == "efficiency":
            all_y.extend(add_ideal_line("strong_efficiency", all_x))

        plt.xscale("log", base=2)
        plt.xticks(all_x, [str(v) for v in all_x])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"{title_map.get(family_name, family_name.replace('_', ' ').title())}: {y_label}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if metric_key == "efficiency":
            apply_y_limits(all_y, (0, 1.05))

        plt.tight_layout()
        plt.savefig(output_dir / f"{family_name.lower()}_{suffix}.png", dpi=300)
        plt.close()


def generate_hybrid_global_plots(raw_dir: Path, output_dir: Path):
    hybrid_dir = raw_dir / "hybrid_sharks"

    if not hybrid_dir.exists():
        return

    entries = []
    for file_path in sorted(hybrid_dir.glob("*.txt")):
        entries.extend(parse_entries(file_path))

    if not entries:
        return

    metrics_by_procs = compute_hybrid_global_metrics(entries)

    plot_specs = (
        (
            "time",
            "Execution time (seconds)",
            "Hybrid sharks global: Execution time",
            "time",
            None,
        ),
        (
            "speedup",
            "Speedup (T(1,1) / T(P,T))",
            "Hybrid sharks global: Speedup",
            "speedup",
            None,
        ),
        (
            "efficiency",
            "Efficiency (T(1,1) / (P*T*T(P,T)))",
            "Hybrid sharks global: Efficiency",
            "efficiency",
            (0, 1.05),
        ),
    )

    for metric_key, y_label, title, suffix, y_limits in plot_specs:
        plt.figure(figsize=(10, 5.5))
        all_workers = sorted({row["workers"] for metrics in metrics_by_procs.values() for row in metrics})

        all_y = []

        for procs, metrics in metrics_by_procs.items():
            x_values = [row["workers"] for row in metrics]
            y_values = [row[metric_key] for row in metrics]
            all_y.extend(y_values)
            plt.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.5,
                markersize=4,
                label=f"Hybrid Sharks ({procs} procs)",
            )

        if metric_key == "speedup":
            all_y.extend(add_ideal_line("strong_speedup", all_workers))
        elif metric_key == "efficiency":
            all_y.extend(add_ideal_line("strong_efficiency", all_workers))

        plt.xscale("log", base=2)
        plt.xticks(all_workers, [str(worker) for worker in all_workers])
        plt.xlabel("Number of workers (processes * threads, log2 scale)")
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()

        apply_y_limits(all_y, y_limits)

        plt.tight_layout()
        plt.savefig(output_dir / f"hybrid_sharks_global_{suffix}.png", dpi=300)
        plt.close()


def main():
    script_dir = Path(__file__).resolve().parent
    raw_dir = script_dir / "raw"
    output_dir = script_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    generate_hybrid_global_plots(raw_dir, output_dir)
    generate_weak_scaling_sharks_plots(raw_dir, output_dir)


if __name__ == "__main__":
    main()
