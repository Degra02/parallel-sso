from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt


SERIES_LABELS = {
    "serial_population.txt": "Population",
    "serial_dims.txt": "Dimensions",
    "serial_stages.txt": "Stages",
    "serial_rot.txt": "Rotations",
}


def parse_times(file_path: Path) -> list[float]:
    text = file_path.read_text(encoding="utf-8")
    times = [float(match.group(1)) for match in re.finditer(r"Total time:\s*([0-9]+(?:\.[0-9]+)?)\s*s", text)]
    if not times:
        raise ValueError(f"No timing data found in {file_path}")
    return times


def compute_slowdown(times: list[float]) -> list[float]:
    baseline = times[0]
    if baseline <= 0:
        raise ValueError("Baseline time must be positive")
    return [time_seconds / baseline for time_seconds in times]


def load_series(raw_dir: Path) -> list[tuple[str, list[int], list[float]]]:
    series = []

    for file_path in sorted(raw_dir.glob("serial_*.txt")):
        if file_path.name == "serial_v2.txt":
            continue

        times = parse_times(file_path)
        slowdown = compute_slowdown(times)
        x_values = list(range(1, len(slowdown) + 1))
        label = SERIES_LABELS.get(file_path.name, file_path.stem.replace("serial_", "").replace("_", " ").title())
        series.append((label, x_values, slowdown))

    if not series:
        raise ValueError(f"No serial raw series found in {raw_dir}")

    return series


def plot_serial_slowdown(series: list[tuple[str, list[int], list[float]]], output_path: Path) -> None:
    plt.figure(figsize=(10, 5.5))

    for label, x_values, slowdown in series:
        plt.plot(x_values, slowdown, marker="o", linewidth=1.8, markersize=4, label=label)

    plt.xlabel("Multiplier index")
    plt.ylabel("Slowdown (time / baseline time)")
    plt.title("Serial SSO slowdown across parameter sweeps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    raw_dir = script_dir / "raw" / "serial"
    output_dir = script_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    series = load_series(raw_dir)
    plot_serial_slowdown(series, output_dir / "serial_raw_slowdown.png")


if __name__ == "__main__":
    main()