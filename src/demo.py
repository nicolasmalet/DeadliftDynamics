"""Run the deterministic, headless reference experiment."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brain import make_decision
from config import t
from state import create_state
from update import update_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("assets/trajectory.png"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    state = create_state()
    bar_y = [float(state.bones[-1].end[1])]
    com_x = [float(state.l_gravity_center[-1][0])]
    for _ in range(args.steps):
        update_model(state, make_decision(state))
        bar_y.append(float(state.bones[-1].end[1]))
        com_x.append(float(state.l_gravity_center[-1][0]))

    if not all(np.isfinite(series).all() for series in (bar_y, com_x, state.efforts)) or not np.all(
        (state.efforts >= 0) & (state.efforts <= 1)
    ):
        raise RuntimeError("simulation produced an invalid state")

    time = np.arange(args.steps + 1) * t
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(time, bar_y, color="#d6df43", linewidth=2)
    axes[0].set(title="Bar height", xlabel="Simulated time (s)", ylabel="Height (m)")
    axes[1].plot(time, com_x, color="#ff7657", linewidth=2, label="model")
    axes[1].axhline(0.14, color="#8c95a3", linestyle="--", linewidth=1, label="controller target")
    axes[1].set(title="Horizontal centre of mass", xlabel="Simulated time (s)", ylabel="Position (m)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    print(f"Simulated time: {args.steps * t:.3f} s ({args.steps} steps, dt={t:g} s)")
    print(f"Bar height: {bar_y[0]:.3f} -> {bar_y[-1]:.3f} m ({bar_y[-1] - bar_y[0]:+.3f} m)")
    print(f"Maximum |COM x - 0.14|: {max(abs(x - 0.14) for x in com_x):.4f} m")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
