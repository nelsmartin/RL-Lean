import math
import matplotlib.pyplot as plt


def _rolling_avg(values, window=10):
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(sum(values[start:i + 1]) / (i - start + 1))
    return smoothed


def plot_training_curves(solve_rates, avg_steps, avg_returns, losses,
                         window=10, save_path="training_curves.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Solve rate
    axes[0, 0].plot(solve_rates, alpha=0.3, label="per epoch")
    axes[0, 0].plot(_rolling_avg(solve_rates, window), label=f"rolling avg (w={window})")
    axes[0, 0].set(xlabel="Epoch", ylabel="Solve rate", title="Solve rate per epoch")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].legend()

    # Avg steps to solve
    valid = [(i, s) for i, s in enumerate(avg_steps) if not math.isnan(s)]
    if valid:
        xs, ys = zip(*valid)
        axes[0, 1].scatter(xs, ys, s=10, alpha=0.3, label="per epoch")
        axes[0, 1].plot(xs, _rolling_avg(ys, window), label=f"rolling avg (w={window})")
        axes[0, 1].legend()
    axes[0, 1].set(xlabel="Epoch", ylabel="Avg steps to solve",
                   title="Steps to solve (solved episodes only)")

    # Avg return
    axes[1, 0].plot(avg_returns, alpha=0.3, label="per epoch")
    axes[1, 0].plot(_rolling_avg(avg_returns, window), label=f"rolling avg (w={window})")
    axes[1, 0].set(xlabel="Epoch", ylabel="Avg return", title="Average episode return")
    axes[1, 0].legend()

    # Loss
    axes[1, 1].plot(losses, alpha=0.3, label="per epoch")
    axes[1, 1].plot(_rolling_avg(losses, window), label=f"rolling avg (w={window})")
    axes[1, 1].set(xlabel="Epoch", ylabel="Loss", title="Policy loss")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
