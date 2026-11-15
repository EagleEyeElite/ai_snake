"""Training curve: live window while training, PNG at the end."""

from __future__ import annotations

from pathlib import Path


class TrainingPlot:
    def __init__(self, live: bool = True):
        import matplotlib

        if not live:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.live = live
        if live:
            plt.ion()
        self.fig = plt.figure("AI Snake - training", figsize=(6, 4))

    def update(self, scores: list[int], mean_scores: list[float]) -> None:
        plt = self.plt
        self.fig.clf()
        ax = self.fig.add_subplot()
        ax.set_title("Training")
        ax.set_xlabel("Number of games")
        ax.set_ylabel("Score")
        ax.plot(scores, label="score")
        ax.plot(mean_scores, label="mean score")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left")
        if scores:
            ax.text(len(scores) - 1, scores[-1], str(scores[-1]))
            ax.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]:.1f}")
        if self.live:
            plt.pause(0.001)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=100, bbox_inches="tight")
