"""Command line: ``ai-snake train | play | record``."""

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path

DEFAULT_MODEL = "model/model.pth"


def pretrained_model() -> Path:
    """Path of the model that ships with the package."""
    return Path(str(files("ai_snake") / "pretrained" / "model.pth"))


def resolve_model(path: str | None) -> Path:
    if path:
        return Path(path)
    if Path(DEFAULT_MODEL).exists():
        return Path(DEFAULT_MODEL)
    return pretrained_model()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ai-snake", description="Snake that learns to play itself with Deep Q-Learning."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a new agent")
    p.add_argument("--games", type=int, default=None, help="stop after N games (default: never)")
    p.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"where to save (default: {DEFAULT_MODEL})"
    )
    p.add_argument("--headless", action="store_true", help="no game window and no live plot (fast)")
    p.add_argument("--no-plot", action="store_true", help="no live training plot")
    p.add_argument("--plot-file", default="model/training.png", help="save the training curve here")
    p.add_argument("--speed", type=int, default=0, help="window FPS, 0 = as fast as possible")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, mps or cpu")

    p = sub.add_parser("play", help="watch a trained agent play")
    p.add_argument(
        "--model", default=None, help="model file (default: your model, else pretrained)"
    )
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--speed", type=int, default=20, help="frames per second")
    p.add_argument("--seed", type=int, default=None)

    p = sub.add_parser("record", help="record a trained agent into a GIF (no window needed)")
    p.add_argument(
        "--model", default=None, help="model file (default: your model, else pretrained)"
    )
    p.add_argument("--output", default="assets/demo.gif")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--scale", type=float, default=0.75)
    p.add_argument("--min-score", type=int, default=0, help="retry until a game scores this much")
    p.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)

    if args.command == "train":
        from .agent import train

        result = train(
            games=args.games,
            model_path=args.model,
            render=not args.headless,
            live_plot=not (args.headless or args.no_plot),
            plot_path=args.plot_file or None,
            speed=args.speed,
            seed=args.seed,
            device=args.device,
        )
        print(f"Played {result['games']} games, record {result['record']}, model: {args.model}")
    elif args.command == "play":
        from .agent import play

        model = resolve_model(args.model)
        print(f"Using model {model}")
        play(model, games=args.games, speed=args.speed, seed=args.seed)
    elif args.command == "record":
        from .record import record_gif

        model = resolve_model(args.model)
        print(f"Using model {model}")
        result = record_gif(
            model,
            output=args.output,
            max_frames=args.max_frames,
            fps=args.fps,
            scale=args.scale,
            min_score=args.min_score,
            seed=args.seed,
        )
        print(f"Saved {result['output']} ({result['frames']} frames, score {result['score']})")


if __name__ == "__main__":
    main()
