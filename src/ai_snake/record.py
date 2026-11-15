"""Record a trained agent playing into an animated GIF (no window needed)."""

from __future__ import annotations

from pathlib import Path

from .agent import Agent, get_state, seed_everything
from .game import SnakeGameAI
from .model import Linear_QNet


def record_gif(
    model_path: str | Path,
    output: str | Path = "assets/demo.gif",
    max_frames: int = 600,
    fps: int = 25,
    scale: float = 0.75,
    min_score: int = 0,
    seed: int | None = 0,
    max_attempts: int = 50,
) -> dict:
    """Play headless games and save the first one scoring ``>= min_score`` as a GIF.

    The recording stops after ``max_frames`` frames even if the game is not over.
    """
    from PIL import Image

    seed_everything(seed)
    agent = Agent(Linear_QNet.load(model_path), device="cpu")
    game = SnakeGameAI(render=True, headless=True)

    best = None
    for _ in range(max_attempts):
        game.reset()
        frames = [game.frame()]
        done = False
        while not done and len(frames) < max_frames:
            _, done, score = game.play_step(agent.get_action(get_state(game), explore=False))
            if not done:
                frames.append(game.frame())
        if best is None or game.score > best[0]:
            best = (game.score, frames)
        if game.score >= min_score:
            break

    score, frames = best
    # hold the last frame a bit so the final score is readable
    frames += [frames[-1]] * fps

    w, h = int(game.w * scale), int(game.h * scale)
    images = [Image.fromarray(f).resize((w, h), Image.NEAREST) for f in frames]
    palette = images[0].quantize(colors=16, method=Image.Quantize.MEDIANCUT)
    images = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    return {"output": str(output), "score": score, "frames": len(frames)}
