"""Deep Q-Learning agent plus the train / play loops."""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .game import BLOCK_SIZE, Direction, Point, SnakeGameAI
from .model import Linear_QNet, QTrainer, get_device

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


def get_state(game: SnakeGameAI) -> np.ndarray:
    """11 booleans: danger straight/right/left, direction (4), food left/right/up/down."""
    head = game.snake[0]
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger straight
        (dir_r and game.is_collision(point_r))
        or (dir_l and game.is_collision(point_l))
        or (dir_u and game.is_collision(point_u))
        or (dir_d and game.is_collision(point_d)),
        # Danger right
        (dir_u and game.is_collision(point_r))
        or (dir_d and game.is_collision(point_l))
        or (dir_l and game.is_collision(point_u))
        or (dir_r and game.is_collision(point_d)),
        # Danger left
        (dir_d and game.is_collision(point_r))
        or (dir_u and game.is_collision(point_l))
        or (dir_r and game.is_collision(point_u))
        or (dir_l and game.is_collision(point_d)),
        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,
        # Food location
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y,  # food down
    ]
    return np.array(state, dtype=int)


class Agent:
    def __init__(self, model: Linear_QNet | None = None, device: str = "auto"):
        self.device = get_device(device)
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate, smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = (model or Linear_QNet()).to(self.device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    get_state = staticmethod(get_state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample, strict=True)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, explore: bool = True) -> list[int]:
        # random moves early on: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if explore and random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            with torch.no_grad():
                prediction = self.model(torch.tensor(state, dtype=torch.float, device=self.device))
            move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    games: int | None = None,
    model_path: str | Path = "model/model.pth",
    render: bool = True,
    live_plot: bool = True,
    plot_path: str | Path | None = "model/training.png",
    speed: int = 0,
    seed: int | None = None,
    device: str = "auto",
) -> dict:
    """Train until ``games`` games were played (forever if ``None``).

    The best model so far is saved to ``model_path``. Returns a summary dict.
    """
    seed_everything(seed)
    plot = TrainingPlotOrNone(live_plot, plot_path)
    scores, mean_scores = [], []
    total_score = 0
    record = 0
    agent = Agent(device=device)
    print(f"Training on {agent.device}", flush=True)
    game = SnakeGameAI(render=render, speed=speed)

    try:
        while games is None or agent.n_games < games:
            state_old = get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            state_new = get_state(game)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save(model_path)

                total_score += score
                scores.append(score)
                mean_scores.append(total_score / agent.n_games)
                print(
                    f"Game {agent.n_games:4d}  Score {score:3d}  Record {record:3d}  "
                    f"Mean {mean_scores[-1]:.2f}",
                    flush=True,
                )
                plot.update(scores, mean_scores)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        plot.save()

    if record == 0:  # never scored: still leave a model behind
        agent.model.save(model_path)
    return {
        "games": agent.n_games,
        "record": record,
        "scores": scores,
        "model_path": str(model_path),
    }


class TrainingPlotOrNone:
    """Thin wrapper so the train loop needs no ``if plot`` checks."""

    def __init__(self, live: bool, path: str | Path | None):
        self.path = path
        self.plot = None
        if live or path:
            from .plot import TrainingPlot

            self.plot = TrainingPlot(live=live)

    def update(self, scores, mean_scores):
        if self.plot:
            self.plot.update(scores, mean_scores)

    def save(self):
        if self.plot and self.path:
            self.plot.save(self.path)
            print(f"Training curve saved to {self.path}")


def play(
    model_path: str | Path,
    games: int = 5,
    speed: int = 20,
    seed: int | None = None,
    device: str = "cpu",
) -> list[int]:
    """Watch a trained model play ``games`` games (no exploration, no learning)."""
    seed_everything(seed)
    agent = Agent(Linear_QNet.load(model_path), device=device)
    game = SnakeGameAI(render=True, speed=speed)
    scores = []
    while len(scores) < games:
        _, done, score = game.play_step(agent.get_action(get_state(game), explore=False))
        if done:
            scores.append(score)
            print(f"Game {len(scores)}  Score {score}", flush=True)
            game.reset()
    return scores
