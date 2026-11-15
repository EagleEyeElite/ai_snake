"""Snake environment that an agent controls with relative actions.

An action is a one-hot list ``[straight, right, left]``.
"""

from __future__ import annotations

import os
import random
from collections import namedtuple
from enum import Enum

import numpy as np


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40  # frames per second when a window is shown

CLOCK_WISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]


class SnakeGameAI:
    """Snake game.

    Args:
        w, h: board size in pixels (multiples of ``BLOCK_SIZE``).
        render: draw every frame. Needed for a window or for ``frame()``.
        headless: draw off-screen without opening a window (SDL dummy driver).
        speed: frames per second for a visible window; ``0`` means unlimited.
    """

    def __init__(
        self,
        w: int = 640,
        h: int = 480,
        render: bool = True,
        headless: bool = False,
        speed: int = SPEED,
    ):
        self.w = w
        self.h = h
        self.render = render
        self.headless = headless
        self.speed = speed
        self.display = None
        self.clock = None
        self.font = None

        if render:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            if headless:
                os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            import pygame

            pygame.init()
            self._pg = pygame
            self.font = pygame.font.Font(None, 30)  # pygame's bundled font
            if headless:
                self.display = pygame.Surface((self.w, self.h))
            else:
                self.display = pygame.display.set_mode((self.w, self.h))
                pygame.display.set_caption("AI Snake")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self) -> None:
        self.direction = Direction.RIGHT

        self.head = Point(
            self.w // 2 // BLOCK_SIZE * BLOCK_SIZE, self.h // 2 // BLOCK_SIZE * BLOCK_SIZE
        )
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        if self.render:
            self._update_ui()

    def _place_food(self) -> None:
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                return

    def play_step(self, action) -> tuple[int, bool, int]:
        """Advance one frame. Returns ``(reward, game_over, score)``."""
        self.frame_iteration += 1

        # 1. handle window events (closing the window stops the program)
        if self.render and not self.headless:
            for event in self._pg.event.get():
                if event.type == self._pg.QUIT:
                    self._pg.quit()
                    raise SystemExit(0)

        # 2. move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. game over: collision, or wandering around without eating
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            return -10, True, self.score

        # 4. eat food or just move
        reward = 0
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        if self.render:
            self._update_ui()
            if not self.headless and self.speed:
                self.clock.tick(self.speed)
        return reward, False, self.score

    def is_collision(self, pt: Point | None = None) -> bool:
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        return pt in self.snake[1:]

    def _update_ui(self) -> None:
        pg = self._pg
        self.display.fill(BLACK)

        for pt in self.snake:
            pg.draw.rect(self.display, BLUE1, pg.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pg.draw.rect(self.display, BLUE2, pg.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pg.draw.rect(self.display, RED, pg.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [6, 6])
        if not self.headless:
            pg.display.flip()

    def frame(self) -> np.ndarray:
        """Current frame as an ``(h, w, 3)`` uint8 RGB array."""
        if not self.render:
            raise RuntimeError("frame() needs render=True")
        return np.transpose(self._pg.surfarray.array3d(self.display), (1, 0, 2))

    def _move(self, action) -> None:
        # action = [straight, right, left]
        idx = CLOCK_WISE.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = CLOCK_WISE[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = CLOCK_WISE[(idx + 1) % 4]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            new_dir = CLOCK_WISE[(idx - 1) % 4]  # left turn r -> u -> l -> d
        self.direction = new_dir

        x, y = self.head
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)
