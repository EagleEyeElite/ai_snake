import numpy as np
import pytest
import torch

from ai_snake.agent import Agent, get_state, play, train
from ai_snake.cli import main, pretrained_model
from ai_snake.game import BLOCK_SIZE, Point, SnakeGameAI
from ai_snake.model import Linear_QNet, QTrainer
from ai_snake.record import record_gif


def test_state_has_11_features():
    game = SnakeGameAI(render=False)
    state = get_state(game)
    assert state.shape == (11,)
    assert set(np.unique(state)) <= {0, 1}


def test_wall_collision_ends_game():
    game = SnakeGameAI(render=False)
    game.head = Point(game.w - BLOCK_SIZE, game.head.y)
    game.snake[0] = game.head
    _, done, _ = game.play_step([1, 0, 0])  # straight into the right wall
    assert done


def test_batch_update_uses_each_samples_action():
    torch.manual_seed(0)
    model = Linear_QNet()
    trainer = QTrainer(model, lr=0.0, gamma=0.9)  # lr=0: only inspect the target
    states = np.zeros((2, 11))
    actions = [[1, 0, 0], [0, 0, 1]]
    with torch.no_grad():
        before = model(torch.tensor(states, dtype=torch.float))
    captured = {}
    trainer.criterion = lambda pred, target: (
        captured.setdefault("t", target).sum() * 0 + pred.sum() * 0
    )
    trainer.train_step(states, actions, [10.0, -10.0], states, (True, True))
    target = captured["t"]
    assert target[0, 0] == 10.0 and target[1, 2] == -10.0
    assert torch.allclose(target[0, 1:], before[0, 1:])
    assert torch.allclose(target[1, :2], before[1, :2])


def test_pretrained_model_scores():
    agent = Agent(Linear_QNet.load(pretrained_model()), device="cpu")
    game = SnakeGameAI(render=False)
    done = False
    while not done:
        _, done, score = game.play_step(agent.get_action(get_state(game), explore=False))
    assert score >= 5


def test_short_training_run(tmp_path):
    result = train(
        games=3,
        model_path=tmp_path / "m.pth",
        render=False,
        live_plot=False,
        plot_path=tmp_path / "t.png",
        seed=0,
        device="cpu",
    )
    assert result["games"] == 3
    assert (tmp_path / "m.pth").exists() and (tmp_path / "t.png").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_training_on_gpu(tmp_path):
    result = train(
        games=2,
        model_path=tmp_path / "m.pth",
        render=False,
        live_plot=False,
        plot_path=None,
        seed=0,
        device="cuda",
    )
    assert result["games"] == 2
    Linear_QNet.load(tmp_path / "m.pth")  # GPU-trained weights load on CPU


def test_play_and_record_headless(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    assert len(play(pretrained_model(), games=1, speed=0, seed=0)) == 1
    out = record_gif(pretrained_model(), output=tmp_path / "d.gif", max_frames=50)
    assert (tmp_path / "d.gif").stat().st_size > 0 and out["frames"] > 0


def test_cli_help():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
