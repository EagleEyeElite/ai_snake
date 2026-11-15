"""The Q-network and its trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

INPUT_SIZE = 11
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3


def get_device(name: str = "auto") -> torch.device:
    """``auto`` picks CUDA, then Apple MPS, then CPU."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class Linear_QNet(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, path: str | Path = "model/model.pth") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in self.state_dict().items()}, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> Linear_QNet:
        model = cls()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return model.to(device)


class QTrainer:
    def __init__(self, model: nn.Module, lr: float, gamma: float):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = next(model.parameters()).device
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done) -> float:
        dev = self.device
        state = torch.tensor(np.array(state), dtype=torch.float, device=dev)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float, device=dev)
        action = torch.tensor(np.array(action), dtype=torch.long, device=dev)
        reward = torch.tensor(np.array(reward), dtype=torch.float, device=dev)

        if state.dim() == 1:  # single sample -> batch of one
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # 1. predicted Q values for the current state
        pred = self.model(state)

        # 2. Q_new = r + gamma * max(Q(next_state)) for non-terminal states
        done = torch.tensor(done, dtype=torch.bool, device=dev)
        with torch.no_grad():
            next_q = self.model(next_state).max(dim=1).values
            q_new = reward + self.gamma * next_q * (~done)
            target = pred.detach().clone()
            target[torch.arange(len(done), device=dev), action.argmax(dim=1)] = q_new

        # 3. backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
