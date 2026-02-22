import pathlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


class DemoDataset(Dataset):
    def __init__(self, obs: np.ndarray, act: np.ndarray):
        if obs.shape[0] != act.shape[0]:
            raise ValueError("Observation and action arrays must have matching first dimension.")
        self.obs = torch.from_numpy(obs.astype(np.float32))
        self.act = torch.from_numpy(act.astype(np.float32))

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx]


class BehaviorCloningPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.ReLU())
            last_dim = size
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.net(obs)
        return torch.tanh(out)


@dataclass
class Normalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / self.std


def load_data(obs_path: str, act_path: str):
    obs = np.load(obs_path)
    act = np.load(act_path)
    if obs.ndim != 2:
        raise ValueError("Observations must be 2D (batch, features).")
    if act.ndim == 2 and act.shape[1] == 1:
        act = act.squeeze(1)
    act = act.reshape(-1, 1)
    return obs.astype(np.float32), act.astype(np.float32)


def compute_normalizer(obs: torch.Tensor) -> Normalizer:
    mean = obs.mean(dim=0)
    std = obs.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return Normalizer(mean=mean, std=std)


def main():
    data_dir = pathlib.Path("data")
    obs_path = data_dir / "demos_obs.npy"
    act_path = data_dir / "demos_act.npy"
    output_path = data_dir / "policy.pt"
    hidden_sizes = [64, 64]
    batch_size = 128
    epochs = 50
    learning_rate = 1e-3
    val_frac = 0.1
    device = torch.device("cpu")

    obs_np, act_np = load_data(str(obs_path), str(act_path))
    dataset = DemoDataset(obs_np, act_np)

    val_size = int(len(dataset) * val_frac)
    train_size = len(dataset) - val_size
    if val_size == 0:
        train_dataset = dataset
        val_dataset = None
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    obs_dim = obs_np.shape[1]
    model = BehaviorCloningPolicy(obs_dim, hidden_sizes).to(device)

    # Compute normalizer using full dataset for stability
    normalizer = compute_normalizer(dataset.obs.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for obs_batch, act_batch in loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                obs_norm = normalizer.normalize(obs_batch)
                pred = model(obs_norm)
                loss = loss_fn(pred, act_batch)
                total_loss += loss.item() * obs_batch.size(0)
                count += obs_batch.size(0)
        return total_loss / max(count, 1)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        samples = 0
        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            obs_norm = normalizer.normalize(obs_batch)
            pred = model(obs_norm)
            loss = loss_fn(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * obs_batch.size(0)
            samples += obs_batch.size(0)

        avg_train = running_loss / max(samples, 1)
        if val_loader is not None:
            val_loss = evaluate(val_loader)
            print(f"Epoch {epoch:03d}: train_loss={avg_train:.6f}, val_loss={val_loss:.6f}")
        else:
            print(f"Epoch {epoch:03d}: train_loss={avg_train:.6f}")

    ckpt = {
        "model_state": model.state_dict(),
        "obs_mean": normalizer.mean.cpu(),
        "obs_std": normalizer.std.cpu(),
        "hidden_sizes": hidden_sizes,
        "obs_dim": obs_dim,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    print(f"Saved policy to {output_path}")


if __name__ == "__main__":
    main()
