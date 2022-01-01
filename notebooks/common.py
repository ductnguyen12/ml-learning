from typing import Callable, Dict
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class ModelManager:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Callable[..., optim.Optimizer],
        device: str = None,
        hyperparameters: Dict = None,
    ):
        if device is None:
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device: str = device
        self.hyperparameters = hyperparameters or {}
        self.model: nn.Module = model.to(self.device)
        self.loss_fn: nn.Module = loss_fn
        self.optimizer: optim.Optimizer = optimizer(self.model.parameters(), **self.hyperparameters)

    def train(self, data_loader: DataLoader):
        print(f"Run training using {self.device}")
        size: int = len(data_loader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(self.device), y.to(self.device)

            pred, _ = self.model(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    def test(self, data_loader: DataLoader):
        size: int = len(data_loader.dataset)
        num_batches: int = len(data_loader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred, _ = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def run_train_loop(
        self, 
        train_data_loader: DataLoader,
        test_data_loader: DataLoader,
        epochs: int,
        model_params_path: str = None
    ):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            self.train(data_loader=train_data_loader)
            self.test(data_loader=test_data_loader)
        if model_params_path:
            self.save_parameters(model_params_path)

    def save_parameters(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_parameters(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
