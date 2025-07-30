import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

class GeGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.w_gate = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        self.b_gate = nn.Parameter(torch.zeros(hidden_dim))
        self.w2 = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(output_dim))
        self.gelu = nn.GELU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = torch.einsum("bi,ih->bh", x, self.w1) + self.b1
        g = torch.einsum("bi,ih->bh", x, self.w_gate) + self.b_gate
        h = z * self.gelu(g)
        return torch.einsum("bh,ho->bo", h, self.w2) + self.b2

class ReLUFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.w2 = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(output_dim))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.relu(torch.einsum("bi,ih->bh", x, self.w1) + self.b1)
        return torch.einsum("bh,ho->bo", h, self.w2) + self.b2

class LitModel(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc, prog_bar=False)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc, prog_bar=False)
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def get_dataloaders(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_len = int(len(dataset) * 0.9)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader

def bootstrap_ci(data, n_bootstrap=1000, alpha=0.05):
    data = np.array(data)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return np.mean(data), lower, upper

def random_search_trial(hidden_dim, batch_size, lr, model_type):
    train_loader, val_loader, test_loader = get_dataloaders(batch_size)
    input_dim = 28 * 28
    output_dim = 10

    if model_type == "GeGLU":
        model = GeGLU(input_dim, hidden_dim, output_dim)
    else:
        model = ReLUFFN(input_dim, hidden_dim, output_dim)

    lit_model = LitModel(model, lr)
    trainer = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    trainer.fit(lit_model, train_loader, val_loader)

    val_acc = trainer.callback_metrics.get("val_acc", torch.tensor(0.0)).item()
    test_result = trainer.test(lit_model, test_loader, verbose=False)
    test_acc = test_result[0]["test_acc"] if test_result else 0.0

    return val_acc, test_acc

def run_experiment(k_trials_list=[2,4,8]):
    hidden_dims = [2, 4, 8, 16]
    batch_sizes = [8, 64]
    lrs = [1e-1, 1e-2, 1e-3, 1e-4]
    results = {}

    for k_trials in k_trials_list:
        results[k_trials] = {"GeGLU": {}, "ReLU": {}}
        for model_type in ["GeGLU", "ReLU"]:
            for hdim in hidden_dims:
                best_val_acc = -1
                best_test_accs = []
                for _ in tqdm(range(k_trials), desc=f"{model_type} hidden_dim={hdim} k={k_trials}"):
                    bs = random.choice(batch_sizes)
                    lr = random.choice(lrs)
                    val_acc, test_acc = random_search_trial(hdim, bs, lr, model_type)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_test_accs = [test_acc]
                    elif val_acc == best_val_acc:
                        best_test_accs.append(test_acc)

                mean_acc, ci_lower, ci_upper = bootstrap_ci(best_test_accs)
                results[k_trials][model_type][hdim] = (mean_acc, ci_lower, ci_upper)

    for k_trials in k_trials_list:
        plt.figure(figsize=(8, 5))
        for model_type in ["GeGLU", "ReLU"]:
            means = []
            lowers = []
            uppers = []
            for hdim in hidden_dims:
                mean_acc, ci_l, ci_u = results[k_trials][model_type][hdim]
                means.append(mean_acc)
                lowers.append(mean_acc - ci_l)
                uppers.append(ci_u - mean_acc)
            plt.errorbar(hidden_dims, means, yerr=[lowers, uppers], label=model_type, capsize=3, marker='o')

        plt.title(f"MNIST Test Accuracy vs Hidden Dim (k={k_trials})")
        plt.xlabel("Hidden Dimension")
        plt.ylabel("Test Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"mnist_accuracy_k{k_trials}.png")
        plt.show()

if __name__ == "__main__":
    run_experiment()
