import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

pl.seed_everything(42)

class FFN_GeGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        scale = 0.05
        self.w_in = nn.Parameter(torch.randn(input_dim, hidden_dim)*scale)
        self.b_in = nn.Parameter(torch.zeros(hidden_dim))
        self.w_gate = nn.Parameter(torch.randn(input_dim, hidden_dim)*scale)
        self.b_gate = nn.Parameter(torch.zeros(hidden_dim))
        self.w_out = nn.Parameter(torch.randn(hidden_dim, output_dim)*scale)
        self.b_out = nn.Parameter(torch.zeros(output_dim))
        self.gelu = nn.GELU()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = torch.einsum("bi,ih->bh", x, self.w_in) + self.b_in
        g = torch.einsum("bi,ih->bh", x, self.w_gate) + self.b_gate
        h = z * self.gelu(g)
        return torch.einsum("bh,ho->bo", h, self.w_out) + self.b_out

class FFN_ReLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        scale = 0.05
        self.w_in = nn.Parameter(torch.randn(input_dim, hidden_dim)*scale)
        self.b_in = nn.Parameter(torch.zeros(hidden_dim))
        self.w_out = nn.Parameter(torch.randn(hidden_dim, output_dim)*scale)
        self.b_out = nn.Parameter(torch.zeros(output_dim))
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.relu(torch.einsum("bi,ih->bh", x, self.w_in) + self.b_in)
        return torch.einsum("bh,ho->bo", h, self.w_out) + self.b_out

class LitModel(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss
    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc, prog_bar=False)
        return acc
    def test_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc, prog_bar=False)
        return acc
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def load_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_len = int(len(dataset)*0.9)
    val_len = len(dataset) - train_len
    trainset, valset = random_split(dataset, [train_len, val_len])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, valloader, testloader

def bootstrap_acc(model, testloader, trials=1000):
    accs = []
    preds = []
    targets = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for x,y in testloader:
            preds.append(model(x.to(device)).argmax(dim=1).cpu())
            targets.append(y)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    n = len(targets)
    for _ in range(trials):
        idxs = torch.randint(0, n, (n,))
        acc = (preds[idxs] == targets[idxs]).float().mean().item()
        accs.append(acc)
    mean = np.mean(accs)
    std = np.std(accs)
    return mean, std

def run_trial(model_class, hidden_dim, batch_size, lr, epochs=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    input_dim = 28*28
    output_dim = 10
    model = model_class(input_dim, hidden_dim, output_dim)
    lit_model = LitModel(model, lr)
    trainloader, valloader, testloader = load_data(batch_size)
    trainer = Trainer(max_epochs=epochs, enable_progress_bar=False, logger=False, enable_checkpointing=False, devices=1 if device=='cuda' else None, accelerator=device)
    trainer.fit(lit_model, trainloader, valloader)
    val_results = trainer.validate(lit_model, valloader, verbose=False)
    val_acc = val_results[0]['val_acc']
    test_results = trainer.test(lit_model, testloader, verbose=False)
    test_acc = test_results[0]['test_acc']
    return val_acc, test_acc, lit_model, testloader

def experiment(k_trials_list=[2,4,8]):
    hidden_dims = [2,4,8,16]
    batch_sizes = [8,64]
    lrs = [1e-1,1e-2,1e-3,1e-4]

    results = {}  # model -> k -> hidden_dim -> (mean_test_acc, std_test_acc)

    for model_name, model_class in [("GeGLU", FFN_GeGLU), ("ReLU", FFN_ReLU)]:
        results[model_name] = {}
        for k in k_trials_list:
            results[model_name][k] = {}
            for hidden_dim in hidden_dims:
                best_val_acc = -1
                best_test_acc = None
                best_model = None
                best_testloader = None
                for _ in range(k):
                    bs = random.choice(batch_sizes)
                    lr = random.choice(lrs)
                    val_acc, test_acc, lit_model, testloader = run_trial(model_class, hidden_dim, bs, lr)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_test_acc = test_acc
                        best_model = lit_model
                        best_testloader = testloader
                mean_acc, std_acc = bootstrap_acc(best_model.model, best_testloader)
                results[model_name][k][hidden_dim] = (mean_acc, std_acc)
                print(f"{model_name} k={k} hidden={hidden_dim} best val_acc={best_val_acc:.4f} test_acc={mean_acc:.4f} ± {std_acc:.4f}")

    # Plotting results
    for k in k_trials_list:
        plt.figure(figsize=(8,6))
        for model_name in results.keys():
            means = []
            stds = []
            for h in hidden_dims:
                m,s = results[model_name][k][h]
                means.append(m)
                stds.append(s)
            plt.errorbar(hidden_dims, means, yerr=stds, label=model_name, marker='o', capsize=4)
        plt.title(f"Test Accuracy vs Hidden Dim (k={k} trials)")
        plt.xlabel("Hidden Dimension")
        plt.ylabel("Test Accuracy")
        plt.legend()
        plt.grid(True)
        plt.ylim(0.85, 1.0)
        plt.show()

    # Final claim check
    for k in k_trials_list:
        print(f"\nClaim check for k={k}:")
        for h in hidden_dims:
            g_acc, _ = results["GeGLU"][k][h]
            r_acc, _ = results["ReLU"][k][h]
            print(f"Hidden dim {h}: GeGLU={g_acc:.4f}, ReLU={r_acc:.4f} => {'GeGLU better' if g_acc > r_acc else 'ReLU better or equal'}")

if __name__=="__main__":
    experiment()
