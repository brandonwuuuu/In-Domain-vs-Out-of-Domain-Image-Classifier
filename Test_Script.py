import os
from time import time
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
from copy import deepcopy
from contextlib import nullcontext
from Main_Script import InDomainDataset, OutDomainDataset, FastCNN, EvalDataset

cudnn.benchmark = True

def compute_accuracy(path_to_eval_folder, model, transform=None, img_size=128):
    # Default transforms if none provided
    img_size = 128
    kernel_size = int(0.1 * img_size)

    # Ensure kernel size is odd for GaussianBlur
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = 0.1 + 0.3 * np.random.rand()

    # Data augmentation transforms
    if transform is None:
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
    dataset = EvalDataset(path_to_eval_folder, transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    device = next(model.parameters()).device
    model.eval()

    correct, total = 0, 0
    # No gradient computation needed for evaluation
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            preds = logits.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def learn_with_history(
    path_to_in_domain,
    path_to_out_domain,
    path_to_in_eval,
    path_to_out_eval,
    lambda_ood=0.02,
    max_epochs=320,
    img_size=64,
    transform=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    # Default transforms if none provided
    if transform is None:
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    # -------------------------------
    # LOAD DATA (with caching)
    # -------------------------------
    in_ds = InDomainDataset(path_to_in_domain, transform, cache=True)
    out_ds = OutDomainDataset(path_to_out_domain, transform, cache=True)

    # DataLoaders with prefetching and pin_memory for speed
    in_loader  = DataLoader(
        in_ds, batch_size=516, shuffle=True,
        num_workers=4,                
        pin_memory=use_cuda,          
    )

    # Out-of-domain DataLoader 
    out_loader = DataLoader(
        out_ds, batch_size=516, shuffle=True,
        num_workers=4,
        pin_memory=use_cuda,
    )

    num_classes = len(in_ds.class_to_idx)

    model = FastCNN(num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    ce_loss = nn.CrossEntropyLoss()

    # scaler only if CUDA
    scaler = GradScaler() if use_cuda else None

    # choose autocast/no-op context correctly
    autocast_ctx = autocast(device_type="cuda") if use_cuda else nullcontext()

    model.train()

    # history dict to store per-epoch results
    history = {
        "epochs": [],
        "in_acc": [],
        "out_acc": [],
        "in_train_acc": [],
        "out_train_acc": []
    }

    for epoch in range(1, max_epochs + 1):
        model.train()

        
        print(f"Epoch {epoch+1}/{max_epochs}...")

        for (imgs_in, labels_in), imgs_out in zip(in_loader, out_loader):

            imgs_in   = imgs_in.to(device, non_blocking=use_cuda)
            labels_in = labels_in.to(device, non_blocking=use_cuda)
            imgs_out  = imgs_out.to(device, non_blocking=use_cuda)

            optimizer.zero_grad()

            # Mixed precision context
            with autocast_ctx:
                logits_in = model(imgs_in)
                loss = ce_loss(logits_in, labels_in)

                logits_out = model(imgs_out)
                p = torch.softmax(logits_out, dim=1)
                entropy = -(p * torch.log(p + 1e-8)).sum(1).mean()
                loss = loss - lambda_ood * entropy

            # Backpropagation with optional mixed precision
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        acc_in = compute_accuracy(path_to_in_eval, model, img_size=img_size, transform=transform)
        acc_out = compute_accuracy(path_to_out_eval, model, img_size=img_size, transform=transform)
        acc_train_in = compute_accuracy(path_to_in_domain, model, img_size=img_size, transform=transform)
        acc_train_out = compute_accuracy(path_to_out_domain, model, img_size=img_size, transform=transform)

        history["epochs"].append(epoch)
        history["in_acc"].append(acc_in)
        history["out_acc"].append(acc_out)
        history["in_train_acc"].append(acc_train_in)
        history["out_train_acc"].append(acc_train_out)
        model.eval()
    return model, history
def run_experiments_transforms(
    transform_dict,
    lambda_ood=0.05,
    max_epochs=50,
    path_to_in_train="",
    path_to_out_train="",
    path_to_in_eval="",
    path_to_out_eval="",
    img_size=64,
):
    all_histories = {}

    for name, transform in transform_dict.items():
        print("\n==============================")
        print(f"Training with transform = {name}, lambda_ood = {lambda_ood}")
        print("==============================")

        model, hist = learn_with_history(
            path_to_in_train,
            path_to_out_train,
            path_to_in_eval,
            path_to_out_eval,
            lambda_ood=lambda_ood,
            max_epochs=max_epochs,
            img_size=img_size,
            transform=transform,
        )

        all_histories[name] = hist

    # ---------- PLOTS ----------

    # In-domain TRAIN accuracy
    plt.figure(figsize=(10, 6))
    for name, hist in all_histories.items():
        plt.plot(hist["epochs"], hist["in_train_acc"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("In-domain Train Accuracy")
    plt.title(f"In-domain Train Accuracy vs Epoch (lambda_ood={lambda_ood})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Out-domain TRAIN accuracy
    plt.figure(figsize=(10, 6))
    for name, hist in all_histories.items():
        plt.plot(hist["epochs"], hist["out_train_acc"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Out-domain Train Accuracy")
    plt.title(f"Out-domain Train Accuracy vs Epoch (lambda_ood={lambda_ood})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # In-domain EVAL accuracy
    plt.figure(figsize=(10, 6))
    for name, hist in all_histories.items():
        plt.plot(hist["epochs"], hist["in_acc"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("In-domain Eval Accuracy")
    plt.title(f"In-domain Eval Accuracy vs Epoch (lambda_ood={lambda_ood})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Out-domain EVAL accuracy
    plt.figure(figsize=(10, 6))
    for name, hist in all_histories.items():
        plt.plot(hist["epochs"], hist["out_acc"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Out-of-domain Eval Accuracy")
    plt.title(f"Out-of-domain Eval Accuracy vs Epoch (lambda_ood={lambda_ood})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Difference (ID - OOD) on EVAL
    plt.figure(figsize=(10, 6))
    for name, hist in all_histories.items():
        epochs = hist["epochs"]
        in_acc = np.array(hist["in_acc"])
        out_acc = np.array(hist["out_acc"])
        diff = in_acc - out_acc
        plt.plot(epochs, diff, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy difference (ID - OOD)")
    plt.title(f"In-domain - Out-of-domain Eval Accuracy vs Epoch (lambda_ood={lambda_ood})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return all_histories

if __name__ == "__main__":
    img_size = 128

    kernel_size = int(0.1 * img_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Some example transforms to compare
    kernel_sizes = [5, 11, 17, 25, 35]

    transform_configs = {
        f"blur{k}": T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.GaussianBlur(kernel_size=k, sigma=5),
        ])
        for k in kernel_sizes
    }

    img_size = 128

    all_histories = run_experiments_transforms(
        transform_dict=transform_configs,
        lambda_ood=0.05,
        max_epochs=50,
        img_size=img_size,
        path_to_in_train="",
        path_to_out_train="",
        path_to_in_eval="",
        path_to_out_eval="",
    )

