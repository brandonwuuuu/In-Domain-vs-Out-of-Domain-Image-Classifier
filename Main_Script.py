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

cudnn.benchmark = True


# ---------------------------------------------------------------------
# DATASETS
# ---------------------------------------------------------------------

# In-Domain Dataset: labeled images
# IT Caches images in RAM for faster training
class InDomainDataset(Dataset):
    def __init__(self, root, transform, cache: bool = False):
        self.transform = transform
        self.cache = cache

        class_names = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        self.samples = []
        for c in class_names:
            cidx = self.class_to_idx[c]
            cdir = os.path.join(root, c)
            for f in os.listdir(cdir):
                p = os.path.join(cdir, f)
                if os.path.isfile(p):
                    self.samples.append((p, cidx))

        # Cache images in RAM (optional)
        self._cached_imgs = None
        if self.cache:
            self._cached_imgs = []
            for path, _ in self.samples:
                img = Image.open(path).convert("RGB")
                # deepcopy so we can safely reuse the image
                self._cached_imgs.append(deepcopy(img))
                img.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        if self.cache and self._cached_imgs is not None:
            img = self._cached_imgs[idx]
        else:
            img = Image.open(path).convert("RGB")

        return self.transform(img), label

# Out-of-Domain Dataset: unlabeled images
# IT Caches images in RAM for faster training
class OutDomainDataset(Dataset):
    def __init__(self, root, transform, cache: bool = False):
        self.transform = transform
        self.cache = cache

        self.paths = []
        for c in os.listdir(root):
            cdir = os.path.join(root, c)
            if os.path.isdir(cdir):
                for f in os.listdir(cdir):
                    p = os.path.join(cdir, f)
                    if os.path.isfile(p):
                        self.paths.append(p)

        # Cache images in RAM
        self._cached_imgs = None
        if self.cache:
            self._cached_imgs = []
            for path in self.paths:
                img = Image.open(path).convert("RGB")
                self._cached_imgs.append(deepcopy(img))
                img.close()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.cache and self._cached_imgs is not None:
            img = self._cached_imgs[idx]
        else:
            img = Image.open(self.paths[idx]).convert("RGB")

        return self.transform(img)

# Evaluation Dataset: labeled images
class EvalDataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        class_names = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        self.samples = []
        for c in class_names:
            cidx = self.class_to_idx[c]
            cdir = os.path.join(root, c)
            for f in os.listdir(cdir):
                p = os.path.join(cdir, f)
                if os.path.isfile(p):
                    self.samples.append((p, cidx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        return self.transform(img), label



# ---------------------------------------------------------------------
# FAST + ACCURATE MODEL
# ---------------------------------------------------------------------

# Simple CNN architecture
# Uses convolutional layers with BatchNorm and ReLU activations to train quickly
# Also includes Dropout for regularization
class FastCNN(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )

        self.dropout = nn.Dropout(0.4)  # prevents overfitting
        self.fc = nn.Linear(256, nc)

    def forward(self, x):
        x = self.features(x)
        x = x.mean((2, 3))
        x = self.dropout(x)
        return self.fc(x)

# ---------------------------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------------------------

# Trains a FastCNN model using in-domain and out-of-domain datasets
def learn(path_to_in_domain, path_to_out_domain):
    img_size = 128
    num_epochs = 50
    lambda_ood = 0.05
    kernel_size = int(0.1 * img_size)

    # Ensure kernel size is odd for GaussianBlur
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = 0.1 + 0.3 * np.random.rand()

    # Data augmentation transforms
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.GaussianBlur(kernel_size, sigma),
    ])
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

    # TRAINING LOOP 
    # Learn from both in-domain and out-of-domain data with entropy regularization
    for epoch in range(num_epochs):
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}...")

        # Zip the two dataloaders together to retrieve batches simultaneously
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

    model.eval()
    return model


# ---------------------------------------------------------------------
# ACCURACY FUNCTION
# ---------------------------------------------------------------------

# Computes accuracy of the model on a given evaluation dataset
def compute_accuracy(path_to_eval_folder, model):
    # Default transforms if none provided
    img_size = 128
    kernel_size = int(0.1 * img_size)

    # Ensure kernel size is odd for GaussianBlur
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = 0.1 + 0.3 * np.random.rand()

    # Data augmentation transforms
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.GaussianBlur(kernel_size, sigma),
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

# ---------------------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------------------
# Check if this script is being run directly
if __name__ == "__main__":
    startTime = time()

    # Change this value to experiment with different image sizes
    img_size = 128
    num_epochs = 50
    lambda_ood = 0.05
    kernel_size = int(0.1 * img_size)

    # Ensure kernel size is odd for GaussianBlur
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = 0.1 + 0.3 * np.random.rand()

    # Data augmentation transforms
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.GaussianBlur(kernel_size, sigma),
    ])

    # Train the model with the specified parameters and data
    model = learn('./in-domain-train','./out-domain-train')

    # Measure and print model training time
    modelEndTime = time()
    print(f"Model training time: {modelEndTime - startTime:.2f} seconds")

    inTrainAccuracy = compute_accuracy('./in-domain-train', model)
    outTrainAccuracy = compute_accuracy('./out-domain-train', model)
    inAccuracy = compute_accuracy('./in-domain-eval', model)
    outAccuracy = compute_accuracy('./out-domain-eval', model)
    
    endTime = time()
    print(f"Total evaluation time: {endTime - modelEndTime:.2f} seconds")
    print(f"In-domain Train Accuracy: {inTrainAccuracy*100:.2f}%")
    print(f"Out-of-domain Train Accuracy: {outTrainAccuracy*100:.2f}%")
    print(f"In-domain Eval Accuracy: {inAccuracy*100:.2f}%")
    print(f"Out-of-domain Eval Accuracy: {outAccuracy*100:.2f}%")