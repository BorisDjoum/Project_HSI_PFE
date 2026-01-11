# ...existing code...
"""
Transfer learning mono-fichier — PyTorch, lazy loader, memory-friendly, MPS/CUDA support.
Remplace la version TensorFlow. Conçu pour macOS Apple Silicon (MPS) ou CUDA.
"""

import argparse
import os
import time
import tempfile
from collections import Counter
import numpy as np
import math
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.models as tvmodels

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Device selection: prefer MPS on Apple Silicon, then CUDA, else CPU
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def extract_class_from_filename(file_name):
    base = os.path.basename(file_name)
    if 'G' in base:
        part = base.split('G')[-1]
        lbl = part.split('_')[0]
        return int(lbl)
    # fallback: try digits in name
    digits = ''.join(ch for ch in base if ch.isdigit())
    return int(digits) if digits else 0


def build_index(feature_files, data_type='reflec', patch_size=31, stride=10, max_patches_per_file=50, max_total=5000):
    """
    Build lazy index of (file, i, j) for reflec or (file, row_idx, None) for ghost.
    Returns list of (file_path, param1, param2) and labels list.
    """
    half = patch_size // 2
    index = []
    labels = []
    total = 0

    for f in feature_files:
        if total >= max_total:
            break
        try:
            arr = np.load(f, mmap_mode='r')
        except Exception:
            continue

        label = extract_class_from_filename(f) - 1

        if data_type == 'reflec' and arr.ndim == 3:
            h, w, _ = arr.shape
            count = 0
            for i in range(half, h - half, stride):
                for j in range(half, w - half, stride):
                    index.append((f, int(i), int(j)))
                    labels.append(label)
                    count += 1
                    total += 1
                    if count >= max_patches_per_file or total >= max_total:
                        break
                if count >= max_patches_per_file or total >= max_total:
                    break

        elif data_type == 'ghost' and arr.ndim == 2:
            n = min(arr.shape[0], max_patches_per_file)
            for i in range(n):
                index.append((f, int(i), None))
                labels.append(label)
                total += 1
                if total >= max_total:
                    break

        else:
            # file type unsupported: skip
            continue

    return index, np.array(labels, dtype=np.int64)


class LazyHyperspecDataset(Dataset):
    """
    Lazy dataset: loads small patch from disk on __getitem__.
    Yields tensor (C,H,W), label.
    """

    def __init__(self, index_list, labels, data_type='reflec', patch_size=31, to_rgb=True):
        self.index = index_list
        self.labels = labels
        self.data_type = data_type
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.to_rgb = to_rgb

    def __len__(self):
        return len(self.labels)

    def _convert_to_rgb(self, patch):
        # patch: H x W x B  or H x W x 1
        if patch.ndim == 2:
            patch = patch[:, :, None]
        b = patch.shape[2]
        if b == 3:
            out = patch
        elif b == 1:
            out = np.repeat(patch, 3, axis=2)
        else:
            idx = [0, b // 2, b - 1]
            out = patch[:, :, idx]
        # normalize per-patch
        mn, mx = out.min(), out.max()
        if mx > mn:
            out = (out - mn) / (mx - mn)
        return out.astype(np.float32)

    def __getitem__(self, idx):
        fpath, p1, p2 = self.index[idx]
        label = int(self.labels[idx])

        try:
            arr = np.load(fpath, mmap_mode='r')
            if self.data_type == 'reflec':
                i, j = p1, p2
                patch = arr[i - self.half: i + self.half + 1, j - self.half: j + self.half + 1, :]
                # If bands last, shape is (H,W,B)
                if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                    # fallback: center crop / pad to desired size
                    patch = _resize_or_pad(patch, self.patch_size)
                if self.to_rgb:
                    patch = self._convert_to_rgb(patch)
                # convert to C,H,W
                tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
            else:  # ghost
                row_idx = p1
                arr_row = arr[row_idx]
                # reshape to square
                nfeat = arr_row.shape[0]
                side = int(math.ceil(math.sqrt(nfeat)))
                padded = np.zeros(side * side, dtype=arr_row.dtype)
                padded[:nfeat] = arr_row
                patch = padded.reshape(side, side)
                # resize to patch_size using torch
                patch_t = torch.from_numpy(patch.astype(np.float32))[None, None]
                patch_t = F.interpolate(patch_t, size=(self.patch_size, self.patch_size), mode='bilinear', align_corners=False)
                patch = patch_t.squeeze().numpy()
                if self.to_rgb:
                    patch = np.repeat(patch[:, :, None], 3, axis=2)
                tensor = torch.from_numpy(patch).permute(2, 0, 1).float()

            # basic normalization: 0-1
            if tensor.max() > 1.0:
                tensor = tensor / 255.0

            return tensor, label

        except Exception as e:
            # return zero tensor instead of crashing
            return torch.zeros(3, self.patch_size, self.patch_size, dtype=torch.float32), label


def _resize_or_pad(patch, size):
    """Ensure patch is size x size x C — pad or center-crop and pad."""
    h, w = patch.shape[0], patch.shape[1]
    c = patch.shape[2] if patch.ndim == 3 else 1
    out = np.zeros((size, size, c), dtype=patch.dtype)
    # center crop or pad
    top = max(0, (size - h) // 2)
    left = max(0, (size - w) // 2)
    insert_h = min(h, size)
    insert_w = min(w, size)
    src_top = max(0, (h - size) // 2)
    src_left = max(0, (w - size) // 2)
    out[top:top+insert_h, left:left+insert_w] = patch[src_top:src_top+insert_h, src_left:src_left+insert_w, :c] if c>1 else patch[src_top:src_top+insert_h, src_left:src_left+insert_w][:, :, None]
    return out


def get_backbone(arch, pretrained=True):
    arch = arch.lower()
    if 'resnet' in arch:
        model = tvmodels.resnet50(weights=tvmodels.ResNet50_Weights.DEFAULT) if pretrained else tvmodels.resnet50()
        feat_dim = model.fc.in_features
        # remove fc
        modules = list(model.children())[:-1]
        backbone = nn.Sequential(*modules)
    elif 'mobilenet' in arch:
        model = tvmodels.mobilenet_v2(weights=tvmodels.MobileNet_V2_Weights.DEFAULT) if pretrained else tvmodels.mobilenet_v2()
        feat_dim = model.classifier[1].in_features
        backbone = nn.Sequential(*list(model.features), nn.AdaptiveAvgPool2d((1,1)))
    elif 'densenet' in arch:
        model = tvmodels.densenet121(weights=tvmodels.DenseNet121_Weights.DEFAULT) if pretrained else tvmodels.densenet121()
        feat_dim = model.classifier.in_features
        backbone = nn.Sequential(model.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1,1)))
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return backbone, feat_dim


class TransferModel(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True):
        super().__init__()
        self.backbone, feat_dim = get_backbone(arch, pretrained=pretrained)
        # freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return running_loss / total, 100. * correct / total


def validate_one_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            running_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return running_loss / total, 100. * correct / total


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--data_type', choices=['reflec', 'ghost'], default='reflec')
    p.add_argument('--patch_size', type=int, default=32)
    p.add_argument('--stride', type=int, default=10)
    p.add_argument('--max_patches_per_file', type=int, default=50)
    p.add_argument('--max_samples', type=int, default=2000)
    p.add_argument('--arch', choices=['resnet50', 'mobilenet_v2', 'densenet121'], default='mobilenet_v2')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--save_model', type=str, default='best_transfer.pt')
    p.add_argument('--save_history', type=str, default='history_transfer.png')
    return p.parse_args()


def main():
    args = parse_args()

    # find files
    feature_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.npy')]
    if len(feature_files) == 0:
        raise SystemExit("No .npy files found in data_dir")

    print(f"Device: {device}")
    print(f"Files: {len(feature_files)}  Arch: {args.arch}  Patch: {args.patch_size}")

    # build index (lazy)
    idx_list, labels = build_index(feature_files, data_type=args.data_type,
                                   patch_size=args.patch_size, stride=args.stride,
                                   max_patches_per_file=args.max_patches_per_file,
                                   max_total=args.max_samples)
    if len(idx_list) == 0:
        raise SystemExit("No patches indexed - adjust stride/patch_size/max_patches_per_file")

    # split indices
    train_idx, test_idx, y_train, y_test = train_test_split(
        np.arange(len(idx_list)), labels, test_size=0.2, stratify=labels, random_state=42)
    train_idx, val_idx, y_train, y_val = train_test_split(train_idx, y_train, test_size=0.1, stratify=y_train, random_state=42)

    train_index = [idx_list[i] for i in train_idx]
    val_index = [idx_list[i] for i in val_idx]
    test_index = [idx_list[i] for i in test_idx]
    y_train_arr = labels[train_idx]
    y_val_arr = labels[val_idx]
    y_test_arr = labels[test_idx]

    print(f"Indexed patches: total={len(idx_list)}, train={len(train_index)}, val={len(val_index)}, test={len(test_index)}")
    gc.collect()

    train_ds = LazyHyperspecDataset(train_index, y_train_arr, data_type=args.data_type, patch_size=args.patch_size, to_rgb=True)
    val_ds = LazyHyperspecDataset(val_index, y_val_arr, data_type=args.data_type, patch_size=args.patch_size, to_rgb=True)
    test_ds = LazyHyperspecDataset(test_index, y_test_arr, data_type=args.data_type, patch_size=args.patch_size, to_rgb=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    num_classes = int(max(labels) + 1)

    model = TransferModel(args.arch, num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    patience = 0
    early_stop_patience = 150

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.2f}%  val_loss={val_loss:.4f} val_acc={val_acc:.2f}%  time={time.time()-t0:.1f}s")

        if val_acc > best_val:
            best_val = val_acc
            patience = 0
            torch.save({'model_state': model.state_dict(), 'arch': args.arch, 'num_classes': num_classes}, args.save_model)
            print(f"  Saved best model ({best_val:.2f}%) -> {args.save_model}")
        else:
            patience += 1
            if patience >= early_stop_patience:
                print("Early stopping")
                break

    # load best and evaluate
    chk = torch.load(args.save_model, map_location=device)
    model.load_state_dict(chk['model_state'])
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("Confusion matrix:")
    print(confusion_matrix(all_targets, all_preds))
    print("Classification report:")
    print(classification_report(all_targets, all_preds, zero_division=0))

    # save history plot
    if args.save_history:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], label='train')
        axes[0].plot(epochs, history['val_loss'], label='val')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
        axes[1].plot(epochs, history['train_acc'], label='train')
        axes[1].plot(epochs, history['val_acc'], label='val')
        axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True)
        plt.tight_layout()
        plt.savefig(args.save_history, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved history: {args.save_history}")


if __name__ == '__main__':
    main()
# ...existing code...