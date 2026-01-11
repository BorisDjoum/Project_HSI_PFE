"""
CNN 3D PyTorch optimisé pour Apple Silicon (M1/M2/M3/M4)
Utilise MPS (Metal Performance Shaders) pour l'accélération GPU
"""

import argparse
import os
import numpy as np
from collections import Counter
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def configure_device():
    """
    Configure PyTorch pour utiliser le GPU Apple Silicon (MPS).
    
    PyTorch 2.0+ supporte le backend MPS (Metal Performance Shaders)
    pour les puces M1/M2/M3/M4.
    """
    print(f"\n{'='*70}")
    print("CONFIGURATION DEVICE (PyTorch)")
    print(f"{'='*70}")
    print(f"Version PyTorch: {torch.__version__}")
    
    # Vérification des devices disponibles
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n✅ GPU Apple Silicon (MPS) détecté!")
        print(f"   Device: {device}")
        print(f"   Backend: Metal Performance Shaders")
        
        # Test du device
        try:
            test_tensor = torch.randn(10, 10).to(device)
            result = torch.matmul(test_tensor, test_tensor)
            print(f"   ✓ Test GPU réussi: calcul matriciel OK")
        except Exception as e:
            print(f"   ⚠️  Erreur test GPU: {e}")
            print(f"   Fallback vers CPU")
            device = torch.device("cpu")
    
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n✅ GPU CUDA détecté!")
        print(f"   Device: {device}")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    
    else:
        device = torch.device("cpu")
        print(f"\n⚠️  Aucun GPU détecté - Utilisation du CPU")
        print(f"   Pour activer MPS sur Mac:")
        print(f"   pip install --upgrade torch torchvision")
    
    print(f"{'='*70}\n")
    return device


def extract_class_from_filename(file_name):
    """Extrait le label de classe depuis le nom de fichier."""
    base_name = os.path.basename(file_name)
    if 'G' in base_name:
        class_part = base_name.split('G')[-1]
        class_label = class_part.split('_')[0]
        return int(class_label)
    else:
        raise ValueError(f"Le nom de fichier {file_name} ne contient pas d'information de classe valide.")


def extract_3d_patches(cube, patch_size=7, stride=1):
    """
    Extrait des patchs 3D d'un cube hyperspectral.
    
    Args:
        cube: Cube 3D (H, W, Bands)
        patch_size: Taille spatiale du patch
        stride: Pas de déplacement
    
    Returns:
        patches: Array 4D (n_patches, patch_size, patch_size, bands)
    """
    h, w, bands = cube.shape
    half_patch = patch_size // 2
    
    patches = []
    
    for i in range(half_patch, h - half_patch, stride):
        for j in range(half_patch, w - half_patch, stride):
            patch = cube[i - half_patch:i + half_patch + 1,
                        j - half_patch:j + half_patch + 1,
                        :]
            
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    
    return np.array(patches)


def load_hyperspectral_data(data_dir, patch_size=7, max_patches_per_file=500, 
                            max_total_samples=50000, stride=3, seed=42):
    """
    Charge les données hyperspectrales et extrait des patchs 3D.
    
    Args:
        data_dir: Liste des chemins vers les fichiers .npy
        patch_size: Taille des patchs 3D
        max_patches_per_file: Nombre max de patchs par fichier
        max_total_samples: Nombre max total de patchs
        stride: Pas pour l'extraction
        seed: Seed pour reproductibilité
    """
    np.random.seed(seed)
    
    all_patches = []
    all_labels = []
    total_patches = 0
    
    print(f"Chargement des données hyperspectrales...")
    print(f"Paramètres: patch_size={patch_size}x{patch_size}, stride={stride}")
    
    for idx, file_name in enumerate(data_dir):
        if total_patches >= max_total_samples:
            print(f"Limite de {max_total_samples} patchs atteinte.")
            break
        
        try:
            cube = np.load(file_name)
            
            if cube.ndim != 3:
                print(f"Fichier ignoré (pas 3D): {file_name}, shape={cube.shape}")
                continue
            
            h, w, bands = cube.shape
            
            if h < patch_size or w < patch_size:
                print(f"Fichier trop petit: {file_name}, shape={cube.shape}")
                continue
            
            patches = extract_3d_patches(cube, patch_size, stride)
            
            if len(patches) > max_patches_per_file:
                indices = np.random.choice(len(patches), max_patches_per_file, replace=False)
                patches = patches[indices]
            
            class_label = extract_class_from_filename(file_name)
            
            if class_label < 1:
                print(f"Label invalide: {file_name}")
                continue
            
            all_patches.append(patches.astype(np.float32))
            all_labels.append(np.full(len(patches), class_label - 1, dtype=np.int64))
            
            total_patches += len(patches)
            
            if (idx + 1) % 10 == 0:
                print(f"Traité {idx + 1}/{len(data_dir)} fichiers, {total_patches} patchs extraits")
        
        except Exception as e:
            print(f"Erreur: {file_name}: {e}")
            continue
    
    if len(all_patches) == 0:
        raise ValueError("Aucune donnée chargée!")
    
    print("Concaténation des patchs...")
    X = np.vstack(all_patches)
    y = np.concatenate(all_labels)
    
    del all_patches, all_labels
    gc.collect()
    
    print(f"\nDonnées chargées:")
    print(f"  X shape: {X.shape} (n_patches, height, width, bands)")
    print(f"  y shape: {y.shape}")
    print(f"  Classes: {np.unique(y)}")
    print(f"  Distribution: {dict(Counter(y))}")
    
    return X, y


class HyperspectralDataset(Dataset):
    """
    Dataset PyTorch pour les patchs hyperspectraux 3D.
    
    PyTorch attend le format: (Channels, Depth, Height, Width)
    Nos données sont: (Height, Width, Bands)
    → On transpose en: (Bands, Height, Width)
    """
    
    def __init__(self, X, y, transform=None):
        # Conversion numpy → torch tensors
        # Transpose (N, H, W, C) → (N, C, H, W) pour PyTorch
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2)  # (N, Bands, H, W)
        self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class Pool3DFallback(nn.Module):
    """
    Wrapper pour max/avg pooling 3D.
    Si l'input est sur MPS et que l'op n'est pas implémentée, on bascule temporairement sur CPU,
    on exécute le pooling, puis on renvoie le résultat sur l'appareil d'origine.
    """
    def __init__(self, kernel_size, pool_type='max', stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool_type = pool_type
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        dev = x.device
        try:
            # Essayer d'exécuter directement (si supporté sur le device)
            if self.pool_type == 'max':
                return F.max_pool3d(x, self.kernel_size, self.stride, self.padding)
            else:
                return F.avg_pool3d(x, self.kernel_size, self.stride, self.padding)
        except NotImplementedError:
            # Fallback: déplacer sur CPU, appliquer le pooling, revenir sur le device d'origine
            x_cpu = x.to('cpu')
            if self.pool_type == 'max':
                y = F.max_pool3d(x_cpu, self.kernel_size, self.stride, self.padding)
            else:
                y = F.avg_pool3d(x_cpu, self.kernel_size, self.stride, self.padding)
            return y.to(dev)

class CNN3D(nn.Module):
    """
    Réseau CNN 3D pour classification hyperspectrale.
    
    Architecture adaptée pour PyTorch avec format (N, C, H, W).
    Pour Conv3D, on traite les bandes spectrales comme la "profondeur" temporelle.
    """
    
    def __init__(self, input_channels, num_classes, architecture='simple', patch_size=7, pool_type='max'):
        super(CNN3D, self).__init__()
        
        self.architecture = architecture
        
        # Choix du type de pooling (max ou avg) — utile pour MPS qui ne supporte pas max_pool3d entièrement.
        Pool = lambda kernel_size: Pool3DFallback(kernel_size, pool_type=pool_type)
        
        # Calcul de la taille après convolutions et pooling
        # Patch initial: patch_size × patch_size
        # Après chaque MaxPool3d(2): divisé par 2
        
        if architecture == 'simple':
            # Architecture simple
            self.conv1 = nn.Conv3d(1, 32, kernel_size=(7, 3, 3), padding=(3, 1, 1))
            self.bn1 = nn.BatchNorm3d(32)
            self.pool1 = Pool((2, 2, 2))
            self.dropout1 = nn.Dropout3d(0.3)

            self.conv2 = nn.Conv3d(32, 64, kernel_size=(5, 3, 3), padding=(2, 1, 1))
            self.bn2 = nn.BatchNorm3d(64)
            self.pool2 = Pool((2, 2, 2))
            self.dropout2 = nn.Dropout3d(0.4)

            # Adaptive pooling pour garantir une sortie de taille fixe
            self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            
            self.fc1 = nn.Linear(64 * 1 * 1 * 1, 128)
            self.dropout3 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, num_classes)

        elif architecture == 'medium':
            # Architecture moyenne
            self.conv1 = nn.Conv3d(1, 32, kernel_size=(7, 3, 3), padding=(3, 1, 1))
            self.bn1 = nn.BatchNorm3d(32)
            self.pool1 = Pool((2, 2, 2))

            self.conv2 = nn.Conv3d(32, 64, kernel_size=(5, 3, 3), padding=(2, 1, 1))
            self.bn2 = nn.BatchNorm3d(64)
            self.pool2 = Pool((2, 2, 2))
            self.dropout1 = nn.Dropout3d(0.3)
            
            self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.bn3 = nn.BatchNorm3d(128)
            self.dropout2 = nn.Dropout3d(0.4)

            # Adaptive pooling pour garantir une sortie de taille fixe
            self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            
            self.fc1 = nn.Linear(128 * 1 * 1 * 1, 256)
            self.dropout3 = nn.Dropout(0.4)
            self.fc2 = nn.Linear(256, 128)
            self.dropout4 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(128, num_classes)

        elif architecture == 'deep':
            self.conv1a = nn.Conv3d(1, 32, kernel_size=(7, 3, 3), padding=(3, 1, 1))
            self.bn1a = nn.BatchNorm3d(32)
            self.conv1b = nn.Conv3d(32, 32, kernel_size=(7, 3, 3), padding=(3, 1, 1))
            self.bn1b = nn.BatchNorm3d(32)
            self.pool1 = Pool((2, 2, 2))
            self.dropout1 = nn.Dropout3d(0.2)

            self.conv2a = nn.Conv3d(32, 64, kernel_size=(5, 3, 3), padding=(2, 1, 1))
            self.bn2a = nn.BatchNorm3d(64)
            self.conv2b = nn.Conv3d(64, 64, kernel_size=(5, 3, 3), padding=(2, 1, 1))
            self.bn2b = nn.BatchNorm3d(64)
            self.pool2 = Pool((2, 2, 2))
            self.dropout2 = nn.Dropout3d(0.3)
            
            self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.bn3 = nn.BatchNorm3d(128)
            self.dropout3 = nn.Dropout3d(0.3)

            # Adaptive pooling pour garantir une sortie de taille fixe
            self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            
            self.fc1 = nn.Linear(128 * 1 * 1 * 1, 256)
            self.bn_fc1 = nn.BatchNorm1d(256)
            self.dropout4 = nn.Dropout(0.4)
            self.fc2 = nn.Linear(256, 128)
            self.dropout5 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(128, num_classes)
        
        else:
            raise ValueError(f"Architecture inconnue: {architecture}")
    
    def forward(self, x):
        # Input: (N, Bands, H, W)
        # Pour Conv3D, on ajoute une dimension: (N, 1, Bands, H, W)
        x = x.unsqueeze(1)
        
        if self.architecture == 'simple':
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            x = self.dropout1(x)
            
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            x = self.dropout2(x)

            x = self.adaptive_pool(x)  # Réduire à (N, 64, 1, 1, 1)
            x = x.view(x.size(0), -1)  # Flatten
            x = F.relu(self.fc1(x))
            x = self.dropout3(x)
            x = self.fc2(x)
        
        elif self.architecture == 'medium':
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            x = self.dropout1(x)
            
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.dropout2(x)
            
            x = self.adaptive_pool(x)  # Réduire à (N, 128, 1, 1, 1)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout3(x)
            x = F.relu(self.fc2(x))
            x = self.dropout4(x)
            x = self.fc3(x)
        
        elif self.architecture == 'deep':
            x = F.relu(self.bn1a(self.conv1a(x)))
            x = F.relu(self.bn1b(self.conv1b(x)))
            x = self.pool1(x)
            x = self.dropout1(x)
            
            x = F.relu(self.bn2a(self.conv2a(x)))
            x = F.relu(self.bn2b(self.conv2b(x)))
            x = self.pool2(x)
            x = self.dropout2(x)
            
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.dropout3(x)
            
            x = self.adaptive_pool(x)  # Réduire à (N, 128, 1, 1, 1)
            x = x.view(x.size(0), -1)
            x = F.relu(self.bn_fc1(self.fc1(x)))
            x = self.dropout4(x)
            x = F.relu(self.fc2(x))
            x = self.dropout5(x)
            x = self.fc3(x)
        
        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entraîne le modèle pour une époque."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Évalue le modèle sur le set de validation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def test_model(model, test_loader, device):
    """Teste le modèle et retourne les prédictions."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())
    
    return np.array(all_targets), np.array(all_preds)


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path=None):
    """Affiche les courbes d'apprentissage."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss
    axes[0].plot(epochs, train_losses, 'b-', label='Train')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(epochs, train_accs, 'b-', label='Train')
    axes[1].plot(epochs, val_accs, 'r-', label='Validation')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Courbes sauvegardées: {save_path}")
    
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D CNN (PyTorch) on hyperspectral data.')
    
    # Données
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing .npy files.')
    parser.add_argument('--patch_size', type=int, default=7,
                        help='Spatial size of 3D patches.')
    parser.add_argument('--stride', type=int, default=3,
                        help='Stride for patch extraction.')
    parser.add_argument('--max_patches_per_file', type=int, default=500,
                        help='Max patches per file.')
    parser.add_argument('--max_samples', type=int, default=50000,
                        help='Max total patches.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set proportion.')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set proportion.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    
    # Architecture
    parser.add_argument('--architecture', type=str, default='medium',
                        choices=['simple', 'medium', 'deep'],
                        help='Model architecture.')
    
    # Entraînement
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop'],
                        help='Optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization).')
    
    # Callbacks
    parser.add_argument('--early_stopping', type=int, default=15,
                        help='Early stopping patience (0 to disable).')
    parser.add_argument('--reduce_lr_patience', type=int, default=7,
                        help='Patience for reducing LR (0 to disable).')
    
    # Sauvegarde
    parser.add_argument('--save_model', type=str, default='best_cnn3d_pytorch.pth',
                        help='Path to save best model.')
    parser.add_argument('--save_history', type=str, default='training_history_pytorch.png',
                        help='Path to save training curves.')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Configuration du device (GPU/CPU)
    device = configure_device()
    
    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'mps':
        torch.mps.manual_seed(args.seed)
    elif device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Vérification du répertoire
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Le répertoire {args.data_dir} n'existe pas!")
    
    feature_files = [os.path.join(args.data_dir, f) 
                     for f in os.listdir(args.data_dir) 
                     if f.endswith('.npy')]
    
    if len(feature_files) == 0:
        raise ValueError(f"Aucun fichier .npy trouvé dans {args.data_dir}")
    
    print(f"{'='*70}")
    print(f"CONFIGURATION CNN 3D - PyTorch")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Fichiers trouvés: {len(feature_files)}")
    print(f"  Patch size: {args.patch_size}x{args.patch_size}")
    print(f"  Stride: {args.stride}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"{'='*70}\n")
    
    # Chargement des données
    X, y = load_hyperspectral_data(
        feature_files,
        patch_size=args.patch_size,
        max_patches_per_file=args.max_patches_per_file,
        max_total_samples=args.max_samples,
        stride=args.stride,
        seed=args.seed
    )
    
    num_classes = len(np.unique(y))
    input_channels = X.shape[-1]  # Nombre de bandes spectrales
    
    print(f"\nNombre de classes: {num_classes}")
    print(f"Nombre de bandes spectrales: {input_channels}")
    
    # Split train/val/test
    print("\nSplit des données...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=args.val_size/(1-args.test_size), 
        random_state=args.seed, stratify=y_train
    )
    
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Normalisation
    print("\nNormalisation des données...")
    scaler = StandardScaler()
    
    # Reshape pour normaliser par bande
    X_train_flat = X_train.reshape(-1, input_channels)
    X_val_flat = X_val.reshape(-1, input_channels)
    X_test_flat = X_test.reshape(-1, input_channels)
    
    X_train_norm = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_norm = scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test_norm = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    print(f"Stats train: mean={X_train_norm.mean():.4f}, std={X_train_norm.std():.4f}")
    
    # Création des datasets PyTorch
    train_dataset = HyperspectralDataset(X_train_norm, y_train)
    val_dataset = HyperspectralDataset(X_val_norm, y_val)
    test_dataset = HyperspectralDataset(X_test_norm, y_test)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=0)
    
    # Libération mémoire
    del X, y, X_train, X_val, X_test, X_train_flat, X_val_flat, X_test_flat
    gc.collect()
    
    # Construction du modèle
    print("\nConstruction du modèle...")
    # Si on tourne sur MPS, utiliser AvgPool3d pour éviter l'op non implémentée
    pool_type = 'avg' if device.type == 'mps' else 'max'
    model = CNN3D(input_channels, num_classes, args.architecture, args.patch_size, pool_type=pool_type)
    model = model.to(device)
    
    # Affichage du nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total paramètres: {total_params:,}")
    print(f"Paramètres entraînables: {trainable_params:,}")
    
    # Optimizer et loss
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, 
                             momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, 
                                 weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    if args.reduce_lr_patience > 0:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.reduce_lr_patience
        )
    
    # Entraînement
    print(f"\n{'='*70}")
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print(f"{'='*70}\n")
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        # Entraînement
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Stockage de l'historique
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        if args.reduce_lr_patience > 0:
            scheduler.step(val_loss)
        
        # Affichage
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if args.save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'architecture': args.architecture,
                    'num_classes': num_classes,
                    'input_channels': input_channels,
                    'patch_size': args.patch_size
                }, args.save_model)
                print(f"  ✓ Meilleur modèle sauvegardé (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"\nEarly stopping après {epoch} époques (patience: {args.early_stopping})")
            break
        
        print()
    
    # Chargement du meilleur modèle pour l'évaluation
    if args.save_model and os.path.exists(args.save_model):
        print(f"Chargement du meilleur modèle pour évaluation...")
        checkpoint = torch.load(args.save_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Meilleure val_acc: {checkpoint['val_acc']:.2f}% (epoch {checkpoint['epoch']})")
    
    # Évaluation sur test
    print(f"\n{'='*70}")
    print("ÉVALUATION SUR LE TEST SET")
    print(f"{'='*70}\n")
    
    y_true, y_pred = test_model(model, test_loader, device)
    
    # Métriques
    test_acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print("\nMatrice de confusion:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print("\nRapport de classification:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Courbes d'apprentissage
    if args.save_history:
        plot_training_history(train_losses, train_accs, val_losses, val_accs, args.save_history)
    
    print(f"\n{'='*70}")
    print("ENTRAÎNEMENT TERMINÉ!")
    print(f"{'='*70}")
    print(f"Meilleur modèle sauvegardé: {args.save_model}")
    print(f"Test Accuracy finale: {test_acc:.4f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()