"""
CNN 2D PyTorch pour les données GHOST (histogrammes)
Optimisé pour Apple Silicon (M1/M2/M3/M4) avec MPS
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def configure_device():
    """
    Configure PyTorch pour utiliser le GPU Apple Silicon (MPS).
    """
    print(f"\n{'='*70}")
    print("CONFIGURATION DEVICE (PyTorch)")
    print(f"{'='*70}")
    print(f"Version PyTorch: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n✅ GPU Apple Silicon (MPS) détecté!")
        print(f"   Device: {device}")
        print(f"   Backend: Metal Performance Shaders")
        
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


def load_ghost_data(data_dir, max_samples_per_file=500, max_total_samples=50000, seed=42):
    """
    Charge les données GHOST (déjà en 2D par échantillon).
    
    Chaque fichier .npy doit contenir soit:
      - une matrice 2D (n_samples, n_features) qui représente des vecteurs 1D -> NON attendu ici
      - ou des images 2D par échantillon : (n_samples, height, width)
      - ou (n_samples, channels, height, width)
    
    Cette fonction renvoie X, y sans reshape supplémentaire.
    """
    np.random.seed(seed)
    
    all_samples = []
    all_labels = []
    total_samples = 0
    
    print(f"Chargement des données GHOST...")
    
    for idx, file_name in enumerate(data_dir):
        if total_samples >= max_total_samples:
            print(f"Limite de {max_total_samples} échantillons atteinte.")
            break
        
        try:
            data = np.load(file_name)
            
            # On accepte (n_samples, H, W) ou (n_samples, C, H, W) ou (n_features,) (cas rare)
            if data.ndim == 3 or data.ndim == 4:
                samples = data
            elif data.ndim == 2:
                # Si on reçoit (n_samples, n_features) — laisser l'utilisateur gérer ; ici on prend tel quel
                samples = data
            elif data.ndim == 1:
                samples = data[np.newaxis, :]
            else:
                print(f"Format inattendu pour {file_name}: {data.shape}. Ignoré.")
                continue
            
            # Extraction du label
            class_label = extract_class_from_filename(file_name)
            
            if class_label < 1:
                print(f"Label invalide: {file_name}")
                continue
            
            all_samples.append(samples.astype(np.float32))
            all_labels.append(np.full(len(samples), class_label - 1, dtype=np.int64))
            
            total_samples += len(samples)
            
            if (idx + 1) % 10 == 0:
                print(f"Traité {idx + 1}/{len(data_dir)} fichiers, {total_samples} échantillons")
        
        except Exception as e:
            print(f"Erreur: {file_name}: {e}")
            continue
    
    if len(all_samples) == 0:
        raise ValueError("Aucune donnée chargée!")
    
    print("Concaténation des échantillons...")
    X = np.vstack(all_samples)
    y = np.concatenate(all_labels)
    
    del all_samples, all_labels
    gc.collect()
    
    print(f"\nDonnées chargées:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Classes: {np.unique(y)}")
    print(f"  Distribution: {dict(Counter(y))}")
    
    return X, y


class GHOSTDataset(Dataset):
    """Dataset PyTorch pour les données GHOST."""
    
    def __init__(self, X, y, transform=None):
        # X doit être au format (N, C, H, W) ou (N, H, W) — on convertit en (N, C, H, W)
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        self.X = torch.FloatTensor(X)
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


class SafeMaxPool2d(nn.Module):
    """MaxPool2d qui ne s'applique que si la taille spatiale >= kernel_size."""
    def __init__(self, kernel_size=2, stride=None, padding=0, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self._pool = nn.MaxPool2d(kernel_size, self.stride, padding, ceil_mode=ceil_mode)

    def forward(self, x):
        _, _, h, w = x.size()
        if isinstance(self.kernel_size, int):
            k_h = k_w = self.kernel_size
        else:
            k_h, k_w = self.kernel_size
        # Si une dimension est trop petite pour le kernel, on retourne l'input tel quel
        if h < k_h or w < k_w:
            return x
        return self._pool(x)

class CNN2D_GHOST(nn.Module):
    """
    Réseau CNN 2D pour classification des histogrammes GHOST.
    
    Architecture adaptée pour traiter des "images" 2D créées à partir
    des histogrammes concaténés ou données 2D natives.
    """
    
    def __init__(self, input_shape, num_classes, architecture='simple'):
        """
        Args:
            input_shape: (channels, height, width) ex: (1, 16, 16)
            num_classes: Nombre de classes
            architecture: 'simple', 'medium', ou 'deep'
        """
        super(CNN2D_GHOST, self).__init__()
        
        self.architecture = architecture
        channels, height, width = input_shape
        
        if architecture == 'simple':
            # Architecture simple (rapide)
            self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool1 = SafeMaxPool2d(2, 2)
            self.dropout1 = nn.Dropout2d(0.25)
            
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool2 = SafeMaxPool2d(2, 2)
            self.dropout2 = nn.Dropout2d(0.3)
            
            # Calcul de la dimension après convolutions
            # 2 poolings de 2 : dimension divisée par 4
            fc_height = max(1, height // 4)
            fc_width = max(1, width // 4)
            fc_input_dim = 64 * fc_height * fc_width
            
            self.fc1 = nn.Linear(fc_input_dim, 128)
            self.dropout3 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, num_classes)
        
        elif architecture == 'medium':
            # Architecture moyenne
            self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool1 = SafeMaxPool2d(2, 2)
            
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool2 = SafeMaxPool2d(2, 2)
            self.dropout1 = nn.Dropout2d(0.3)
            
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool3 = SafeMaxPool2d(2, 2)
            self.dropout2 = nn.Dropout2d(0.4)
            
            fc_height = max(1, height // 8)
            fc_width = max(1, width // 8)
            fc_input_dim = 128 * fc_height * fc_width
            
            self.fc1 = nn.Linear(fc_input_dim, 256)
            self.dropout3 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, 128)
            self.dropout4 = nn.Dropout(0.5)
            self.fc3 = nn.Linear(128, num_classes)
        
        elif architecture == 'deep':
            # Architecture profonde
            self.conv1a = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
            self.bn1a = nn.BatchNorm2d(32)
            self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.bn1b = nn.BatchNorm2d(32)
            self.pool1 = SafeMaxPool2d(2, 2)
            self.dropout1 = nn.Dropout2d(0.2)
            
            self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2a = nn.BatchNorm2d(64)
            self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn2b = nn.BatchNorm2d(64)
            self.pool2 = SafeMaxPool2d(2, 2)
            self.dropout2 = nn.Dropout2d(0.3)
            
            self.conv3a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3a = nn.BatchNorm2d(128)
            self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn3b = nn.BatchNorm2d(128)
            self.pool3 = SafeMaxPool2d(2, 2)
            self.dropout3 = nn.Dropout2d(0.4)
            
            fc_height = max(1, height // 8)
            fc_width = max(1, width // 8)
            fc_input_dim = 128 * fc_height * fc_width
            
            self.fc1 = nn.Linear(fc_input_dim, 512)
            self.bn_fc1 = nn.BatchNorm1d(512)
            self.dropout4 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, 256)
            self.dropout5 = nn.Dropout(0.5)
            self.fc3 = nn.Linear(256, num_classes)
        
        else:
            raise ValueError(f"Architecture inconnue: {architecture}")
    
    def forward(self, x):
        if self.architecture == 'simple':
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            x = self.dropout1(x)
            
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            x = self.dropout2(x)
            
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
            x = self.pool3(x)
            x = self.dropout2(x)
            
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
            
            x = F.relu(self.bn3a(self.conv3a(x)))
            x = F.relu(self.bn3b(self.conv3b(x)))
            x = self.pool3(x)
            x = self.dropout3(x)
            
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
    
    axes[0].plot(epochs, train_losses, 'b-', label='Train')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
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
    parser = argparse.ArgumentParser(description='Train 2D CNN (PyTorch) on GHOST data.')
    
    # Données
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing .npy files.')
    parser.add_argument('--max_samples_per_file', type=int, default=500,
                        help='Max samples per file.')
    parser.add_argument('--max_samples', type=int, default=50000,
                        help='Max total samples.')
    parser.add_argument('--target_height', type=int, default=128,
                        help='(Ignored) target height for reshaping to 2D — keep for backward compatibility.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set proportion.')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set proportion.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    
    # SMOTE
    parser.add_argument('--use_smote', action='store_true',
                        help='Apply SMOTE to balance classes.')
    parser.add_argument('--smote_k_neighbors', type=int, default=5,
                        help='Number of neighbors for SMOTE.')
    
    # Architecture
    parser.add_argument('--architecture', type=str, default='medium',
                        choices=['simple', 'medium', 'deep'],
                        help='Model architecture.')
    
    # Entraînement
    parser.add_argument('--batch_size', type=int, default=64,
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
    parser.add_argument('--early_stopping', type=int, default=0,
                        help='Early stopping patience (0 to disable).')
    parser.add_argument('--reduce_lr_patience', type=int, default=7,
                        help='Patience for reducing LR (0 to disable).')
    
    # Sauvegarde
    parser.add_argument('--save_model', type=str, default='best_cnn2d_ghost.pth',
                        help='Path to save best model.')
    parser.add_argument('--save_history', type=str, default='training_history_ghost.png',
                        help='Path to save training curves.')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Configuration du device
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
    print(f"CONFIGURATION CNN 2D - PyTorch (GHOST Data)")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Fichiers trouvés: {len(feature_files)}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  SMOTE: {'Activé' if args.use_smote else 'Désactivé'}")
    print(f"{'='*70}\n")
    
    # Chargement des données (déjà 2D par échantillon)
    X, y = load_ghost_data(
        feature_files,
        max_samples_per_file=args.max_samples_per_file,
        max_total_samples=args.max_samples,
        seed=args.seed
    )
    
    num_classes = len(np.unique(y))
    print(f"\nNombre de classes: {num_classes}")
    
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
    print(f"  Distribution train: {dict(Counter(y_train))}")
    
    # Normalisation (appliquée per-feature si les échantillons sont vectoriels,
    # ou flatten -> scaler -> reshape si nécessaire)
    print("\nNormalisation des données...")
    scaler = StandardScaler()
    
    # Si les échantillons sont des images 2D (N, H, W) ou (N, C, H, W), on aplatie par échantillon pour scaler per-feature
    def normalize_images(X_array):
        if X_array.ndim == 4:
            N, C, H, W = X_array.shape
            flat = X_array.reshape(N, C * H * W)
            flat_norm = scaler.fit_transform(flat) if not hasattr(scaler, "mean_") else scaler.transform(flat)
            return flat_norm.reshape(N, C, H, W)
        elif X_array.ndim == 3:
            N, H, W = X_array.shape
            flat = X_array.reshape(N, H * W)
            flat_norm = scaler.fit_transform(flat) if not hasattr(scaler, "mean_") else scaler.transform(flat)
            return flat_norm.reshape(N, H, W)
        else:
            # fallback for vector data
            return scaler.fit_transform(X_array) if not hasattr(scaler, "mean_") else scaler.transform(X_array)
    
    # Fit scaler on train (handle both vector and image cases)
    if X_train.ndim == 4:
        N, C, H, W = X_train.shape
        flat_train = X_train.reshape(N, C * H * W)
        scaler.fit(flat_train)
        X_train_norm = flat_train.reshape(N, C, H, W)
        X_train_norm = scaler.transform(flat_train).reshape(N, C, H, W)
        
        # transform val/test
        Nval = X_val.shape[0]
        X_val_norm = scaler.transform(X_val.reshape(Nval, C * H * W)).reshape(Nval, C, H, W)
        Ntest = X_test.shape[0]
        X_test_norm = scaler.transform(X_test.reshape(Ntest, C * H * W)).reshape(Ntest, C, H, W)
    elif X_train.ndim == 3:
        N, H, W = X_train.shape
        flat_train = X_train.reshape(N, H * W)
        scaler.fit(flat_train)
        X_train_norm = scaler.transform(flat_train).reshape(N, H, W)
        
        Nval = X_val.shape[0]
        X_val_norm = scaler.transform(X_val.reshape(Nval, H * W)).reshape(Nval, H, W)
        Ntest = X_test.shape[0]
        X_test_norm = scaler.transform(X_test.reshape(Ntest, H * W)).reshape(Ntest, H, W)
    else:
        # vector case
        X_train_norm = scaler.fit_transform(X_train)
        X_val_norm = scaler.transform(X_val)
        X_test_norm = scaler.transform(X_test)
    
    print(f"Stats train: mean={X_train_norm.mean():.4f}, std={X_train_norm.std():.4f}")
    
    # Application de SMOTE si demandé (SMOTE attend des vecteurs 2D)
    if args.use_smote:
        print(f"\n{'='*70}")
        print("APPLICATION DE SMOTE")
        print(f"{'='*70}")
        print(f"Distribution AVANT SMOTE: {dict(Counter(y_train))}")
        
        # SMOTE sur représentation vectorielle
        if X_train_norm.ndim == 4:
            N, C, H, W = X_train_norm.shape
            flat = X_train_norm.reshape(N, C * H * W)
            smote = SMOTE(k_neighbors=args.smote_k_neighbors, random_state=args.seed)
            flat_res, y_train = smote.fit_resample(flat, y_train)
            X_train_norm = flat_res.reshape(len(y_train), C, H, W)
        elif X_train_norm.ndim == 3:
            N, H, W = X_train_norm.shape
            flat = X_train_norm.reshape(N, H * W)
            smote = SMOTE(k_neighbors=args.smote_k_neighbors, random_state=args.seed)
            flat_res, y_train = smote.fit_resample(flat, y_train)
            X_train_norm = flat_res.reshape(len(y_train), H, W)
        else:
            smote = SMOTE(k_neighbors=args.smote_k_neighbors, random_state=args.seed)
            X_train_norm, y_train = smote.fit_resample(X_train_norm, y_train)
        
        print(f"Distribution APRÈS SMOTE: {dict(Counter(y_train))}")
        print(f"Total échantillons: {len(y_train)}")
        print(f"{'='*70}\n")
    
    # Les données sont déjà 2D ; s'assurer du format (N, C, H, W)
    if X_train_norm.ndim == 3:
        X_train_2d = X_train_norm[:, np.newaxis, :, :]
        X_val_2d = X_val_norm[:, np.newaxis, :, :]
        X_test_2d = X_test_norm[:, np.newaxis, :, :]
    elif X_train_norm.ndim == 4:
        X_train_2d = X_train_norm
        X_val_2d = X_val_norm
        X_test_2d = X_test_norm
    else:
        # cas vecteur (N, features) — essayer de convertir en (N, 1, H, W) si carré
        if X_train_norm.ndim == 2:
            n_feat = X_train_norm.shape[1]
            side = int(np.sqrt(n_feat))
            if side * side == n_feat:
                X_train_2d = X_train_norm.reshape(-1, 1, side, side)
                X_val_2d = X_val_norm.reshape(-1, 1, side, side)
                X_test_2d = X_test_norm.reshape(-1, 1, side, side)
            else:
                raise ValueError("Données vectorielles: impossible de reshape automatique en 2D — fournissez des images 2D.")
        else:
            raise ValueError("Format de données non supporté. Attendu (N,H,W) ou (N,C,H,W).")
    
    input_shape = X_train_2d.shape[1:]  # (C, H, W)
    
    # Création des datasets PyTorch
    train_dataset = GHOSTDataset(X_train_2d, y_train)
    val_dataset = GHOSTDataset(X_val_2d, y_val)
    test_dataset = GHOSTDataset(X_test_2d, y_test)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=0)
    
    # Libération mémoire
    del X, y, X_train, X_val, X_test, X_train_norm, X_val_norm, X_test_norm
    del X_train_2d, X_val_2d, X_test_2d
    gc.collect()
    
    # Construction du modèle
    print("\nConstruction du modèle...")
    model = CNN2D_GHOST(input_shape, num_classes, args.architecture)
    model = model.to(device)
    
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
        # Remove verbose kwarg for compatibility
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
                    'input_shape': input_shape
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