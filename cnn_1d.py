"""
CNN 1D pour classification hyperspectrale avec centres depuis CSV

Processus :
1. Lit les coordonnées des centres depuis hyperspectral_dataset_summary.csv
2. Extrait 10000 pixels autour du centre pour chaque fichier
3. Entraîne un CNN 1D sur les spectres individuels
"""

import argparse
import os
import numpy as np
from collections import Counter
import gc
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


def configure_device():
    """Configure PyTorch pour utiliser le GPU (MPS ou CUDA)."""
    print(f"\n{'='*70}")
    print("CONFIGURATION DEVICE (PyTorch)")
    print(f"{'='*70}")
    print(f"Version PyTorch: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n✅ GPU Apple Silicon (MPS) détecté!")
        print(f"   Device: {device}")
        
        try:
            test_tensor = torch.randn(10, 10).to(device)
            result = torch.matmul(test_tensor, test_tensor)
            print(f"   ✓ Test GPU réussi")
        except Exception as e:
            print(f"   ⚠️  Erreur: {e}, fallback CPU")
            device = torch.device("cpu")
    
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n✅ GPU CUDA détecté!")
        print(f"   Device: {device}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    else:
        device = torch.device("cpu")
        print(f"\n⚠️  Aucun GPU, utilisation CPU")
    
    print(f"{'='*70}\n")
    return device


def load_centers_from_csv(csv_path):
    """
    Charge les coordonnées des centres depuis le fichier CSV.
    
    Args:
        csv_path: Chemin vers hyperspectral_dataset_summary.csv
    
    Returns:
        centers_dict: Dictionnaire {filename: (Yc, Xc, class_label)}
    """
    print(f"Chargement des centres depuis: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier CSV introuvable: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Vérification des colonnes requises
    required_cols = ['Nom_Fichier_npy', 'Yc', 'Xc', 'Classe']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing_cols}")
    
    centers_dict = {}
    
    for _, row in df.iterrows():
        filename = row['Nom_Fichier_npy']
        yc = int(row['Yc'])
        xc = int(row['Xc'])
        class_label = int(row['Classe'])
        
        centers_dict[filename] = (yc, xc, class_label)
    
    print(f"✓ Centres chargés: {len(centers_dict)} fichiers")
    print(f"  Exemple: {list(centers_dict.items())[0]}")
    
    return centers_dict


def extract_pixels_around_center(cube, center_y, center_x, n_pixels=10000, 
                                 method='circular'):
    """
    Extrait des pixels autour du centre de l'objet.
    
    Args:
        cube: Cube 3D (H, W, Bands)
        center_y, center_x: Centre de l'objet
        n_pixels: Nombre de pixels à extraire
        method: 'circular' (cercle concentrique) ou 'square' (carré)
    
    Returns:
        pixels: Array (n_pixels, bands)
        positions: Liste de positions (y, x)
    """
    h, w, bands = cube.shape
    
    if method == 'circular':
        # Extraire pixels dans un cercle centré
        # Calculer le rayon nécessaire pour avoir n_pixels
        radius = int(np.sqrt(n_pixels / np.pi))
        
        # Générer grille de distances au centre
        y_grid, x_grid = np.ogrid[:h, :w]
        distances = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
        
        # Trouver pixels dans le cercle
        mask_circle = distances <= radius
        
        # Extraire positions
        positions = np.argwhere(mask_circle)
        
        # Si pas assez de pixels, augmenter le rayon
        while len(positions) < n_pixels and radius < max(h, w):
            radius += 5
            mask_circle = distances <= radius
            positions = np.argwhere(mask_circle)
        
        # Échantillonnage aléatoire si trop de pixels
        if len(positions) > n_pixels:
            indices = np.random.choice(len(positions), n_pixels, replace=False)
            positions = positions[indices]
    
    elif method == 'square':
        # Extraire pixels dans un carré centré
        side = int(np.sqrt(n_pixels))
        half_side = side // 2
        
        y_start = max(0, center_y - half_side)
        y_end = min(h, center_y + half_side)
        x_start = max(0, center_x - half_side)
        x_end = min(w, center_x + half_side)
        
        # Positions dans le carré
        y_coords, x_coords = np.meshgrid(range(y_start, y_end), 
                                         range(x_start, x_end), indexing='ij')
        positions = np.column_stack([y_coords.ravel(), x_coords.ravel()])
        
        # Échantillonnage si nécessaire
        if len(positions) > n_pixels:
            indices = np.random.choice(len(positions), n_pixels, replace=False)
            positions = positions[indices]
    
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    # Extraction des spectres
    pixels = cube[positions[:, 0], positions[:, 1], :]
    
    return pixels, positions


def visualize_sampling(cube, center_y, center_x, object_positions, class_label,
                       save_path='sampling_visualization.png'):
    """
    Visualise l'échantillonnage des pixels objet.
    """
    h, w, bands = cube.shape
    
    # Créer une image RGB pour visualisation
    if bands >= 3:
        # Prendre 3 bandes pour RGB
        rgb_image = cube[:, :, [bands//4, bands//2, 3*bands//4]]
    else:
        # Grayscale
        rgb_image = np.repeat(cube[:, :, 0:1], 3, axis=2)
    
    # Normalisation
    rgb_image = ((rgb_image - rgb_image.min()) / 
                 (rgb_image.max() - rgb_image.min() + 1e-8) * 255).astype(np.uint8)
    
    # Marquer les pixels sélectionnés
    overlay = rgb_image.copy()
    
    # Pixels objet en vert
    for pos in object_positions[::10]:  # Sous-échantillonner pour visibilité
        y, x = pos
        cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1)
    
    # Centre en bleu
    cv2.circle(overlay, (center_x, center_y), 5, (0, 0, 255), -1)
    
    # Blend
    result = cv2.addWeighted(rgb_image, 0.7, overlay, 0.3, 0)
    
    # Sauvegarder
    plt.figure(figsize=(10, 8))
    plt.imshow(result)
    plt.title(f'Échantillonnage Classe {class_label}\n'
              f'Centre: ({center_y}, {center_x}), Pixels: {len(object_positions)}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualisation sauvegardée: {save_path}")


def load_hyperspectral_from_csv(data_dir, csv_path, n_pixels_object=10000, 
                                max_files=120, seed=42):
    """
    Charge les données hyperspectrales en utilisant les centres du CSV.
    
    Returns:
        X: Array (n_total_pixels, bands)
        y: Array (n_total_pixels,) avec classes des objets
        class_mapping: Dictionnaire {class_id: class_name}
    """
    np.random.seed(seed)
    
    # Charger les centres depuis le CSV
    centers_dict = load_centers_from_csv(csv_path)
    
    all_pixels = []
    all_labels = []
    
    print(f"\nChargement avec centres CSV...")
    print(f"  Pixels par objet: {n_pixels_object}")
    
    files_processed = 0
    
    # Lister les fichiers .npy disponibles
    available_files = {}
    for file_path in data_dir:
        filename = os.path.basename(file_path)
        available_files[filename] = file_path
    
    print(f"  Fichiers .npy disponibles: {len(available_files)}")
    
    # Traiter les fichiers qui ont des centres dans le CSV
    for filename, (center_y, center_x, class_label) in centers_dict.items():
        if files_processed >= max_files:
            break
        
        if filename not in available_files:
            print(f"  ⚠️  Fichier non trouvé: {filename}")
            continue
        
        file_path = available_files[filename]
        
        try:
            cube = np.load(file_path)
            
            if cube.ndim != 3:
                print(f"  Ignoré (pas 3D): {filename}")
                continue
            
            h, w, bands = cube.shape
            
            print(f"\n  Fichier {files_processed + 1}: {filename}")
            print(f"    Shape: {cube.shape}, Classe: {class_label}")
            print(f"    Centre CSV: ({center_y}, {center_x})")
            
            # Vérifier que le centre est dans les limites
            if not (0 <= center_y < h and 0 <= center_x < w):
                print(f"    ⚠️  Centre hors limites, ajustement...")
                center_y = min(max(0, center_y), h - 1)
                center_x = min(max(0, center_x), w - 1)
                print(f"    Centre ajusté: ({center_y}, {center_x})")
            
            # Extraire pixels autour du centre
            object_pixels, object_positions = extract_pixels_around_center(
                cube, center_y, center_x, n_pixels_object, method='circular'
            )
            print(f"    Pixels extraits: {len(object_pixels)}")
            
            # Visualisation (première image seulement)
            if files_processed == 0:
                visualize_sampling(cube, center_y, center_x, 
                                 object_positions, class_label,
                                 f'sampling_class_{class_label}.png')
            
            # Stockage
            all_pixels.append(object_pixels.astype(np.float32))
            all_labels.append(np.full(len(object_pixels), class_label, dtype=np.int32))
            
            files_processed += 1
            
        except Exception as e:
            print(f"  Erreur {filename}: {e}")
            continue
    
    if len(all_pixels) == 0:
        raise ValueError("Aucune donnée chargée!")
    
    # Concaténation
    print("\nConcaténation...")
    X = np.vstack(all_pixels)
    y = np.concatenate(all_labels)
    
    del all_pixels, all_labels
    gc.collect()
    
    # Mapping des classes
    unique_classes = np.unique(y)
    class_mapping = {int(c): f'Classe_{c}' for c in unique_classes}
    
    print(f"\n✓ Données chargées:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Classes: {unique_classes}")
    print(f"  Distribution: {dict(Counter(y))}")
    print(f"  Mapping: {class_mapping}")
    
    return X, y, class_mapping


class SpectralDataset(Dataset):
    """Dataset PyTorch pour spectres 1D."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN1D_Spectral(nn.Module):
    """
    CNN 1D pour classification de spectres hyperspectraux.
    """
    
    def __init__(self, input_channels, num_classes, architecture='medium'):
        super(CNN1D_Spectral, self).__init__()
        
        self.architecture = architecture
        
        # Validate that the sequence length after repeated pooling remains >= 1
        pool_counts = {'simple': 2, 'medium': 3, 'deep': 3}
        n_pools = pool_counts.get(architecture, 3)
        downsample = 2 ** n_pools
        reduced_len = input_channels // downsample
        if reduced_len == 0:
            raise ValueError(
                f"input_channels={input_channels} too small for architecture '{architecture}'. "
                f"Need at least {downsample} spectral bands (2^{n_pools}). "
                f"Use a simpler architecture (e.g. --architecture simple) or increase the number of bands."
            )
        
        if architecture == 'simple':
            self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
            self.bn1 = nn.BatchNorm1d(32)
            self.pool1 = nn.MaxPool1d(2)
            self.dropout1 = nn.Dropout(0.3)
            
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool2 = nn.MaxPool1d(2)
            self.dropout2 = nn.Dropout(0.4)
            
            fc_input_dim = 64 * reduced_len
            
            self.fc1 = nn.Linear(fc_input_dim, 128)
            self.dropout3 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, num_classes)
        
        elif architecture == 'medium':
            self.conv1 = nn.Conv1d(1, 64, kernel_size=9, padding=4)
            self.bn1 = nn.BatchNorm1d(64)
            self.pool1 = nn.MaxPool1d(2)
            
            self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
            self.bn2 = nn.BatchNorm1d(128)
            self.pool2 = nn.MaxPool1d(2)
            self.dropout1 = nn.Dropout(0.3)
            
            self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
            self.bn3 = nn.BatchNorm1d(256)
            self.pool3 = nn.MaxPool1d(2)
            self.dropout2 = nn.Dropout(0.4)
            
            fc_input_dim = 256 * reduced_len
            
            self.fc1 = nn.Linear(fc_input_dim, 512)
            self.dropout3 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, 256)
            self.dropout4 = nn.Dropout(0.5)
            self.fc3 = nn.Linear(256, num_classes)
        
        elif architecture == 'deep':
            self.conv1a = nn.Conv1d(1, 64, kernel_size=11, padding=5)
            self.bn1a = nn.BatchNorm1d(64)
            self.conv1b = nn.Conv1d(64, 64, kernel_size=9, padding=4)
            self.bn1b = nn.BatchNorm1d(64)
            self.pool1 = nn.MaxPool1d(2)
            self.dropout1 = nn.Dropout(0.2)
            
            self.conv2a = nn.Conv1d(64, 128, kernel_size=7, padding=3)
            self.bn2a = nn.BatchNorm1d(128)
            self.conv2b = nn.Conv1d(128, 128, kernel_size=7, padding=3)
            self.bn2b = nn.BatchNorm1d(128)
            self.pool2 = nn.MaxPool1d(2)
            self.dropout2 = nn.Dropout(0.3)
            
            self.conv3a = nn.Conv1d(128, 256, kernel_size=5, padding=2)
            self.bn3a = nn.BatchNorm1d(256)
            self.conv3b = nn.Conv1d(256, 256, kernel_size=5, padding=2)
            self.bn3b = nn.BatchNorm1d(256)
            self.pool3 = nn.MaxPool1d(2)
            self.dropout3 = nn.Dropout(0.4)
            
            fc_input_dim = 256 * reduced_len
            
            self.fc1 = nn.Linear(fc_input_dim, 1024)
            self.bn_fc1 = nn.BatchNorm1d(1024)
            self.dropout4 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(1024, 512)
            self.dropout5 = nn.Dropout(0.5)
            self.fc3 = nn.Linear(512, num_classes)
        
        else:
            raise ValueError(f"Architecture inconnue: {architecture}")
    
    def forward(self, x):
        # Input: (N, bands) → (N, 1, bands) pour Conv1D
        x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected input tensor of shape (N, bands) so after unsqueeze got 3D, but got shape {x.shape}")
        
        if self.architecture == 'simple':
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            x = self.dropout1(x)
            
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            x = self.dropout2(x)
            
            x = x.view(x.size(0), -1)
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
    
    for data, target in train_loader:
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
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Évalue le modèle."""
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
    
    return running_loss / len(val_loader), 100. * correct / total


def test_model(model, test_loader, device):
    """Teste le modèle."""
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
        print(f"Courbes: {save_path}")
    
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='CNN 1D avec centres depuis CSV')
    
    # Données
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Répertoire contenant les fichiers .npy')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Chemin vers hyperspectral_dataset_summary.csv')
    parser.add_argument('--n_pixels_object', type=int, default=10000,
                        help='Pixels autour du centre (objet)')
    parser.add_argument('--max_files', type=int, default=109)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    
    # Architecture
    parser.add_argument('--architecture', type=str, default='medium',
                        choices=['simple', 'medium', 'deep'])
    
    # Entraînement
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Callbacks
    parser.add_argument('--early_stopping', type=int, default=15)
    parser.add_argument('--reduce_lr_patience', type=int, default=7)
    
    # Sauvegarde
    parser.add_argument('--save_model', type=str, default='best_cnn1d_gh_csv.pth')
    parser.add_argument('--save_history', type=str, default='history_cnn1d_gh_csv.png')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Configuration device
    device = configure_device()
    
    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'mps':
        torch.mps.manual_seed(args.seed)
    elif device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Vérification
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Répertoire inexistant: {args.data_dir}")
    
    if not os.path.exists(args.csv_path):
        raise ValueError(f"Fichier CSV inexistant: {args.csv_path}")
    
    feature_files = [os.path.join(args.data_dir, f) 
                     for f in os.listdir(args.data_dir) 
                     if f.endswith('.npy')]
    
    if len(feature_files) == 0:
        raise ValueError(f"Aucun .npy dans {args.data_dir}")
    
    print(f"{'='*70}")
    print(f"CNN 1D AVEC CENTRES DEPUIS CSV")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Fichiers .npy: {len(feature_files)}")
    print(f"  CSV: {args.csv_path}")
    print(f"  Pixels objet/fichier: {args.n_pixels_object}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"{'='*70}\n")
    
    # Chargement avec centres CSV
    X, y, class_mapping = load_hyperspectral_from_csv(
        feature_files,
        args.csv_path,
        n_pixels_object=args.n_pixels_object,
        max_files=args.max_files,
        seed=args.seed
    )
    
    num_classes = len(np.unique(y))
    input_channels = X.shape[1]
    
    print(f"\nNombre de classes: {num_classes}")
    print(f"Nombre de bandes spectrales: {input_channels}")
    
    # Split
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
    print(f"  Distribution val: {dict(Counter(y_val))}")
    print(f"  Distribution test: {dict(Counter(y_test))}")
    
    # Normalisation
    print("\nNormalisation...")
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    print(f"Stats train: mean={X_train_norm.mean():.4f}, std={X_train_norm.std():.4f}")
    
    # Datasets PyTorch
    train_dataset = SpectralDataset(X_train_norm, y_train)
    val_dataset = SpectralDataset(X_val_norm, y_val)
    test_dataset = SpectralDataset(X_test_norm, y_test)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    
    # Libération mémoire
    del X, y, X_train, X_val, X_test, X_train_norm, X_val_norm, X_test_norm
    gc.collect()
    
    # Construction du modèle
    print("\nConstruction du modèle...")
    model = CNN1D_Spectral(input_channels, num_classes, args.architecture)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total paramètres: {total_params:,}")
    print(f"Paramètres entraînables: {trainable_params:,}")
    
    # Optimizer et loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                          weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Scheduler
    scheduler = None
    if args.reduce_lr_patience > 0:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.reduce_lr_patience
        )
        print(f"Learning rate scheduler activé (ReduceLROnPlateau)")
        print(f"  Mode: min, Factor: 0.5, Patience: {args.reduce_lr_patience}")
    
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
        
        # Stockage
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Scheduler
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"  ⚡ Learning rate réduit: {old_lr:.2e} → {new_lr:.2e}")
        
        # Affichage
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Sauvegarde meilleur modèle
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
                    'class_mapping': class_mapping
                }, args.save_model)
                print(f"  ✓ Meilleur modèle sauvegardé (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"\nEarly stopping après {epoch} époques")
            break
        
        print()
    
    # Chargement meilleur modèle
    if args.save_model and os.path.exists(args.save_model):
        print(f"Chargement du meilleur modèle...")
        checkpoint = torch.load(args.save_model, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Meilleure val_acc: {checkpoint['val_acc']:.2f}% (epoch {checkpoint['epoch']})")
    
    # Évaluation
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
    target_names = [class_mapping[i] for i in sorted(class_mapping.keys())]
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    # Courbes
    if args.save_history:
        plot_training_history(train_losses, train_accs, val_losses, val_accs, args.save_history)
    
    print(f"\n{'='*70}")
    print("ENTRAÎNEMENT TERMINÉ!")
    print(f"{'='*70}")
    print(f"Meilleur modèle: {args.save_model}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Classes: {class_mapping}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()