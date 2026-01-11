"""
Script de test pour CNN 1D sur dataset_reflec_test

Utilise le modèle pré-entraîné pour évaluer les performances sur de nouvelles données.
"""

import argparse
import os
import numpy as np
from collections import Counter
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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
    required_cols = ["Nom_Fichier_npy", 'Yc', 'Xc', 'Classe']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing_cols}")
    
    centers_dict = {}
    
    for _, row in df.iterrows():
        filename = row["Nom_Fichier_npy"]
        yc = int(row['Yc'])
        xc = int(row['Xc'])
        class_label = int(row['Classe'])
        
        centers_dict[filename] = (yc, xc, class_label)
    
    print(f"✓ Centres chargés: {len(centers_dict)} fichiers")
    
    return centers_dict


def extract_pixels_around_center(cube, center_y, center_x, n_pixels=10000, 
                                 method='circular'):
    """
    Extrait des pixels autour du centre de l'objet.
    """
    h, w, bands = cube.shape
    
    if method == 'circular':
        radius = int(np.sqrt(n_pixels / np.pi))
        y_grid, x_grid = np.ogrid[:h, :w]
        distances = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
        mask_circle = distances <= radius
        positions = np.argwhere(mask_circle)
        
        while len(positions) < n_pixels and radius < max(h, w):
            radius += 5
            mask_circle = distances <= radius
            positions = np.argwhere(mask_circle)
        
        if len(positions) > n_pixels:
            indices = np.random.choice(len(positions), n_pixels, replace=False)
            positions = positions[indices]
    
    elif method == 'square':
        side = int(np.sqrt(n_pixels))
        half_side = side // 2
        
        y_start = max(0, center_y - half_side)
        y_end = min(h, center_y + half_side)
        x_start = max(0, center_x - half_side)
        x_end = min(w, center_x + half_side)
        
        y_coords, x_coords = np.meshgrid(range(y_start, y_end), 
                                         range(x_start, x_end), indexing='ij')
        positions = np.column_stack([y_coords.ravel(), x_coords.ravel()])
        
        if len(positions) > n_pixels:
            indices = np.random.choice(len(positions), n_pixels, replace=False)
            positions = positions[indices]
    
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    pixels = cube[positions[:, 0], positions[:, 1], :]
    
    return pixels, positions


def load_test_data(data_dir, csv_path, n_pixels_object=10000, seed=42):
    """
    Charge les données de test en utilisant les centres du CSV.
    
    Returns:
        X: Array (n_total_pixels, bands)
        y: Array (n_total_pixels,) avec classes des objets
        file_labels: Liste des tuples (filename, true_class, n_pixels)
    """
    np.random.seed(seed)
    
    centers_dict = load_centers_from_csv(csv_path)
    
    all_pixels = []
    all_labels = []
    file_labels = []
    
    print(f"\nChargement des données de test...")
    print(f"  Pixels par objet: {n_pixels_object}")
    
    # Lister les fichiers .npy disponibles
    if os.path.isdir(data_dir):
        available_files = {}
        for f in os.listdir(data_dir):
            if f.endswith('.npy'):
                filename = f
                file_path = os.path.join(data_dir, f)
                available_files[filename] = file_path
    else:
        raise ValueError(f"Le répertoire {data_dir} n'existe pas")
    
    print(f"  Fichiers .npy disponibles: {len(available_files)}")
    
    files_processed = 0
    
    # Traiter les fichiers qui ont des centres dans le CSV
    for filename, (center_y, center_x, class_label) in centers_dict.items():
        
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
            
            # Stockage
            all_pixels.append(object_pixels.astype(np.float32))
            all_labels.append(np.full(len(object_pixels), class_label, dtype=np.int32))
            file_labels.append((filename, class_label, len(object_pixels)))
            
            files_processed += 1
            
        except Exception as e:
            print(f"  Erreur {filename}: {e}")
            continue
    
    if len(all_pixels) == 0:
        raise ValueError("Aucune donnée de test chargée!")
    
    # Concaténation
    print("\nConcaténation...")
    X = np.vstack(all_pixels)
    y = np.concatenate(all_labels)
    
    print(f"\n✓ Données de test chargées:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Classes: {np.unique(y)}")
    print(f"  Distribution: {dict(Counter(y))}")
    print(f"  Fichiers traités: {files_processed}")
    
    return X, y, file_labels


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
    """CNN 1D pour classification de spectres hyperspectraux."""
    
    def __init__(self, input_channels, num_classes, architecture='medium'):
        super(CNN1D_Spectral, self).__init__()
        
        self.architecture = architecture
        
        if architecture == 'simple':
            self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
            self.bn1 = nn.BatchNorm1d(32)
            self.pool1 = nn.MaxPool1d(2)
            self.dropout1 = nn.Dropout(0.3)
            
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool2 = nn.MaxPool1d(2)
            self.dropout2 = nn.Dropout(0.4)
            
            fc_input_dim = 64 * (input_channels // 4)
            
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
            
            fc_input_dim = 256 * (input_channels // 8)
            
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
            
            fc_input_dim = 256 * (input_channels // 8)
            
            self.fc1 = nn.Linear(fc_input_dim, 1024)
            self.bn_fc1 = nn.BatchNorm1d(1024)
            self.dropout4 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(1024, 512)
            self.dropout5 = nn.Dropout(0.5)
            self.fc3 = nn.Linear(512, num_classes)
        
        else:
            raise ValueError(f"Architecture inconnue: {architecture}")
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
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


def test_model(model, test_loader, device):
    """Teste le modèle et retourne prédictions et probabilités."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            
            # Probabilités
            probs = F.softmax(output, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            # Prédictions
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())
    
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix_test.png'):
    """Affiche la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion - Test Set')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matrice de confusion sauvegardée: {save_path}")


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path='per_class_accuracy_test.png'):
    """Affiche l'accuracy par classe."""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), per_class_acc * 100)
    
    # Colorer les barres
    for i, bar in enumerate(bars):
        if per_class_acc[i] >= 0.9:
            bar.set_color('green')
        elif per_class_acc[i] >= 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.xlabel('Classe')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy par Classe - Test Set')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim([0, 100])
    plt.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(per_class_acc):
        plt.text(i, v * 100 + 2, f'{v*100:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Accuracy par classe sauvegardée: {save_path}")


def analyze_misclassifications(y_true, y_pred, file_labels, class_mapping, 
                               save_path='misclassifications_analysis.txt'):
    """Analyse détaillée des erreurs de classification."""
    
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ANALYSE DES ERREURS DE CLASSIFICATION\n")
        f.write("="*70 + "\n\n")
        
        # Calculer les erreurs par fichier
        current_idx = 0
        for filename, true_class, n_pixels in file_labels:
            end_idx = current_idx + n_pixels
            
            file_y_true = y_true[current_idx:end_idx]
            file_y_pred = y_pred[current_idx:end_idx]
            
            # Accuracy pour ce fichier
            file_acc = accuracy_score(file_y_true, file_y_pred)
            
            # Classe prédite majoritaire
            pred_counts = Counter(file_y_pred)
            pred_class = pred_counts.most_common(1)[0][0]
            pred_percentage = pred_counts[pred_class] / n_pixels * 100
            
            f.write(f"Fichier: {filename}\n")
            f.write(f"  Classe vraie: {class_mapping.get(true_class, f'Classe_{true_class}')}\n")
            f.write(f"  Classe prédite (majoritaire): {class_mapping.get(pred_class, f'Classe_{pred_class}')} ({pred_percentage:.1f}%)\n")
            f.write(f"  Accuracy: {file_acc:.4f} ({file_acc*100:.2f}%)\n")
            
            if pred_class != true_class:
                f.write(f"  ⚠️  ERREUR DE CLASSIFICATION!\n")
            
            f.write("\n")
            
            current_idx = end_idx
    
    print(f"Analyse des erreurs sauvegardée: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Test CNN 1D sur nouvelles données')
    
    # Données
    parser.add_argument('--test_data_dir', type=str, required=True,
                        help='Répertoire contenant les fichiers .npy de test')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Chemin vers hyperspectral_dataset_summary.csv')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Chemin vers le modèle sauvegardé (.pth)')
    
    # Paramètres d'extraction
    parser.add_argument('--n_pixels_object', type=int, default=10000,
                        help='Pixels autour du centre')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    
    # Sauvegarde
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Répertoire pour sauvegarder les résultats')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration device
    device = configure_device()
    
    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Vérifications
    if not os.path.exists(args.test_data_dir):
        raise ValueError(f"Répertoire inexistant: {args.test_data_dir}")
    
    if not os.path.exists(args.csv_path):
        raise ValueError(f"Fichier CSV inexistant: {args.csv_path}")
    
    if not os.path.exists(args.model_path):
        raise ValueError(f"Modèle inexistant: {args.model_path}")
    
    print(f"{'='*70}")
    print(f"TEST CNN 1D SUR NOUVELLES DONNÉES")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Répertoire de test: {args.test_data_dir}")
    print(f"  CSV: {args.csv_path}")
    print(f"  Modèle: {args.model_path}")
    print(f"  Pixels/objet: {args.n_pixels_object}")
    print(f"{'='*70}\n")
    
    # Chargement du modèle
    print("Chargement du modèle...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    architecture = checkpoint['architecture']
    num_classes = checkpoint['num_classes']
    input_channels = checkpoint['input_channels']
    class_mapping = checkpoint['class_mapping']
    
    print(f"  Architecture: {architecture}")
    print(f"  Classes: {num_classes}")
    print(f"  Bandes spectrales: {input_channels}")
    print(f"  Mapping: {class_mapping}")
    
    # Reconstruction du modèle
    model = CNN1D_Spectral(input_channels, num_classes, architecture)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("✓ Modèle chargé avec succès\n")
    
    # Chargement des données de test
    X_test, y_test, file_labels = load_test_data(
        args.test_data_dir,
        args.csv_path,
        n_pixels_object=args.n_pixels_object,
        seed=args.seed
    )
    
    # Normalisation (utiliser les mêmes stats que l'entraînement si disponibles)
    # Pour simplifier, on normalise sur les données de test
    print("\nNormalisation des données de test...")
    scaler = StandardScaler()
    X_test_norm = scaler.fit_transform(X_test)
    
    # Dataset et DataLoader
    test_dataset = SpectralDataset(X_test_norm, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    
    # Test
    print(f"\n{'='*70}")
    print("ÉVALUATION")
    print(f"{'='*70}\n")
    
    y_true, y_pred, y_probs = test_model(model, test_loader, device)
    
    # Métriques globales
    test_acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Matrice de confusion
    print("\nMatrice de confusion:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Rapport de classification
    print("\nRapport de classification:")
    class_names = [class_mapping[i] for i in sorted(class_mapping.keys())]
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # Sauvegarder les visualisations
    conf_matrix_path = os.path.join(args.output_dir, 'confusion_matrix_test.png')
    plot_confusion_matrix(y_true, y_pred, class_names, conf_matrix_path)
    
    per_class_path = os.path.join(args.output_dir, 'per_class_accuracy_test.png')
    plot_per_class_accuracy(y_true, y_pred, class_names, per_class_path)
    
    # Analyse des erreurs
    misclass_path = os.path.join(args.output_dir, 'misclassifications_analysis.txt')
    analyze_misclassifications(y_true, y_pred, file_labels, class_mapping, misclass_path)
    
    # Sauvegarder les résultats détaillés
    results_path = os.path.join(args.output_dir, 'test_results_summary.txt')
    with open(results_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("RÉSULTATS DU TEST\n")
        f.write("="*70 + "\n\n")
        f.write(f"Modèle: {args.model_path}\n")
        f.write(f"Dataset de test: {args.test_data_dir}\n")
        f.write(f"Nombre de fichiers testés: {len(file_labels)}\n")
        f.write(f"Nombre total de pixels: {len(y_true)}\n")
        f.write(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n\n")
        f.write("Matrice de confusion:\n")
        f.write(str(cm) + "\n\n")
        f.write("Rapport de classification:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    print(f"\n{'='*70}")
    print("TEST TERMINÉ!")
    print(f"{'='*70}")
    print(f"Résultats sauvegardés dans: {args.output_dir}")
    print(f"  - {conf_matrix_path}")
    print(f"  - {per_class_path}")
    print(f"  - {misclass_path}")
    print(f"  - {results_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()