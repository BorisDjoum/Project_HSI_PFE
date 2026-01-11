# Projet de Classification d'Images Hyperspectrales avec Deep Learning

## 📋 Description du Projet

Ce projet vise à classifier des objets à partir d'images hyperspectrales en utilisant différentes approches de deep learning. Les images hyperspectrales contiennent des informations spectrales riches (150-200 bandes) qui permettent une discrimination fine des matériaux.

### Objectifs
- Extraire et analyser des images hyperspectrales (cubes 3D)
- Détecter automatiquement les centres des objets d'intérêt
- Entraîner plusieurs architectures de réseaux de neurones pour la classification
- Évaluer les performances sur des données de test

### Datasets
- **Dataset d'entraînement** : `dataset_reflec` (50 fichiers .npy)
- **Dataset de test** : `dataset_reflec_test`
- **Fichier de métadonnées** : `hyperspectral_dataset_summary.csv` (contient les coordonnées des centres Xc, Yc et les classes)

---

## 🗂️ Structure du Projet

```
projet_hyperspectral/
│
├── dataset_reflec/              # Données d'entraînement (.npy)
├── dataset_reflec_test/         # Données de test (.npy)
├── hyperspectral_dataset_summary.csv  # Métadonnées (centres, classes)
│
├── 1_analyze_dataset.py         # Analyse et extraction des métadonnées
├── 2_cnn_2d_patch.py           # CNN 2D sur patches spatiaux-spectraux
├── 3_cnn_3d.py                 # CNN 3D sur cubes 3D
├── 4_cnn_1d_spectral.py        # CNN 1D sur spectres individuels (centres CSV)
├── 5_test_cnn1d.py             # Test du modèle CNN 1D sur nouvelles données
│
├── models/                      # Dossier pour sauvegarder les modèles
├── visualizations/              # Dossier pour sauvegarder les visualisations
├── test_results/                # Dossier pour les résultats de test
│
└── README.md                    # Ce fichier
```

---

## 🚀 Installation

### Prérequis
- Python 3.8+
- PyTorch (avec support GPU recommandé)
- CUDA (optionnel, pour GPU NVIDIA)

### Installation des dépendances

```bash
pip install numpy pandas torch torchvision scikit-learn matplotlib seaborn opencv-python
```

Pour Apple Silicon (M1/M2/M3) :
```bash
# PyTorch avec support MPS
pip install torch torchvision
```

Pour GPU NVIDIA :
```bash
# PyTorch avec support CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📊 Étape 1 : Analyse du Dataset

### Script : `1_analyze_dataset.py`

**Objectif** : Analyser les images hyperspectrales, détecter les centres des objets, et générer un fichier CSV de métadonnées.

**Fonctionnalités** :
- Détection automatique du centre des objets (par variance spectrale)
- Calcul de statistiques (dimensions, nombre de bandes, valeurs min/max)
- Génération de visualisations RGB
- Export des métadonnées vers `hyperspectral_dataset_summary.csv`

**Lancement** :
```bash
python 1_analyze_dataset.py \
    --data_dir dataset_reflec \
    --output_csv hyperspectral_dataset_summary.csv \
    --output_dir visualizations \
    --max_files 50
```

**Paramètres** :
- `--data_dir` : Répertoire contenant les fichiers .npy
- `--output_csv` : Nom du fichier CSV de sortie
- `--output_dir` : Dossier pour sauvegarder les visualisations
- `--max_files` : Nombre maximum de fichiers à analyser

**Sorties** :
- `hyperspectral_dataset_summary.csv` : Métadonnées (Filename, Height, Width, Bands, Xc, Yc, Class, etc.)
- Visualisations RGB de chaque image avec centre marqué

---

## 🧠 Étape 2 : Entraînement CNN 2D sur Patches

### Script : `2_cnn_2d_patch.py`

**Objectif** : Entraîner un CNN 2D en extrayant des patches spatiaux-spectraux autour du centre.

**Architecture** :
- Extraction de patches 2D (ex: 32×32 pixels × N bandes)
- CNN 2D avec convolutions spatiales
- Classification multi-classe

**Lancement** :
```bash
python 2_cnn_2d_patch.py \
    --data_dir dataset_reflec \
    --csv_path hyperspectral_dataset_summary.csv \
    --patch_size 32 \
    --architecture medium \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 0.001 \
    --save_model models/best_cnn2d_patch.pth \
    --save_history models/history_cnn2d_patch.png
```

**Paramètres principaux** :
- `--patch_size` : Taille du patch (ex: 16, 32, 64)
- `--architecture` : `simple`, `medium`, ou `deep`
- `--batch_size` : Taille des batchs
- `--epochs` : Nombre d'époques
- `--early_stopping` : Patience pour l'early stopping (défaut: 15)

**Modèle sauvegardé** : `models/best_cnn2d_patch.pth`

---

## 🔮 Étape 3 : Entraînement CNN 3D

### Script : `3_cnn_3d.py`

**Objectif** : Entraîner un CNN 3D en extrayant des cubes 3D (spatial + spectral) autour du centre.

**Architecture** :
- Extraction de cubes 3D (ex: 16×16×16 ou 32×32×32)
- Convolutions 3D pour capturer les relations spatiales ET spectrales
- Classification multi-classe

**Lancement** :
```bash
python 3_cnn_3d.py \
    --data_dir dataset_reflec \
    --csv_path hyperspectral_dataset_summary.csv \
    --cube_size 32 \
    --architecture medium \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 0.001 \
    --save_model models/best_cnn3d.pth \
    --save_history models/history_cnn3d.png
```

**Paramètres principaux** :
- `--cube_size` : Taille du cube 3D (ex: 16, 32)
- `--architecture` : `simple`, `medium`, ou `deep`
- `--batch_size` : Taille des batchs (plus petit pour CNN 3D, mémoire GPU)

**Modèle sauvegardé** : `models/best_cnn3d.pth`

---

## 🎯 Étape 4 : Entraînement CNN 1D Spectral (Recommandé)

### Script : `4_cnn_1d_spectral.py`

**Objectif** : Entraîner un CNN 1D en traitant chaque pixel comme un vecteur spectral individuel.

**Architecture** :
- Extraction de 10 000 pixels autour du centre
- CNN 1D avec convolutions le long de la dimension spectrale
- Classification pixel par pixel puis agrégation

**Lancement** :
```bash
python 4_cnn_1d_spectral.py \
    --data_dir dataset_reflec \
    --csv_path hyperspectral_dataset_summary.csv \
    --n_pixels_object 10000 \
    --max_files 50 \
    --architecture medium \
    --batch_size 128 \
    --epochs 50 \
    --learning_rate 0.001 \
    --save_model models/best_cnn1d_csv.pth \
    --save_history models/history_cnn1d_csv.png
```

**Paramètres principaux** :
- `--n_pixels_object` : Nombre de pixels à extraire autour du centre
- `--architecture` : `simple`, `medium`, ou `deep`
- `--batch_size` : Taille des batchs (peut être plus élevé)
- `--early_stopping` : Patience pour l'early stopping (défaut: 15)
- `--reduce_lr_patience` : Patience pour réduire le learning rate (défaut: 7)

**Modèle sauvegardé** : `models/best_cnn1d_csv.pth`

**Performances attendues** :
- Accuracy : **>92%**
- Loss : **~0.19**

---

## 🧪 Étape 5 : Test sur Nouvelles Données

### Script : `5_test_cnn1d.py`

**Objectif** : Évaluer le modèle CNN 1D entraîné sur le dataset de test `dataset_reflec_test`.

**Fonctionnalités** :
- Chargement du modèle pré-entraîné
- Extraction des pixels de test selon le CSV
- Calcul des métriques (accuracy, précision, rappel, F1-score)
- Génération de visualisations (matrice de confusion, accuracy par classe)
- Analyse détaillée des erreurs par fichier

**Lancement** :
```bash
python 5_test_cnn1d.py \
    --test_data_dir dataset_reflec_test \
    --csv_path hyperspectral_dataset_summary.csv \
    --model_path models/best_cnn1d_csv.pth \
    --n_pixels_object 10000 \
    --batch_size 128 \
    --output_dir test_results
```

**Paramètres** :
- `--test_data_dir` : Répertoire contenant les données de test
- `--csv_path` : Fichier CSV avec les centres des objets de test
- `--model_path` : Chemin vers le modèle entraîné
- `--output_dir` : Dossier pour sauvegarder les résultats

**Sorties** :
- `test_results/confusion_matrix_test.png` : Matrice de confusion
- `test_results/per_class_accuracy_test.png` : Accuracy par classe
- `test_results/misclassifications_analysis.txt` : Analyse des erreurs
- `test_results/test_results_summary.txt` : Résumé complet

---

## 📁 Modèles Sauvegardés

Les modèles sont sauvegardés au format PyTorch (`.pth`) avec les métadonnées complètes :

### Structure d'un modèle sauvegardé :
```python
{
    'epoch': int,                    # Époque du meilleur modèle
    'model_state_dict': dict,        # Poids du réseau
    'optimizer_state_dict': dict,    # État de l'optimiseur
    'val_acc': float,                # Meilleure accuracy de validation
    'architecture': str,             # Type d'architecture
    'num_classes': int,              # Nombre de classes
    'input_channels': int,           # Nombre de bandes spectrales
    'class_mapping': dict            # Mapping classe_id -> nom_classe
}
```

### Liste des modèles :

| Modèle | Fichier | Architecture | Performance |
|--------|---------|--------------|-------------|
| CNN 2D Patch | `models/best_cnn2d_patch.pth` | CNN 2D spatial | Variable |
| CNN 3D | `models/best_cnn3d.pth` | CNN 3D spatial-spectral | Variable |
| **CNN 1D Spectral** | `models/best_cnn1d_csv.pth` | CNN 1D spectral | **>92%** ✅ |

---

## 📈 Visualisations Générées

### Pendant l'entraînement :
- `history_*.png` : Courbes de loss et accuracy (train/validation)
- `sampling_class_*.png` : Visualisation de l'échantillonnage des pixels

### Pendant le test :
- `confusion_matrix_test.png` : Matrice de confusion
- `per_class_accuracy_test.png` : Histogramme d'accuracy par classe
- `misclassifications_analysis.txt` : Analyse détaillée des erreurs

---

## 🛠️ Conseils d'Utilisation

### 1. Choix de l'architecture

**CNN 1D Spectral (Recommandé)** :
- ✅ Meilleure performance (>92%)
- ✅ Rapide à entraîner
- ✅ Moins de mémoire GPU
- ✅ Traite efficacement les données hyperspectrales

**CNN 2D Patch** :
- Capture les relations spatiales locales
- Bon pour objets avec texture
- Plus lent que CNN 1D

**CNN 3D** :
- Capture relations spatiales ET spectrales
- Très gourmand en mémoire
- Temps d'entraînement long

### 2. Hyperparamètres recommandés

Pour CNN 1D (architecture `medium`) :
```bash
--batch_size 128
--epochs 50
--learning_rate 0.001
--weight_decay 1e-4
--early_stopping 15
--reduce_lr_patience 7
```

### 3. GPU vs CPU

**Apple Silicon (M1/M2/M3)** :
- Le script détecte automatiquement MPS
- Accélération GPU native

**NVIDIA GPU** :
- Détection automatique de CUDA
- Vérifier avec `torch.cuda.is_available()`

**CPU** :
- Fallback automatique
- Temps d'entraînement plus long

---

## 🔍 Format des Données

### Fichiers .npy
Cubes 3D au format NumPy :
- Shape : `(Height, Width, Bands)`
- Type : `float32` ou `float64`
- Valeurs : Réflectance normalisée (généralement entre 0 et 1)

### CSV de métadonnées
Colonnes requises :
- `Filename` : Nom du fichier .npy
- `Height`, `Width`, `Bands` : Dimensions
- `Xc`, `Yc` : Coordonnées du centre de l'objet
- `Class` : Label de classe (entier)
- `Min_Value`, `Max_Value`, `Mean_Value`, `Std_Value` : Statistiques

---

## 📊 Résultats Attendus

### CNN 1D Spectral (meilleure approche)
- **Accuracy globale** : >92%
- **Loss finale** : ~0.19
- **Temps d'entraînement** : ~10-20 min (GPU) / ~1-2h (CPU)

### Par classe
Les performances varient selon la classe :
- Classes bien séparées spectralement : >95%
- Classes similaires : 85-90%

---

## 🐛 Dépannage

### Erreur de mémoire GPU
```bash
# Réduire la taille du batch
--batch_size 64  # au lieu de 128

# Pour CNN 3D
--batch_size 16  # au lieu de 32
```

### Fichiers CSV non trouvés
Vérifier que le fichier `hyperspectral_dataset_summary.csv` existe :
```bash
python 1_analyze_dataset.py --data_dir dataset_reflec
```

### Modèle ne se charge pas
Vérifier la compatibilité PyTorch :
```python
checkpoint = torch.load('model.pth', weights_only=False)
```

---

## 📝 Citation

Si vous utilisez ce code dans vos recherches, veuillez citer :

```
Projet de Classification d'Images Hyperspectrales
Deep Learning pour analyse spectrale
2025
```

---

## 👥 Auteurs

Projet développé dans le cadre de recherches en télédétection hyperspectrale et deep learning.

---

## 📄 Licence

Ce projet est sous licence MIT. Libre d'utilisation pour la recherche et l'éducation.

---

## 🔗 Ressources Complémentaires

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Introduction à l'imagerie hyperspectrale](https://en.wikipedia.org/wiki/Hyperspectral_imaging)

---

**Bonne classification ! 🚀**