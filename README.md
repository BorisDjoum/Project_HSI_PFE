# Projet ImViA — Classification d'Images Hyperspectrales

## 📋 Description

Ce dépôt contient des scripts pour l'extraction, l'entraînement et l'évaluation de modèles de classification sur images hyperspectrales. Les approches disponibles incluent :
- CNN 1D (spectral) — traitement pixel par pixel le long des bandes
- CNN 2D (patchs) — convolutions spatiales sur patches multispectraux
- CNN 3D (cubes) — convolutions spatiales+spectrales

Le workflow typique : extraire des pixels/cubes autour des centres d'objets, entraîner un modèle, puis évaluer sur un jeu de test.

---

## 📁 Fichiers et scripts importants

- `data_processor.py` : traitements d'utilité (extraction, normalisation)
- `loader.py` : utilitaires de chargement et création des datasets
- `cnn_1d.py` : entraînement et test du CNN 1D (spectral) — principal script utilisé
- `cnn_2d.py` : entraînement CNN 2D (patches)
- `cnn_3d.py` : entraînement CNN 3D (cubes)
- `transfer.py` : entraînement par transfert (patch-based, fine-tuning de backbones pré-entraînés)
- `test_cnn1d.py` / `test_spectral.ipynb` : évaluation et analyses
- `train.ipynb`, `hyper_classifier.ipynb` : notebooks d'expérimentation
- `dataset_reflec/`, `dataset_reflec_test/`, `dataset_hist/`, `dataset_hist2/` : exemples de dossiers de données
- `hyperspectral_dataset_summary.csv` : fichier de métadonnées (centres, classes)

---

## 🚀 Installation rapide

Prérequis : Python 3.8+, PyTorch (MPS ou CUDA si disponible).

Installation minimale :

```bash
pip install -r requirements.txt
# ou
pip install numpy pandas torch torchvision scikit-learn matplotlib seaborn opencv-python
```

Pour Apple Silicon, installez la roue PyTorch compatible MPS ; pour NVIDIA, installez la roue CUDA adaptée.

---

## 🔧 Exemples d'utilisation

### 1) Entraîner le CNN 1D (spectral)

```bash
python cnn_1d.py \
  --data_dir dataset_hist2 \
  --csv_path hyperspectral_dataset_summary.csv \
  --n_pixels_object 10000 \
  --max_files 109 \
  --architecture medium \
  --batch_size 128 \
  --epochs 60 \
  --learning_rate 0.001
```

Paramètres principaux : `--data_dir`, `--csv_path`, `--n_pixels_object`, `--architecture` (`simple|medium|deep`), `--batch_size`, `--epochs`.

### 2) Entraîner le CNN 2D (patches)

```bash
python cnn_2d.py \
  --data_dir dataset_reflec \
  --csv_path hyperspectral_dataset_summary.csv \
  --patch_size 32 \
  --architecture medium \
  --batch_size 64 \
  --epochs 100
```

### 3) Entraîner le CNN 3D (cubes)

```bash
python cnn_3d.py \
  --data_dir dataset_reflec \
  --csv_path hyperspectral_dataset_summary.csv \
  --patch_size 11 \
  --stride 2 \
  --architecture deep \
  --batch_size 16 \
  --epochs 100
```

### 4) Entraînement par transfert — `transfer.py`

**Objectif** : Fine-tuning d'un backbone pré-entraîné (ResNet / MobileNet / DenseNet) sur des patches extraits des `.npy`. Le jeu de données est indexé de façon paresseuse (lazy), utile pour traiter de grands volumes sans tout charger en mémoire.

**Options principales** :
- `--data_dir` : répertoire contenant les `.npy`
- `--data_type` : `reflec` (cubes H×W×B) ou `ghost` (histogrammes 1D)
- `--patch_size` : taille des patches extraits (ex: 11, 32)
- `--stride` : pas d'extraction des patches
- `--max_patches_per_file` : nombre max de patches par fichier
- `--max_samples` : nombre total maximal d'échantillons indexés
- `--arch` : `resnet50` | `mobilenet_v2` | `densenet121` (backbones pré-entraînés)
- `--batch_size`, `--epochs`, `--lr`
- `--save_model` (par défaut `best_transfer.pt`), `--save_history`

**Particularités** :
- Le backbone est gelé (les poids ne sont pas entraînés) et seule la tête (classifier) est entraînée par défaut.
- Le dataset est construit de façon paresseuse : seuls les patches nécessaires sont lus au runtime.
- Le script détecte automatiquement MPS / CUDA / CPU.

**Exemple d'exécution (commande typique)** :
```bash
python transfer.py \
  --data_dir dataset_reflec \
  --data_type reflec \
  --arch densenet121 \
  --max_samples 100000 \
  --max_patches_per_file 100 \
  --stride 5 \
  --batch_size 64 \
  --epochs 50
```

**Sorties** :
- `best_transfer.pt` : checkpoint du meilleur modèle (etat du classifier et métadonnées)
- `history_transfer.png` : courbes d'entraînement (loss / accuracy)

### 5) Évaluer un modèle pré-entraîné

```bash
python test_cnn1d.py \
  --test_data_dir dataset_reflec_test \
  --csv_path hyperspectral_dataset_summary.csv \
  --model_path best_cnn1d_csv.pth \
  --n_pixels_object 10000 \
  --batch_size 128
```

---

## ⚠️ Notes importantes et dépannage

- Validation de la taille spectrale (nouvelle sécurité) :
  - Le script `cnn_1d.py` vérifie désormais que le nombre de bandes (`input_channels`) est suffisant pour l'architecture choisie (ex. `medium`/`deep` effectuent 3 poolings → nécessité d'au moins 8 bandes). Si `input_channels` est trop petit, une **ValueError** explicite sera levée avec un message d'aide.
  - Solution : utiliser `--architecture simple` ou fournir des fichiers `.npy` avec plus de bandes.

- Erreur liée à `MaxPool1d` (séquence trop courte) : signifie généralement que la longueur spectrale a été réduite à 0 après pooling — voir point précédent.

- GPU OOM : réduire `--batch_size` (ex. 128 → 64 ou 16 pour CNN 3D).

- CSV manquant : vérifiez que `hyperspectral_dataset_summary.csv` contient les colonnes requises (`Nom_Fichier_npy` / `Filename`, `Yc`, `Xc`, `Classe` selon le script utilisé).

---

## 📄 Format des données

- `.npy` : cubes 3D `(Height, Width, Bands)`
- CSV de centres : contient au minimum les colonnes indiquant le nom du fichier et les coordonnées du centre (Xc/Yc) et la classe.

---

## 🧾 Remarques finales

- Les notebooks (`train.ipynb`, `hyper_classifier.ipynb`) contiennent des expériences et visualisations complémentaires.
- Pour toute question ou problème reproductible, ouvrez une issue en précisant la commande exécutée et l'erreur complète.

---

*README mis à jour pour refléter les scripts et comportements actuels du dépôt.*

Si vous utilisez ce code dans vos recherches, veuillez citer :

```
Projet de Classification d'Images Hyperspectrales
Deep Learning pour analyse spectrale
2025
```

---

## 👥 Auteurs

- Boris DJOUM

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