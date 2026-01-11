# === IMPORTS : Bibliothèques nécessaires ===
import argparse  # Pour parser les arguments en ligne de commande
import os  # Pour manipuler les fichiers et répertoires
import numpy as np  # Pour les calculs matriciels et tableaux
from sklearn import svm  # Algorithme SVM (Support Vector Machine)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Métriques d'évaluation
from sklearn.model_selection import train_test_split, cross_val_score  # Split des données et validation croisée
from sklearn.preprocessing import StandardScaler  # Normalisation des données
from sklearn.utils import shuffle  # Mélanger les données
from imblearn.over_sampling import SVMSMOTE  # SMOTE basé sur SVM pour gérer le déséquilibre
from collections import Counter  # Pour compter les échantillons par classe
import joblib  # Pour sauvegarder/charger les modèles
import gc  # Garbage collector pour libérer la mémoire


def extract_class_from_filename(file_name):
    """Extrait le label de classe depuis le nom de fichier."""
    # Exemple de nom de fichier : "image_G3_sample1.npy" où 3 est la classe
    
    base_name = os.path.basename(file_name)  # Extrait juste le nom du fichier (sans le chemin)
    # Exemple : "/path/to/image_G3_sample1.npy" → "image_G3_sample1.npy"
    
    if 'G' in base_name:  # Vérifie que le fichier contient 'G' (pour "Groupe" ou "Class")
        class_part = base_name.split('G')[-1]  # Split par 'G' et prend la dernière partie
        # Exemple : "image_G3_sample1.npy" → split → ['image_', '3_sample1.npy'] → prend '3_sample1.npy'
        
        class_label = class_part.split('_')[0]  # Split par '_' et prend le premier élément
        # Exemple : '3_sample1.npy' → split → ['3', 'sample1.npy'] → prend '3'
        
        return int(class_label)  # Convertit la string '3' en entier 3
    else:
        # Si pas de 'G' dans le nom, on lève une erreur
        raise ValueError(f"Le nom de fichier {file_name} ne contient pas d'information de classe valide.")


def sample_center_pixels(data, max_pixels):
    """
    Échantillonne les pixels au centre de l'image hyperspectrale.
    
    Pour une image de réflectance, on prend les pixels centraux car ils sont généralement
    plus représentatifs et moins affectés par les effets de bord.
    
    Args:
        data: Cube 3D (H, W, Bands)
        max_pixels: Nombre maximum de pixels à extraire
    
    Returns:
        feature_matrix: Matrice 2D (n_pixels, bands)
    """
    h, w, bands = data.shape
    total_pixels = h * w
    
    if total_pixels <= max_pixels:
        # Si le nombre total de pixels est inférieur à max_pixels, on prend tout
        return data.reshape(-1, bands)
    
    # Calculer le centre de l'image
    center_h = h // 2
    center_w = w // 2
    
    # Calculer la taille de la région centrale à extraire
    # On veut une région carrée (ou rectangulaire) au centre contenant max_pixels
    pixels_per_side = int(np.sqrt(max_pixels))
    
    # Calculer les bornes de la région centrale
    half_size_h = min(pixels_per_side // 2, center_h, h - center_h)
    half_size_w = min(pixels_per_side // 2, center_w, w - center_w)
    
    # Extraire la région centrale
    h_start = max(0, center_h - half_size_h)
    h_end = min(h, center_h + half_size_h)
    w_start = max(0, center_w - half_size_w)
    w_end = min(w, center_w + half_size_w)
    
    # Extraction de la région centrale
    center_region = data[h_start:h_end, w_start:w_end, :]
    
    # Reshape en matrice 2D
    feature_matrix = center_region.reshape(-1, bands)
    
    # Si on a plus de pixels que nécessaire, on en prend un sous-ensemble aléatoire
    if feature_matrix.shape[0] > max_pixels:
        indices = np.random.choice(feature_matrix.shape[0], max_pixels, replace=False)
        feature_matrix = feature_matrix[indices]
    
    print(f"  → Région centrale extraite: [{h_start}:{h_end}, {w_start}:{w_end}] = {feature_matrix.shape[0]} pixels")
    
    return feature_matrix


def load_data_memory_efficient(data_dir, data_type='reflec', max_pixels_per_image=1000, 
                                max_total_samples=100000, seed=42, use_center_sampling=True):
    """
    Charge les données de manière efficace en mémoire.
    
    Cette fonction lit les fichiers .npy, extrait les features et les labels,
    tout en contrôlant la mémoire utilisée.
    
    Args:
        data_dir: Liste des chemins vers les fichiers .npy
        data_type: Type de données ('reflec' pour réflectance, 'ghost' pour histogrammes GHOST)
        max_pixels_per_image: Nombre maximum de pixels/échantillons à extraire par fichier
        max_total_samples: Nombre maximum total d'échantillons à charger
        seed: Seed pour la reproductibilité
        use_center_sampling: Si True, échantillonne au centre des images (pour réflectance uniquement)
    """
    np.random.seed(seed)  # Fixe la seed pour que l'échantillonnage aléatoire soit reproductible
    
    # === INITIALISATION DES LISTES ===
    data_X = []  # Contiendra toutes les features (vecteurs de caractéristiques)
    data_y = []  # Contiendra tous les labels (classes)
    total_samples = 0  # Compteur du nombre total d'échantillons chargés
    
    print(f"Chargement des données (type: {data_type}) depuis {len(data_dir)} fichiers...")
    if data_type == 'reflec' and use_center_sampling:
        print("  Mode: Échantillonnage au CENTRE des images")
    else:
        print("  Mode: Échantillonnage aléatoire")
    
    # === BOUCLE SUR TOUS LES FICHIERS ===
    for idx, file_name in enumerate(data_dir):  # idx = index, file_name = chemin du fichier
        
        # Vérification de la limite totale d'échantillons
        if total_samples >= max_total_samples:
            print(f"Limite de {max_total_samples} échantillons atteinte. Arrêt du chargement.")
            break  # On arrête de charger plus de fichiers
            
        try:
            # === CHARGEMENT DU FICHIER .NPY ===
            data = np.load(file_name)  # Charge le fichier numpy
            # data peut être : 3D (H, W, Bands), 2D (N_samples, N_features), ou 1D (N_features)
            
            # === TRAITEMENT SELON LE TYPE DE DONNÉES ===
            if data_type == 'ghost':
                # GHOST : Matrice 2D (histogrammes concaténés)
                # Chaque ligne = un échantillon (histogramme complet)
                if data.ndim == 2:
                    # data.shape = (n_samples, n_features)
                    # n_features = hist_magnitude + hist_direction + hist_intensity + hist_shape
                    
                    n_samples_in_file = data.shape[0]
                    
                    # Échantillonnage aléatoire si nécessaire (pas de centre pour GHOST)
                    if n_samples_in_file > max_pixels_per_image:
                        indices = np.random.choice(n_samples_in_file, max_pixels_per_image, replace=False)
                        feature_matrix = data[indices]
                    else:
                        feature_matrix = data
                        
                elif data.ndim == 1:
                    # Cas d'un seul échantillon (vecteur 1D)
                    feature_matrix = data[np.newaxis, :]
                else:
                    print(f"Format GHOST inattendu pour {file_name}: {data.shape}. Ignoré.")
                    continue
                    
            elif data_type == 'reflec':
                # Réflectance: Cube 3D (L, C, B) ou vecteur 1D
                if data.ndim == 3:
                    # Cube de réflectance (hauteur, largeur, bandes spectrales)
                    h, w, bands = data.shape
                    
                    # Choix de la stratégie d'échantillonnage
                    if use_center_sampling:
                        # ⭐ ÉCHANTILLONNAGE AU CENTRE
                        feature_matrix = sample_center_pixels(data, max_pixels_per_image)
                    else:
                        # Échantillonnage spatial aléatoire (ancien comportement)
                        total_pixels = h * w
                        if total_pixels > max_pixels_per_image:
                            indices = np.random.choice(total_pixels, max_pixels_per_image, replace=False)
                            feature_matrix = data.reshape(-1, bands)[indices]
                        else:
                            feature_matrix = data.reshape(-1, bands)
                        
                elif data.ndim == 1:
                    # Vecteur de features moyennes
                    feature_matrix = data[np.newaxis, :]
                else:
                    print(f"Format réflectance inattendu pour {file_name}: {data.shape}. Ignoré.")
                    continue
            else:
                raise ValueError(f"Type de données non supporté: {data_type}")
            
            # Libérer la mémoire du tableau original
            del data
            
            # Extraction du label de classe
            class_label = extract_class_from_filename(file_name)
            
            # Vérification de validité
            if class_label < 1:
                print(f"Attention: label invalide {class_label} pour {file_name}. Ignoré.")
                continue
            
            # Ajout des données
            data_X.append(feature_matrix.astype(np.float32))
            # ✅ CORRECTION: Utilisation de np.full avec syntaxe correcte
            data_y.append(np.full(feature_matrix.shape[0], class_label - 1, dtype=np.int32))
            
            total_samples += feature_matrix.shape[0]
            
            if (idx + 1) % 10 == 0:
                print(f"Traité {idx + 1}/{len(data_dir)} fichiers, {total_samples} échantillons")
                
        except Exception as e:
            print(f"Erreur lors du chargement de {file_name}: {e}")
            continue
    
    if len(data_X) == 0:
        raise ValueError("Aucune donnée n'a pu être chargée!")
    
    # Concaténation finale
    print("Concaténation des données...")
    X_final = np.vstack(data_X)
    y_final = np.concatenate(data_y)
    
    # Libération mémoire
    del data_X, data_y
    gc.collect()
    
    print(f"Données chargées: X shape = {X_final.shape}, y shape = {y_final.shape}")
    print(f"Classes présentes: {np.unique(y_final)}")
    print(f"Distribution des classes: {dict(Counter(y_final))}")
    
    return X_final, y_final


def apply_svmsmote(X_train, y_train, sampling_strategy='auto', k_neighbors=5, svm_estimator_params=None):
    """
    Applique SVMSMOTE pour rééquilibrer les classes.
    
    SVMSMOTE combine SMOTE avec un classificateur SVM pour générer des échantillons
    synthétiques uniquement dans les régions de décision difficiles.
    
    Args:
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        sampling_strategy: Stratégie de rééquilibrage
            - 'auto': rééquilibre toutes les classes minoritaires
            - 'all': rééquilibre toutes les classes à la taille de la classe majoritaire
            - dict: {class_label: n_samples} pour un contrôle fin
        k_neighbors: Nombre de voisins pour SMOTE
        svm_estimator_params: Paramètres pour le SVM interne de SVMSMOTE
    
    Returns:
        X_resampled, y_resampled: Données rééquilibrées
    """
    print(f"\n{'='*60}")
    print("APPLICATION DE SVMSMOTE")
    print(f"{'='*60}")
    
    # Affichage de la distribution avant
    print(f"\nDistribution AVANT SVMSMOTE:")
    counter_before = Counter(y_train)
    for class_label, count in sorted(counter_before.items()):
        print(f"  Classe {class_label}: {count} échantillons ({100*count/len(y_train):.2f}%)")
    
    # Calcul du ratio de déséquilibre
    max_count = max(counter_before.values())
    min_count = min(counter_before.values())
    imbalance_ratio = max_count / min_count
    print(f"\nRatio de déséquilibre: {imbalance_ratio:.2f}:1")
    
    # Vérification si SVMSMOTE est nécessaire
    if imbalance_ratio < 1.5:
        print("⚠️  Dataset relativement équilibré (ratio < 1.5), SVMSMOTE peut ne pas être nécessaire.")
        user_input = input("Continuer avec SVMSMOTE ? (y/n): ")
        if user_input.lower() != 'y':
            print("SVMSMOTE annulé, utilisation du dataset original.")
            return X_train, y_train
    
    # Configuration par défaut du SVM pour SVMSMOTE
    if svm_estimator_params is None:
        svm_estimator_params = {
            'kernel': 'linear',
            'class_weight': 'balanced'
        }
    
    try:
        # Création de l'objet SVMSMOTE
        print(f"\nApplication de SVMSMOTE (k_neighbors={k_neighbors})...")
        smote = SVMSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            svm_estimator=svm.SVC(**svm_estimator_params),
            random_state=42
        )
        
        # Application du rééquilibrage
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Affichage de la distribution après
        print(f"\nDistribution APRÈS SVMSMOTE:")
        counter_after = Counter(y_resampled)
        for class_label, count in sorted(counter_after.items()):
            diff = count - counter_before.get(class_label, 0)
            print(f"  Classe {class_label}: {count} échantillons (+{diff} synthétiques)")
        
        print(f"\nTotal échantillons: {len(y_train)} → {len(y_resampled)} (+{len(y_resampled)-len(y_train)})")
        print("✓ SVMSMOTE appliqué avec succès!")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'application de SVMSMOTE: {e}")
        print("Utilisation du dataset original sans rééquilibrage.")
        return X_train, y_train


def parse_args():
    parser = argparse.ArgumentParser(description='Train SVM on hyperspectral or GHOST data with SVMSMOTE.')
    
    # Paramètres des données
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the directory containing .npy files.')
    parser.add_argument('--data_type', type=str, default='reflec',
                        choices=['reflec', 'ghost'],
                        help='Type of data: reflec (hyperspectral) or ghost (histograms).')
    parser.add_argument('--max_pixels', type=int, default=900,
                        help='Max samples per file.')
    parser.add_argument('--max_samples', type=int, default=1000000,
                        help='Max total samples to load.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set proportion.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--center_sampling', action='store_true',
                        help='Sample pixels from center of images (reflec only).')
    
    # Paramètres SVMSMOTE
    parser.add_argument('--use_smote', action='store_true',
                        help='Apply SVMSMOTE to balance classes.')
    parser.add_argument('--smote_strategy', type=str, default='auto',
                        help='SMOTE sampling strategy (auto, all, or minority).')
    parser.add_argument('--smote_k_neighbors', type=int, default=5,
                        help='Number of neighbors for SMOTE.')
    
    # Paramètres SVM
    parser.add_argument('--kernel', type=str, default='rbf',
                        choices=['linear', 'poly', 'rbf', 'sigmoid'],
                        help='Kernel type for SVM.')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Regularization parameter for SVM.')
    parser.add_argument('--gamma', type=str, default='scale',
                        help='Kernel coefficient for SVM.')
    parser.add_argument('--degree', type=int, default=3,
                        help='Degree for polynomial kernel.')
    parser.add_argument('--cache_size', type=int, default=500,
                        help='Kernel cache size in MB.')
    parser.add_argument('--class_weight', type=str, default=None,
                        help='Class weights (None or balanced).')
    
    # Paramètres d'entraînement
    parser.add_argument('--repeat', type=int, default=1,
                        help='Number of training repetitions.')
    parser.add_argument('--cv', action='store_true',
                        help='Perform cross-validation.')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of CV folds.')
    
    # Sauvegarde
    parser.add_argument('--save_model', type=str, default=None,
                        help='Path to save the trained model.')
    parser.add_argument('--save_scaler', type=str, default=None,
                        help='Path to save the scaler.')
    
    return parser.parse_args()


def train_and_evaluate(clf, X_train, y_train, X_test, y_test, run_idx=1, do_cv=False, cv_folds=5):
    """Entraîne et évalue le modèle SVM."""
    
    print(f"\n--- {'Validation croisée' if do_cv else f'Répétition {run_idx}'} ---")
    
    # Validation croisée optionnelle
    if do_cv:
        print(f"Validation croisée avec {cv_folds} folds...")
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv_folds, 
                                     scoring='accuracy', n_jobs=-1, verbose=1)
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"Scores par fold: {cv_scores}")
    
    # Entraînement
    print("Entraînement en cours...")
    clf.fit(X_train, y_train)
    
    # Prédiction
    print("Prédiction...")
    y_pred = clf.predict(X_test)
    
    # Métriques
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy sur le test set: {acc:.4f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_sum[cm_sum == 0] = 1  # Éviter division par zéro
    cm_percent = (cm.astype('float') / cm_sum) * 100
    
    print("\nMatrice de confusion (pourcentage par classe):")
    np.set_printoptions(precision=2, suppress=True)
    print(cm_percent)
    
    print("\nMatrice de confusion (nombres absolus):")
    print(cm)
    
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return acc, clf


def main():
    args = parse_args()
    
    # Définir la seed globale
    np.random.seed(args.seed)
    
    # Vérification du répertoire
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Le répertoire {args.data_dir} n'existe pas!")
    
    # Liste des fichiers
    feature_files = [os.path.join(args.data_dir, f) 
                     for f in os.listdir(args.data_dir) 
                     if f.endswith('.npy')]
    
    if len(feature_files) == 0:
        raise ValueError(f"Aucun fichier .npy trouvé dans {args.data_dir}")
    
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Type de données: {args.data_type}")
    print(f"  - Fichiers trouvés: {len(feature_files)}")
    print(f"  - Échantillonnage: {'Centre des images' if args.center_sampling else 'Aléatoire'}")
    print(f"  - SVMSMOTE: {'Activé' if args.use_smote else 'Désactivé'}")
    print(f"  - Kernel SVM: {args.kernel}")
    print(f"  - C: {args.C}, gamma: {args.gamma}")
    print(f"  - Seed: {args.seed}")
    print(f"{'='*60}\n")
    
    # Chargement des données
    X, y = load_data_memory_efficient(
        feature_files,
        data_type=args.data_type,
        max_pixels_per_image=args.max_pixels,
        max_total_samples=args.max_samples,
        seed=args.seed,
        use_center_sampling=args.center_sampling
    )
    
    # Split train/test
    print("\nSéparation train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Distribution train: {dict(Counter(y_train))}")
    print(f"Distribution test: {dict(Counter(y_test))}")
    
    # Normalisation AVANT SVMSMOTE
    print("\nNormalisation des données (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Stats après normalisation - Train: mean={X_train.mean():.4f}, std={X_train.std():.4f}")
    print(f"Stats après normalisation - Test: mean={X_test.mean():.4f}, std={X_test.std():.4f}")
    
    # Application de SVMSMOTE si demandé
    if args.use_smote:
        X_train, y_train = apply_svmsmote(
            X_train, y_train,
            sampling_strategy=args.smote_strategy,
            k_neighbors=args.smote_k_neighbors
        )
    
    # Libération mémoire
    del X, y
    gc.collect()
    
    # Entraînement
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT SVM")
    print(f"{'='*60}\n")
    
    accuracies = []
    best_acc = 0
    best_clf = None
    
    for i in range(args.repeat):
        # IMPORTANT : Change la seed à chaque répétition pour tester la robustesse
        current_seed = args.seed + i
        print(f"\n🎲 Utilisation de seed={current_seed} pour cette répétition")
        
        # Mélange différent à chaque répétition
        if args.repeat > 1:
            X_train_shuffled, y_train_shuffled = shuffle(
                X_train, y_train, random_state=current_seed
            )
        else:
            X_train_shuffled, y_train_shuffled = X_train, y_train
        
        # Création du modèle
        clf = svm.SVC(
            kernel=args.kernel,
            C=args.C,
            gamma=args.gamma,
            degree=args.degree,
            cache_size=args.cache_size,
            class_weight=args.class_weight,
            random_state=current_seed
        )
        
        # Entraînement et évaluation
        acc, trained_clf = train_and_evaluate(
            clf, X_train_shuffled, y_train_shuffled, X_test, y_test,
            run_idx=i+1, do_cv=(args.cv and i == 0), cv_folds=args.cv_folds
        )
        
        accuracies.append(acc)
        
        # Garder le meilleur modèle
        if acc > best_acc:
            best_acc = acc
            best_clf = trained_clf
    
    # Résumé final
    print(f"\n{'='*60}")
    print(f"RÉSULTATS FINAUX")
    print(f"{'='*60}")
    
    if args.repeat > 1:
        print(f"Nombre de répétitions: {args.repeat}")
        print(f"Seeds utilisées: {args.seed} à {args.seed + args.repeat - 1}")
        print(f"\nAccuracy moyenne: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Min: {np.min(accuracies):.4f}, Max: {np.max(accuracies):.4f}")
        print(f"Meilleure accuracy: {best_acc:.4f}")
        print(f"\nToutes les accuracies: {[f'{a:.4f}' for a in accuracies]}")
        
        # 🔍 DIAGNOSTIC DE VARIABILITÉ
        std_acc = np.std(accuracies)
        if std_acc < 0.001:
            print(f"\n⚠️  ATTENTION : Variabilité très faible (std={std_acc:.6f})")
            print("   Causes possibles:")
            print("   1. Seed fixe tout (normal pour reproductibilité)")
            print("   2. Données parfaitement séparables")
            print("   3. Tous les échantillons utilisés à chaque fois")
            print("   → Testez avec --seed différent ou --max_pixels plus petit")
        elif std_acc < 0.01:
            print(f"\n✅ Variabilité normale (std={std_acc:.6f}) - Modèle stable!")
        else:
            print(f"\n⚠️  Variabilité élevée (std={std_acc:.6f})")
            print("   → Considérez augmenter le dataset ou vérifier le déséquilibre des classes")
    else:
        print(f"Accuracy: {accuracies[0]:.4f}")
    
    # Sauvegarde du modèle
    if args.save_model:
        print(f"\nSauvegarde du meilleur modèle dans '{args.save_model}'...")
        joblib.dump(best_clf, args.save_model)
        print("✓ Modèle sauvegardé")
    
    if args.save_scaler:
        print(f"Sauvegarde du scaler dans '{args.save_scaler}'...")
        joblib.dump(scaler, args.save_scaler)
        print("✓ Scaler sauvegardé")
    
    print(f"\n{'='*60}")
    print("Entraînement terminé!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()