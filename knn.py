"""
K-Nearest Neighbors (KNN) pour la classification des données GHOST

KNN est un algorithme simple et efficace qui classe un échantillon
en fonction de ses K voisins les plus proches dans l'espace des features.
"""

import argparse
import os
import numpy as np
from collections import Counter
import gc
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extract_class_from_filename(file_name):
    """Extrait le label de classe depuis le nom de fichier."""
    base_name = os.path.basename(file_name)
    if 'G' in base_name:
        class_part = base_name.split('G')[-1]
        class_label = class_part.split('_')[0]
        return int(class_label)
    else:
        raise ValueError(f"Le nom de fichier {file_name} ne contient pas d'information de classe valide.")


def load_ghost_data(data_dir, max_samples_per_file=2000, max_total_samples=500000, seed=42):
    """
    Charge les données GHOST (histogrammes 2D).
    
    Args:
        data_dir: Liste des chemins vers les fichiers .npy
        max_samples_per_file: Nombre max d'échantillons par fichier
        max_total_samples: Nombre max total d'échantillons
        seed: Seed pour reproductibilité
    """
    np.random.seed(seed)
    
    all_samples = []
    all_labels = []
    total_samples = 0
    
    print(f"Chargement des données GHOST...")
    print(f"Paramètres: max_samples_per_file={max_samples_per_file}, max_total={max_total_samples}")
    
    for idx, file_name in enumerate(data_dir):
        if total_samples >= max_total_samples:
            print(f"Limite de {max_total_samples} échantillons atteinte.")
            break
        
        try:
            data = np.load(file_name)
            
            # GHOST doit être 2D (N_samples, N_features)
            if data.ndim == 2:
                n_samples = data.shape[0]
                
                # Échantillonnage si nécessaire
                if n_samples > max_samples_per_file:
                    indices = np.random.choice(n_samples, max_samples_per_file, replace=False)
                    samples = data[indices]
                else:
                    samples = data
            
            elif data.ndim == 1:
                # Un seul échantillon
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
            all_labels.append(np.full(len(samples), class_label - 1, dtype=np.int32))
            
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
    print(f"  X shape: {X.shape} (n_samples, n_features)")
    print(f"  y shape: {y.shape}")
    print(f"  Classes: {np.unique(y)}")
    print(f"  Distribution: {dict(Counter(y))}")
    
    return X, y


def plot_k_optimization(k_range, scores, save_path=None):
    """
    Affiche la courbe d'optimisation du paramètre K.
    
    Args:
        k_range: Liste des valeurs de K testées
        scores: Scores d'accuracy correspondants
        save_path: Chemin pour sauvegarder le graphique
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Nombre de voisins (K)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Optimisation du paramètre K pour KNN', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Marquer le meilleur K
    best_k = k_range[np.argmax(scores)]
    best_score = max(scores)
    plt.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, 
                label=f'Meilleur K={best_k} (acc={best_score:.4f})')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique d'optimisation sauvegardé: {save_path}")
    
    plt.close()


def optimize_k(X_train, y_train, k_range, cv_folds=5, n_jobs=-1):
    """
    Trouve la meilleure valeur de K par validation croisée.
    
    Args:
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        k_range: Liste des valeurs de K à tester
        cv_folds: Nombre de folds pour la validation croisée
        n_jobs: Nombre de CPU à utiliser (-1 = tous)
    
    Returns:
        best_k: Meilleure valeur de K
        scores: Liste des scores pour chaque K
    """
    print(f"\n{'='*70}")
    print(f"OPTIMISATION DU PARAMÈTRE K")
    print(f"{'='*70}")
    print(f"Valeurs de K à tester: {list(k_range)}")
    print(f"Validation croisée: {cv_folds} folds")
    print(f"Cela peut prendre quelques minutes...\n")
    
    scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=n_jobs)
        cv_scores = cross_val_score(knn, X_train, y_train, cv=cv_folds, 
                                     scoring='accuracy', n_jobs=n_jobs)
        mean_score = cv_scores.mean()
        scores.append(mean_score)
        print(f"K={k:2d}: Accuracy = {mean_score:.4f} (±{cv_scores.std():.4f})")
    
    best_idx = np.argmax(scores)
    best_k = k_range[best_idx]
    best_score = scores[best_idx]
    
    print(f"\n✓ Meilleur K trouvé: {best_k} avec accuracy = {best_score:.4f}")
    print(f"{'='*70}\n")
    
    return best_k, scores


def train_and_evaluate_knn(X_train, y_train, X_test, y_test, 
                          k=5, weights='uniform', metric='minkowski', 
                          algorithm='auto', n_jobs=-1):
    """
    Entraîne et évalue un modèle KNN.
    
    Args:
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
        k: Nombre de voisins
        weights: 'uniform' ou 'distance' (pondération par distance)
        metric: Métrique de distance ('euclidean', 'manhattan', 'minkowski')
        algorithm: 'auto', 'ball_tree', 'kd_tree', ou 'brute'
        n_jobs: Nombre de CPU
    
    Returns:
        model: Modèle KNN entraîné
        y_pred: Prédictions
        metrics: Dictionnaire des métriques
    """
    print(f"\n{'='*70}")
    print(f"ENTRAÎNEMENT KNN")
    print(f"{'='*70}")
    print(f"Paramètres:")
    print(f"  K (n_neighbors): {k}")
    print(f"  Weights: {weights}")
    print(f"  Metric: {metric}")
    print(f"  Algorithm: {algorithm}")
    print(f"  N_jobs: {n_jobs}")
    print(f"{'='*70}\n")
    
    # Création du modèle
    start_time = time.time()
    
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights=weights,
        metric=metric,
        algorithm=algorithm,
        n_jobs=n_jobs
    )
    
    # Entraînement (pour KNN, c'est juste stocker les données)
    print("Entraînement du modèle KNN...")
    knn.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"✓ Entraînement terminé en {train_time:.2f}s")
    
    # Prédiction
    print("Prédiction sur le test set...")
    start_time = time.time()
    y_pred = knn.predict(X_test)
    predict_time = time.time() - start_time
    print(f"✓ Prédiction terminée en {predict_time:.2f}s")
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_time': train_time,
        'predict_time': predict_time
    }
    
    print(f"\n{'='*70}")
    print(f"RÉSULTATS")
    print(f"{'='*70}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Temps entraînement: {train_time:.2f}s")
    print(f"Temps prédiction:   {predict_time:.2f}s")
    print(f"{'='*70}\n")
    
    # Matrice de confusion
    print("Matrice de confusion:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Matrice de confusion en pourcentage
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_sum[cm_sum == 0] = 1
    cm_percent = (cm.astype('float') / cm_sum) * 100
    
    print("\nMatrice de confusion (pourcentage par classe):")
    np.set_printoptions(precision=2, suppress=True)
    print(cm_percent)
    
    # Rapport de classification
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return knn, y_pred, metrics


def compare_distance_metrics(X_train, y_train, X_test, y_test, k=5):
    """
    Compare différentes métriques de distance pour KNN.
    
    Args:
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
        k: Nombre de voisins
    
    Returns:
        results: Dictionnaire des résultats par métrique
    """
    print(f"\n{'='*70}")
    print(f"COMPARAISON DES MÉTRIQUES DE DISTANCE")
    print(f"{'='*70}\n")
    
    metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    results = {}
    
    for metric in metrics:
        print(f"Test avec métrique: {metric}")
        try:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
            
            start_time = time.time()
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            elapsed_time = time.time() - start_time
            
            accuracy = accuracy_score(y_test, y_pred)
            
            results[metric] = {
                'accuracy': accuracy,
                'time': elapsed_time
            }
            
            print(f"  Accuracy: {accuracy:.4f}, Temps: {elapsed_time:.2f}s\n")
        
        except Exception as e:
            print(f"  Erreur avec {metric}: {e}\n")
            results[metric] = {'accuracy': 0.0, 'time': 0.0}
    
    # Affichage du résumé
    print(f"{'='*70}")
    print(f"RÉSUMÉ DES MÉTRIQUES")
    print(f"{'='*70}")
    for metric, res in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{metric:12s}: Accuracy={res['accuracy']:.4f}, Temps={res['time']:.2f}s")
    
    best_metric = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\n✓ Meilleure métrique: {best_metric} (accuracy={results[best_metric]['accuracy']:.4f})")
    print(f"{'='*70}\n")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Train KNN on GHOST data.')
    
    # Données
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing .npy files.')
    parser.add_argument('--max_samples_per_file', type=int, default=500,
                        help='Max samples per file.')
    parser.add_argument('--max_samples', type=int, default=50000,
                        help='Max total samples.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set proportion.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    
    # SMOTE
    parser.add_argument('--use_smote', action='store_true',
                        help='Apply SMOTE to balance classes.')
    parser.add_argument('--smote_k_neighbors', type=int, default=5,
                        help='Number of neighbors for SMOTE.')
    
    # KNN Paramètres
    parser.add_argument('--k', type=int, default=5,
                        help='Number of neighbors (K). Use 0 for auto-optimization.')
    parser.add_argument('--k_range_min', type=int, default=1,
                        help='Min K for optimization.')
    parser.add_argument('--k_range_max', type=int, default=21,
                        help='Max K for optimization.')
    parser.add_argument('--k_range_step', type=int, default=2,
                        help='Step for K range.')
    parser.add_argument('--weights', type=str, default='uniform',
                        choices=['uniform', 'distance'],
                        help='Weight function (uniform or distance).')
    parser.add_argument('--metric', type=str, default='minkowski',
                        choices=['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
                        help='Distance metric.')
    parser.add_argument('--algorithm', type=str, default='auto',
                        choices=['auto', 'ball_tree', 'kd_tree', 'brute'],
                        help='Algorithm to compute nearest neighbors.')
    
    # Optimisation et analyse
    parser.add_argument('--optimize_k', action='store_true',
                        help='Optimize K parameter using cross-validation.')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of folds for cross-validation.')
    parser.add_argument('--compare_metrics', action='store_true',
                        help='Compare different distance metrics.')
    
    # Sauvegarde
    parser.add_argument('--save_k_plot', type=str, default=None,
                        help='Path to save K optimization plot.')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Seed
    np.random.seed(args.seed)
    
    # Vérification du répertoire
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Le répertoire {args.data_dir} n'existe pas!")
    
    feature_files = [os.path.join(args.data_dir, f) 
                     for f in os.listdir(args.data_dir) 
                     if f.endswith('.npy')]
    
    if len(feature_files) == 0:
        raise ValueError(f"Aucun fichier .npy trouvé dans {args.data_dir}")
    
    print(f"{'='*70}")
    print(f"CONFIGURATION KNN - GHOST Data")
    print(f"{'='*70}")
    print(f"  Fichiers trouvés: {len(feature_files)}")
    print(f"  K (n_neighbors): {args.k if args.k > 0 else 'Auto-optimisation'}")
    print(f"  Weights: {args.weights}")
    print(f"  Metric: {args.metric}")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  SMOTE: {'Activé' if args.use_smote else 'Désactivé'}")
    print(f"  Optimize K: {'Activé' if args.optimize_k else 'Désactivé'}")
    print(f"  Compare metrics: {'Activé' if args.compare_metrics else 'Désactivé'}")
    print(f"{'='*70}\n")
    
    # Chargement des données
    X, y = load_ghost_data(
        feature_files,
        max_samples_per_file=args.max_samples_per_file,
        max_total_samples=args.max_samples,
        seed=args.seed
    )
    
    num_classes = len(np.unique(y))
    print(f"\nNombre de classes: {num_classes}")
    
    # Split train/test
    print("\nSplit des données...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Distribution train: {dict(Counter(y_train))}")
    print(f"  Distribution test: {dict(Counter(y_test))}")
    
    # Normalisation (TRÈS IMPORTANT pour KNN!)
    print("\nNormalisation des données (StandardScaler)...")
    print("⚠️  La normalisation est CRUCIALE pour KNN car il est basé sur les distances!")
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    print(f"Stats train: mean={X_train_norm.mean():.4f}, std={X_train_norm.std():.4f}")
    print(f"Stats test: mean={X_test_norm.mean():.4f}, std={X_test_norm.std():.4f}")
    
    # Application de SMOTE si demandé
    if args.use_smote:
        print(f"\n{'='*70}")
        print("APPLICATION DE SMOTE")
        print(f"{'='*70}")
        print(f"Distribution AVANT SMOTE: {dict(Counter(y_train))}")
        
        smote = SMOTE(k_neighbors=args.smote_k_neighbors, random_state=args.seed)
        X_train_norm, y_train = smote.fit_resample(X_train_norm, y_train)
        
        print(f"Distribution APRÈS SMOTE: {dict(Counter(y_train))}")
        print(f"Total échantillons: {len(y_train)}")
        print(f"{'='*70}\n")
    
    # Libération mémoire
    del X, y, X_train, X_test
    gc.collect()
    
    # Comparaison des métriques de distance si demandé
    if args.compare_metrics:
        k_for_comparison = args.k if args.k > 0 else 5
        compare_distance_metrics(X_train_norm, y_train, X_test_norm, y_test, k=k_for_comparison)
    
    # Optimisation de K si demandé
    if args.optimize_k or args.k == 0:
        k_range = range(args.k_range_min, args.k_range_max + 1, args.k_range_step)
        best_k, k_scores = optimize_k(
            X_train_norm, y_train, 
            k_range=k_range, 
            cv_folds=args.cv_folds
        )
        
        # Sauvegarde du graphique d'optimisation
        if args.save_k_plot:
            plot_k_optimization(list(k_range), k_scores, args.save_k_plot)
        
        # Utiliser le meilleur K trouvé
        final_k = best_k
    else:
        final_k = args.k
    
    # Entraînement et évaluation du modèle final
    model, y_pred, metrics = train_and_evaluate_knn(
        X_train_norm, y_train, X_test_norm, y_test,
        k=final_k,
        weights=args.weights,
        metric=args.metric,
        algorithm=args.algorithm,
        n_jobs=-1
    )
    
    # Résumé final
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ FINAL")
    print(f"{'='*70}")
    print(f"Configuration optimale:")
    print(f"  K: {final_k}")
    print(f"  Weights: {args.weights}")
    print(f"  Metric: {args.metric}")
    print(f"  Algorithm: {args.algorithm}")
    print(f"\nPerformances:")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  Temps entraînement: {metrics['train_time']:.2f}s")
    print(f"  Temps prédiction: {metrics['predict_time']:.2f}s")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()