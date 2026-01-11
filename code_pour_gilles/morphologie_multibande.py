# Morphologie multibande avec backend CPU/GPU et fallback,
# conversion d'entrée/sortie pour préserver le type original (numpy <-> cupy).

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

import numpy as np

# --- Gestion du backend (numpy / cupy) -------------------------------------

def _validate_force_device(force_device):
    """
    Valide le paramètre force_device.
    Accepte None, 'CPU' ou 'GPU' (strings exacts). Lève ValueError sinon.
    """
    if force_device is None:
        return
    if not isinstance(force_device, str):
        raise ValueError("force_device doit être None, 'CPU' ou 'GPU'.")
    if force_device not in ('CPU', 'GPU'):
        raise ValueError("force_device doit être None, 'CPU' ou 'GPU'.")

def _get_backend_for_computation(input_arr, force_device=None):
    """
    Détermine le backend (module numpy ou cupy) à utiliser pour le calcul, selon :
      - le type de input_arr (numpy.ndarray ou cupy.ndarray)
      - ou force_device : None / 'CPU' / 'GPU'
    Retourne : (xp, using_cupy, orig_is_cupy) où xp est np ou cp, using_cupy bool
             et orig_is_cupy indique si l'entrée était initialement cupy.
    """
    _validate_force_device(force_device)
    orig_is_cupy = _HAS_CUPY and isinstance(input_arr, cp.ndarray)

    if force_device is None:
        # on respecte le type d'entrée
        if orig_is_cupy:
            return cp, True, True
        else:
            return np, False, False
    else:
        if force_device == 'CPU':
            return np, False, orig_is_cupy
        else:  # force_device == 'GPU'
            if not _HAS_CUPY:
                raise RuntimeError("CuPy non installé mais force_device='GPU' demandé.")
            return cp, True, orig_is_cupy

def _as_backend(x, xp):
    """
    Convertit x vers le backend xp (numpy ou cupy).
    Si x est déjà dans le bon backend, retourne x tel quel.
    """
    if xp is np:
        if _HAS_CUPY and isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        return x
    else:
        # xp is cupy
        if not _HAS_CUPY:
            raise RuntimeError("CuPy non disponible")
        if isinstance(x, cp.ndarray):
            return x
        return cp.asarray(x)

# --- Utilitaires pour l'élément structurant --------------------------------

def _se_offsets(se):
    """
    Prend un SE (2D array-like bool/0-1) et retourne la liste d'offsets (dy,dx)
    relativement au centre du SE. Utilise numpy pour la manipulation du SE.
    """
    se_arr = np.asarray(se)
    if se_arr.ndim != 2:
        raise ValueError("L'élément structurant doit être une matrice 2D.")
    h, w = se_arr.shape
    cy, cx = h // 2, w // 2
    offsets = []
    for i in range(h):
        for j in range(w):
            if se_arr[i, j]:
                offsets.append((i - cy, j - cx))
    return offsets, (cy, cx)

# --- Padding sécurisé (évite le wrap-around de np.roll) ---------------------

def _pad_for_offsets(img, se_shape, xp, pad_cval):
    """
    Pad l'image (H,W,B) selon la taille du SE afin de permettre des extractions
    par translation sans wrap-around. pad_cval sert comme valeur de remplissage.
    """
    h, w = se_shape
    cy, cx = h // 2, w // 2
    pad_top, pad_bottom = cy, h - 1 - cy
    pad_left, pad_right = cx, w - 1 - cx
    pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    return xp.pad(img, pad_width, mode='constant', constant_values=pad_cval), (pad_top, pad_left)

# --- Ordonnancements modulaires --------------------------------------------

_orderings = {}

def ordering_marginal(stack, op):
    """
    Ordonnancement marginal : applique max/min indépendamment par canal.
    stack : array de forme (n_offsets, H, W, B)
    op : 'dilation' ou 'erosion'
    Retour : (H, W, B)
    """
    if op == 'dilation':
        return stack.max(axis=0)
    elif op == 'erosion':
        return stack.min(axis=0)
    else:
        raise ValueError("op doit être 'dilation' ou 'erosion'.")

def ordering_norm(stack, op):
    """
    Ordonnancement par norme L2 : pour chaque pixel, on calcule la norme L2
    du vecteur pour chaque offset, puis on prend le vecteur ayant la plus grande
    (dilation) ou la plus petite (erosion) norme.
    stack : (n, H, W, B)
    """
    # détecter le backend via le type de stack
    xp = cp if (_HAS_CUPY and isinstance(stack, cp.ndarray)) else np
    norms = xp.sqrt(xp.sum(stack * stack, axis=-1))  # (n, H, W)
    if op == 'dilation':
        idx = xp.argmax(norms, axis=0)  # (H, W)
    elif op == 'erosion':
        idx = xp.argmin(norms, axis=0)
    else:
        raise ValueError("op doit être 'dilation' ou 'erosion'.")

    n, H, W, B = stack.shape
    stack2 = stack.reshape(n, H * W, B)
    idx1 = idx.ravel()  # (H*W,)
    positions = xp.arange(H * W)
    chosen = stack2[idx1, positions, :]  # (H*W, B)
    return chosen.reshape(H, W, B)

_orderings['Marginal'] = ordering_marginal
_orderings['Norm'] = ordering_norm

def add_ordering(name, func):
    """
    Ajoute un nouvel ordonnancement.
    func doit être une fonction (stack, op) -> (H,W,B) où stack a pour forme (n, H, W, B).
    """
    if not callable(func):
        raise ValueError("func doit être callable")
    _orderings[name] = func

# --- Opérations morphologiques vectorielles -------------------------------

def _vector_morph(image, se, op='dilation', ordering='Marginal', force_device=None):
    """
    Opération morphologique multibande (dilation ou erosion) avec gestion
    explicite du device (CPU/GPU) et conversion de sortie pour préserver le type
    d'entrée.
    - image : array (H, W, B), numpy ou cupy
    - se : 2D array-like binaire (élément structurant)
    - op : 'dilation' ou 'erosion'
    - ordering : clé dans _orderings (ex: 'Marginal', 'Norm')
    - force_device : None (respecte type d'entrée), 'CPU' ou 'GPU'
    Retour : image transformée (même type que l'entrée originale)
    """
    if op not in ('dilation', 'erosion'):
        raise ValueError("op doit être 'dilation' ou 'erosion'.")

    # déterminer backend de calcul et si l'entrée était cupy
    xp, using_cupy, orig_is_cupy = _get_backend_for_computation(image, force_device=force_device)

    # convertir l'image vers le backend choisi pour effectuer le calcul
    img = _as_backend(image, xp)

    if img.ndim != 3:
        raise ValueError("L'image doit être multibande avec shape (H, W, B).")

    H, W, B = img.shape
    offsets, (cy, cx) = _se_offsets(se)

    # travailler en float pour gérer +/- inf lors du padding
    if xp.issubdtype(img.dtype, xp.floating):
        imgf = img.astype(xp.float64)
    else:
        imgf = img.astype(xp.float64)

    # valeur de pad neutre selon l'opération
    if op == 'dilation':
        pad_val = -xp.inf
    else:
        pad_val = xp.inf

    padded, (pad_top, pad_left) = _pad_for_offsets(imgf, np.asarray(se).shape, xp, pad_val)

    patches = []
    for (dy, dx) in offsets:
        y0 = pad_top + dy
        x0 = pad_left + dx
        patch = padded[y0:y0 + H, x0:x0 + W, :]
        patches.append(patch)

    stack = xp.stack(patches, axis=0)  # (n_offsets, H, W, B)

    if ordering not in _orderings:
        raise KeyError(f"Ordonnancement '{ordering}' inconnu. Méthodes disponibles: {_orderings.keys()}")
    result = _orderings[ordering](stack, op)  # result dans le backend xp

    # reconvertir le résultat vers le type d'entrée original
    if orig_is_cupy:
        # entrée originale était cupy -> on renvoie cupy
        if not (_HAS_CUPY):
            # cas improbable : entrée reportée cupy mais _HAS_CUPY False
            raise RuntimeError("Erreur interne: entrée détectée comme cupy mais CuPy non disponible.")
        # si le calcul a été fait sur numpy, convertir en cupy
        if xp is np:
            result_out = cp.asarray(result)
        else:
            result_out = result
    else:
        # entrée originale était numpy -> on renvoie numpy
        if xp is cp:
            result_out = cp.asnumpy(result)
        else:
            result_out = result

    # préserver dtype d'origine (avec clipping pour entiers)
    dtype_orig = image.dtype
    if np.issubdtype(dtype_orig, np.integer):
        info = np.iinfo(dtype_orig)
        # result_out peut être cupy ou numpy ; utiliser appropriate clip
        if isinstance(result_out, np.ndarray):
            result_out = np.clip(result_out, info.min, info.max).astype(dtype_orig)
        else:
            result_out = cp.clip(result_out, info.min, info.max).astype(dtype_orig)
    else:
        # cast flottant vers dtype d'origine (garde précision si flottant)
        if isinstance(result_out, cp.ndarray):
            result_out = result_out.astype(dtype_orig)
        else:
            result_out = result_out.astype(dtype_orig)

    return result_out

# Fonctions publiques : dilation, erosion, opening, closing ------------------

def dilation(image, se, ordering='Marginal', force_device=None):
    """
    Dilation multibande.
    force_device : None (respecte type d'entrée), 'CPU' ou 'GPU'.
    """
    return _vector_morph(image, se, op='dilation', ordering=ordering, force_device=force_device)

def erosion(image, se, ordering='Marginal', force_device=None):
    """
    Erosion multibande.
    """
    return _vector_morph(image, se, op='erosion', ordering=ordering, force_device=force_device)

def opening(image, se, ordering='Marginal', force_device=None):
    """
    Opening = erosion suivie de dilation.
    """
    e = erosion(image, se, ordering=ordering, force_device=force_device)
    return dilation(e, se, ordering=ordering, force_device=force_device)

def closing(image, se, ordering='Marginal', force_device=None):
    """
    Closing = dilation suivie d'erosion.
    """
    d = dilation(image, se, ordering=ordering, force_device=force_device)
    return erosion(d, se, ordering=ordering, force_device=force_device)

# --- Petit test synthétique ------------------------------------------------

if __name__ == "__main__":
    H, W, B = 8, 10, 3
    img = np.zeros((H, W, B), dtype=np.uint8)
    img[2:6, 3:7, 0] = 200
    img[3:5, 4:6, 1] = 150
    img[1:4, 1:4, 2] = 100

    se = np.ones((3,3), dtype=bool)

    print("Type d'entrée :", type(img))
    d = dilation(img, se, ordering='Marginal', force_device=None)
    print("Dilation (respect type entrée) - type sortie :", type(d), " somme:", d.sum())

    # Forcer CPU (même si on était déjà sur CPU) -> doit retourner numpy
    d_cpu = dilation(img, se, ordering='Norm', force_device='CPU')
    print("Dilation force CPU - type sortie :", type(d_cpu))

    # Si CuPy dispo, tester conversion aller-retour
    if _HAS_CUPY:
        img_gpu = cp.asarray(img)
        print("Type d'entrée GPU :", type(img_gpu))
        # forcer calcul sur GPU et renvoyer GPU
        d_gpu = dilation(img_gpu, se, ordering='Norm', force_device=None)
        print("Dilation GPU respect type - type sortie :", type(d_gpu))
        # forcer calcul sur CPU mais renvoyer GPU (entrée cupy)
        d_gpu_cpucalc = dilation(img_gpu, se, ordering='Norm', force_device='CPU')
        print("Dilation entrée GPU, calcul CPU, sortie GPU - type sortie :", type(d_gpu_cpucalc))
