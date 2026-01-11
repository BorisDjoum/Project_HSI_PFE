import numpy as np
from loader import *
from tkinter import filedialog
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import convolve
from distances import *
from spectral import envi
import matplotlib.pyplot as plt
import numpy as npy

from HSI_distances import dist_KLPD

# Functions ###########################################################################################################

def Gram_Matrix( S ):

    L = len(S)
    G = np.zeros( (L,L) )

    for x in range(0,L):
        for y in range(0,L):

            G[x,y] = np.dot( S[x], S[y] )

    return G

########################################################################################################################

def Compute_gradients(gradients, position):

    x, y = position
    Gx, Gy = gradients

    dI = np.zeros( (Gx.shape[2], 2) )

    # Extraire les gradients à la position donnée pour tous les canaux
    for k in range( Gx.shape[2] ):
        dI[k,0] = Gx[x,y,k]
        dI[k,1] = Gy[x,y,k]

    # Combiner Dx et Dy
    return dI

##########################################################################################################################

def Compute_gradients_parallel(I, positions):
    with ProcessPoolExecutor() as executor:
        # Calcul parallèle des gradients pour toutes les positions
        results = list(executor.map(lambda pos: Compute_gradients(I, pos), positions))
    return results

#########################################################################################################################

def Compute_Correlation_Matrix(gradients, G, position):
    x, y = position

    Nabla_I = Compute_gradients(gradients, (x,y))

    return Nabla_I.T @ G @ Nabla_I

#########################################################################################################################

def Compute_Correlation_Matrix_parallel(I, G, positions):
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(lambda pos: Compute_Correlation_Matrix(I,G, pos), positions))
    return results

#########################################################################################################################


# def Compute_Magnitude_and_Direction(gradients, G, position):
#     M = Compute_Correlation_Matrix(gradients, G, position)

#     tr = np.trace(M)
#     det = np.linalg.det(M)

#     A = np.sqrt( (tr**2) - (4 * det ) )

#     l1 = (1/2) * ( tr + A )
#     l2 = (1/2) * ( tr - A )

#     lplus = max(l1, l2)
#     lmoin = min(l1, l2)

#     numerator   = lplus - M[0,0]
#     denominator = ( 2 * lplus ) - tr
#     argument    = numerator / denominator

#     if l1 != l2:
#         theta = np.sign(M[0,1]) * np.sqrt( np.arcsin( argument ) )
#     else:
#         theta = 0.0

#     T = np.sqrt( np.abs(lplus + lmoin) ) / np.sqrt(2)

#     return T, theta

def Compute_Magnitude_and_Direction(gradients, G, position):
    M = Compute_Correlation_Matrix(gradients, G, position)

    tr = np.trace(M)
    det = np.linalg.det(M)

    # 1. Protection du discriminant : max(0.0, ...) pour éviter np.sqrt d'un négatif
    discriminant = (tr**2) - (4 * det)
    A = np.sqrt(max(0.0, discriminant)) 

    l1 = (1/2) * ( tr + A )
    l2 = (1/2) * ( tr - A )

    lplus = max(l1, l2)
    lmoin = min(l1, l2)

    # --- 2. Protection contre l'instabilité (dénominateur ≈ 0) ---
    
    # Si les deux valeurs propres sont très proches ou si le discriminant est zéro, 
    # la direction n'est pas bien définie. Nous utilisons une protection numérique.
    if np.abs(l1 - l2) < 1e-9: # 1e-9 est le seuil de tolérance flottante
        theta = 0.0
        T = np.sqrt( np.abs(lplus + lmoin) ) / np.sqrt(2)
        return T, theta 

    # --- Calcul de l'angle pour lplus != lmoin ---
    numerator   = lplus - M[0,0]
    denominator = ( 2 * lplus ) - tr
    
    # 3. Protection de l'argument de l'arcsin : Clamper la valeur dans [-1, 1]
    argument = numerator / denominator
    argument = np.clip(argument, -1.0, 1.0) 

    # Calcul de la Direction (theta)
    arcsin_val = np.arcsin(argument)
    if arcsin_val < 0:  # arcsin peut être négatif
        theta = 0.0
    else:
        theta = np.sign(M[0,1]) * np.sqrt(arcsin_val)
        
    # 4. Protection finale contre les NaN générés par la formule complexe (sqrt(arcsin(...)))
    if np.isnan(theta):
         theta = 0.0 

    # Calcul de la Magnitude (T)
    T = np.sqrt( np.abs(lplus + lmoin) ) / np.sqrt(2)
    
    # 5. Si la magnitude est toujours NaN, retourner zéro (protection finale)
    if np.isnan(T):
        T = 0.0

    

    return T, theta

#########################################################################################################################

def Compute_ghost_KLPD(patch, refs, ssf=None, dist= dist_KLPD):
    width, height, channels = patch.shape

    if ssf is None:
        Gram = np.eye(channels)
    else:
        Gram = Gram_Matrix(ssf)

    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    # Initialize arrays to store results
    Gx = np.zeros_like(patch)
    Gy = np.zeros_like(patch)

    for k in range(patch.shape[2]):
        Gx[:, :, k] = convolve(patch[:, :, k], sobel_x)
        Gy[:, :, k] = convolve(patch[:, :, k], sobel_y)

    Magnitude = np.zeros((width, height))
    Direction = np.zeros((width, height))

    for x in range(width):
        for y in range(height):
            Magnitude[x, y], Direction[x, y] = Compute_Magnitude_and_Direction((Gx, Gy), Gram, (x, y))

    if len(refs) > 1:
        Shape = np.zeros((width, height, len(refs)))
        Energy = np.zeros((width, height, len(refs)))

        reshaped_patch = patch.reshape((width * height, channels))

        # Spectral comparison
        for idx, ref in enumerate(refs):
            R = dist(ref, reshaped_patch)
            if isinstance(R, (tuple, list)):
                G = R[0].reshape((width, height))
                W = R[1].reshape((width, height))

            Shape[:, :, idx] = G
            Energy[:, :, idx] = W

        # Concatenate all attributes into a single output array
        result = np.concatenate(
            [Magnitude[:, :, np.newaxis], 
            Direction[:, :, np.newaxis], 
            Shape, 
            Energy], 
            axis=2
        )

        return result
    else:
        Shape   = np.zeros((width, height))
        Energy  = np.zeros((width, height))

        reshaped_patch = patch.reshape((width * height, channels))

        # Spectral comparison
        G, W = dist_KLPD(refs[0], reshaped_patch)
        G = G.reshape((width, height))
        W = W.reshape((width, height))

        Shape[:, :] = G
        Energy[:, :] = W

        # Concatenate all attributes into a single output array
        result = np.stack([
            Direction.ravel(),
            Magnitude.ravel(),
            Shape.ravel(),
            Energy.ravel()],
        axis=1)

        return result.T
    
def Compute_ghost(patch, refs, ssf=None, dist= dist_Jeffreys):
    width, height, channels = patch.shape

    if ssf is None:
        Gram = np.eye(channels)
    else:
        Gram = Gram_Matrix(ssf)

    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    # Initialize arrays to store results
    Gx = np.zeros_like(patch)
    Gy = np.zeros_like(patch)

    for k in range(patch.shape[2]):
        Gx[:, :, k] = convolve(patch[:, :, k], sobel_x)
        Gy[:, :, k] = convolve(patch[:, :, k], sobel_y)

    Magnitude = np.zeros((width, height))
    Direction = np.zeros((width, height))

    for x in range(width):
        for y in range(height):
            Magnitude[x, y], Direction[x, y] = Compute_Magnitude_and_Direction((Gx, Gy), Gram, (x, y))

    if len(refs) > 1:
        distances = np.zeros((width, height, len(refs)))

        reshaped_patch = patch.reshape((width * height, channels))

        # Spectral comparison
        for idx, ref in enumerate(refs):
            R = dist(ref, reshaped_patch)

            distances[:, :, idx] = R


        # Concatenate all attributes into a single output array
        result = np.concatenate(
            [Magnitude[:, :, np.newaxis], 
            Direction[:, :, np.newaxis], 
            distances.ravel()], 
            axis=2
        )

        return result
    else:
        distances   = np.zeros((width, height))
        reshaped_patch = patch.reshape((width * height, channels))

        # Spectral comparison
        R = dist(refs[0], reshaped_patch)
        R = R.reshape((width, height))

        distances[:, :] = R

        # Concatenate all attributes into a single output array
        result = np.stack([
            Direction.ravel(),
            Magnitude.ravel(),
            distances.ravel()],
        axis=1)

        return result.T


#########################################################################################################################

def Compute_all_ghost(patches, References, dist = dist_KLPD):
    
    GHOST = None

    for patch in patches:

        if GHOST is None:
            GHOST = Compute_ghost( patch, References, dist = dist )
        else:
            GHOST = np.vstack( ( GHOST, Compute_ghost( patch, References ) ) )

    return GHOST

### TEST ###

if __name__ == '__main__':
    BAND_START = 30  # Indice de bande pour 410 nm
    BAND_END = 465
    # Fonction pour générer une gaussienne
    def generate_gaussian(waves, center, fwhm, amplitude):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return amplitude * np.exp(-0.5 * ((waves - center) / sigma) ** 2)

    # --- Path to your ENVI image (.hdr or .bin) ---
    npy_path = filedialog.askopenfilename(title="NPY file of your image")

    # --- Open the image ---
    # img = npy.load(npy_path)
    I = np.load(npy_path)
    print(I.shape)
    # --- Load the image data as a NumPy array ---
    # I = img.load()
    I = I[:708, :, :]  # Ensure L_min lines and band selection

    print(I.shape)
    width, height, channels = I.shape

    Gram = np.eye(channels, channels, dtype= float)

    width, height, channels = I.shape

    # bands_sensitivity = []

    # for k in range( len(wavelengths) ):
    #     bands_sensitivity.append( generate_gaussian(np.arange(start = 350, stop= 850, step= 1), wavelengths[k], 5, 1.0) )

    Gram = np.eye(channels, channels, dtype=np.float32) #Gram_Matrix(bands_sensitivity)

    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    # Initialize arrays to store results
    Gx = np.zeros_like(I)
    Gy = np.zeros_like(I)
    print(Gx.shape)
    for k in range( I.shape[2] ):
        k_band = I[:, :, k].reshape((width, height))
        Gx[:, :, k] = convolve(k_band, sobel_x)
        Gy[:, :, k] = convolve(k_band, sobel_y)

    # Magnitude = np.zeros( ( width, height ) )
    # Direction = np.zeros( ( width, height ) )

    Gx2 = np.power(Gx,2)
    Gy2 = np.power(Gy,2)

    print(Gx2.shape)
    print(Gy2.shape)

    Gx2 = np.mean( Gx2, axis=2 )
    Gy2 = np.mean( Gy2, axis=2 )
    print(Gx2.shape)
    print(Gy2.shape)
    print(minimum := np.min(Gx2))
    print(maximum := np.max(Gx2))

    # récupération de toutes les valeur des longueurs d'onde 
    wavelengths_list = np.arange(BAND_START, BAND_END)
    wavelengths = np.array(wavelengths_list)

    #Specral comparison
    

    ref = generate_gaussian(wavelengths, 450, 40, 0.5) + 0.1
    print (ref.shape)
    print(I.reshape( ( width * height,  channels) ).shape)
    G, W = dist_KLPD( ref, I.reshape( ( width * height,  channels) ) )

    G = G.reshape((width, height))
    W = W.reshape((width, height))

    plt.hist(Gx2.ravel(), bins=800, color='blue', alpha=0.7)

    # 1. Préparation de la figure (2 lignes, 2 colonnes)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Sélection d'une bande pour Gx et Gy (ex: la bande centrale)
    band_idx = channels // 2

    # --- Affichage de Gx ---
    percentile = np.percentile(Gx2, 99)
    Gx2 = np.log(Gx2)
    norm_Gx = (Gx2 - np.min(Gx2)) / (np.max(Gx2) - np.min(Gx2))
    Gy2 = np.log(Gy2)
    norm_Gy = (Gy2 - np.min(Gy2)) / (np.max(Gy2) - np.min(Gy2))
    im1 = axes[0, 0].imshow(norm_Gx, cmap='magma')
    axes[0, 0].set_title(f"Gradient Gx (Bande {band_idx})")
    fig.colorbar(im1, ax=axes[0, 0])

    # --- Affichage de Gy ---
    im2 = axes[0, 1].imshow(norm_Gy, cmap='magma')
    axes[0, 1].set_title(f"Gradient Gy (Bande {band_idx})")
    fig.colorbar(im2, ax=axes[0, 1])

    # --- Affichage de G (Distance/Similitude) ---
    norm_G = (G - np.min(G)) / (np.max(G) - np.min(G))
    im3 = axes[1, 0].imshow(norm_G, cmap='viridis')
    axes[1, 0].set_title("Matrice G (KLPD)")
    fig.colorbar(im3, ax=axes[1, 0])

    # --- Affichage de W ---
    norm_W = (W - np.min(W)) / (np.max(W) - np.min(W))
    im4 = axes[1, 1].imshow(norm_W, cmap='plasma')
    axes[1, 1].set_title("Matrice W")
    fig.colorbar(im4, ax=axes[1, 1])

    # Ajustement automatique de l'espacement
    plt.tight_layout()
    plt.show()

    

    # for x in range(I.shape[0]):
    #     for y in range(I.shape[1]):
    #         Magnitude[x,y], Direction[x,y] = Compute_Magnitude_and_Direction( (Gx, Gy), Gram, (x,y) )
    
    # # --- ÉTAPE 1 : Calculer le nombre de bacs (Loi de Sturges) ---
    # # Nous utilisons Magnitude.size car c'est la taille commune de tous les tableaux
    # bins = int( np.ceil( 1 + 3.322*np.log10(Magnitude.size) ) ) 

    # # --- ÉTAPE 2 : Fonction de Quantification (Mapping de Densité) ---
    # def map_to_density(data_array, num_bins):
    #     """Calcule l'histogramme de densité et mappe chaque point à la densité de son bac."""
        
    #     # Calcule les densités ([0]) et les bords des bacs ([1])
    #     densities, edges = np.histogram(data_array, num_bins, density=True)
        
    #     # np.digitize mappe chaque point de donnée au bon indice de bac (en utilisant les bords)
    #     # L'argument right=True assure que les valeurs maximales sont bien placées
    #     indices = np.digitize(data_array, edges[:-1], right=True)
        
    #     # La première valeur (indice 0) est réservée aux valeurs < min(edges).
    #     # Dans certains cas, nous devons décaler l'indice si np.digitize est 1-indexé.
    #     # Pour la plupart des utilisations, 'indices' donne l'index du bac.
    #     # Nous utilisons np.clip pour garantir que l'indice ne dépasse pas la taille de densities (bins - 1)
        
    #     # Indices valides sont de 0 à bins-1.
    #     indices = np.clip(indices, 0, num_bins - 1) 
        
    #     # Le résultat est le tableau des données d'origine, mais avec chaque valeur remplacée 
    #     # par la densité du bac correspondant.
    #     return densities[indices]


    # # --- ÉTAPE 3 : APPLICATION DE LA CORRECTION ---
    # print(Magnitude)
    # print("--------------------------------------------")
    # print(Direction)
    # print(f"Quantification des caractéristiques GHOST utilisant {bins} bacs.")

    # num_bins = int( np.ceil( 1 + 3.322*np.log10(Magnitude.size) ) )
    # hist_T, _ = np.histogram(Magnitude, bins=num_bins, range=(np.min(Magnitude), np.max(Magnitude)), density=True)
    # hist_theta, _ = np.histogram(Direction, bins=num_bins, range=(np.min(Direction), np.max(Direction)), density=True)
    # hist_G, _ = np.histogram(G, bins=num_bins, range=(np.min(G), np.max(G)), density=True)
    # hist_W, _ = np.histogram(W, bins=num_bins, range=(np.min(W), np.max(W)), density=True)

    # Le reste du pipeline peut maintenant utiliser ces nouvelles caractéristiques quantifiées.

    # descriptor = np.stack( ( hist_T.ravel(), hist_theta.ravel(), hist_G.ravel(), hist_W.ravel() ), axis=1 )
    # print(
    #     f"Descriptor shape: {descriptor.shape}"
        
    # )

    # import matplotlib.pyplot as plt
    # import matplotlib.colors as colors
    # from matplotlib.colors import hsv_to_rgb
    # from matplotlib import rc

    # rc('font', size=15)          # Default text size
    # rc('axes', titlesize=20)     # Title font size
    # rc('axes', labelsize=15)     # Axis label font size
    # rc('xtick', labelsize=12)    # X-tick label font size
    # rc('ytick', labelsize=12)    # Y-tick label font size
    # rc('legend', fontsize=12)    # Legend font size

    # # Normalize Direction and Magnitude
    # hue = (Direction + np.pi / 2) / (np.pi)     # Normalize Direction to [0, 1]
    # value = Magnitude / np.max(Magnitude)       # Normalize Magnitude to [0, 1]

    # # Create HSV and convert to RGB
    # hsv = np.stack((hue, np.ones_like(hue), value), axis=-1)  # (H, S=1, V)
    # rgb_hue_magnitude = hsv_to_rgb(hsv)

    # # Création de la figure en 2x2
    # fig, axes = plt.subplots(2, 3, figsize=(10, 10))

    # # Image 1 : Direction
    # im1 = axes[0, 0].imshow( Direction, cmap='hsv', origin='upper', vmax = np.pi / 2, vmin = -np.pi / 2)
    # axes[0, 0].set_title('Direction')
    # fig.colorbar(im1, ax=axes[0, 0], label='Angle')

    # # Image 2 : Magnitude
    # im2 = axes[0, 1].imshow(Magnitude, cmap='viridis', origin='upper')
    # axes[0, 1].set_title('Magnitude')
    # fig.colorbar(im2, ax=axes[0, 1], label='Magnitude')

    # # Image 3 : Shape Difference
    # im3 = axes[0, 2].imshow(G, cmap='turbo', origin='upper')
    # axes[0, 2].set_title('Shape difference')
    # fig.colorbar(im3, ax=axes[0, 2], label='Shape Difference')

    # # Image 4 : Hue modifié par Magnitude
    # axes[1, 0].imshow(rgb_hue_magnitude)
    # axes[1, 0].set_title('Hue & Magnitude')

    # # Image 5 : RGB
    # axes[1, 1].imshow(rgb)
    # axes[1, 1].set_title('RGB')

    # # Image 6 : Energie Difference
    # im4 = axes[1, 2].imshow(W, cmap='turbo', origin='upper')
    # axes[1, 2].set_title('Energie difference')
    # fig.colorbar(im4, ax=axes[1, 2], label='Energie Difference')

    # # Ajustement et affichage
    # plt.show()
    # plt.close()

    # plt.imshow(G + W, cmap='inferno', origin='upper')
    # plt.show()
    # plt.close()

    # ###########################################################################################################################

    # # Création de la figure en 2x2
    # fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # # Image 1 : Direction
    # im1 = axes[0, 0].imshow( Direction, cmap='hsv', origin='upper', vmax = np.pi / 2, vmin = -np.pi / 2)
    # axes[0, 0].set_title('Direction')
    # fig.colorbar(im1, ax=axes[0, 0], label='Direction')

    # # Image 2 : Magnitude
    # im2 = axes[0, 1].imshow(Magnitude, cmap='inferno', origin='upper')
    # axes[0, 1].set_title('Magnitude')
    # fig.colorbar(im2, ax=axes[0, 1], label='Magnitude')

    # # Image 3 : Shape Difference
    # im3 = axes[1, 0].imshow(G, cmap='inferno', origin='upper')
    # axes[1, 0].set_title(r'$\Delta_{Forme}$')
    # fig.colorbar(im3, ax=axes[1, 0], label=r'$KLPD_{Forme}$')

    # # Image 4 : Energie Difference
    # im4 = axes[1, 1].imshow(W, cmap='inferno', origin='upper')
    # axes[1, 1].set_title(r'$\Delta_{Energie}$')
    # fig.colorbar(im4, ax=axes[1, 1], label=r'$KLPD_{Energie}$')

    # # Ajustement et affichage
    # plt.show()
    # plt.close()

    # ###########################################################################################################################

    # # Création de la figure
    # fig, axes = plt.subplots(2, 1, figsize=(5, 10))  # 2 ligne, 1 colonnes

    # N = 128

    # # Histogramme 2D pour les tableaux 1 et 2
    # h1 = axes[0].hist2d(Direction.flatten(), Magnitude.flatten(), bins=N, cmap='turbo', norm = colors.LogNorm())
    # axes[0].set_title('Direction / Magnitude')
    # axes[0].set_xlabel('Direction')
    # axes[0].set_ylabel('Magnitude')
    # fig.colorbar(h1[3], ax=axes[0], label='Densité')

    # # Histogramme 2D pour les tableaux 3 et 4
    # h2 = axes[1].hist2d(G.flatten(), W.flatten(), bins=N, cmap='turbo', norm = colors.LogNorm())
    # axes[1].set_title('Forme / Energie')
    # axes[1].set_xlabel('Forme')
    # axes[1].set_ylabel('Energie')
    # fig.colorbar(h2[3], ax=axes[1], label='Densité')

    # # Ajustement et affichage
    # plt.tight_layout()
    # plt.show()