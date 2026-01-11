import numpy as np
from loader import *
from tkinter import filedialog
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import convolve
from spectral import envi

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

def Compute_Magnitude_and_Direction(gradients, G, position):
    M = Compute_Correlation_Matrix(gradients, G, position)

    tr = np.trace(M)
    det = np.linalg.det(M)

    A = np.sqrt( (tr**2) - (4 * det ) )

    l1 = (1/2) * ( tr + A )
    l2 = (1/2) * ( tr - A )

    lplus = max(l1, l2)
    lmoin = min(l1, l2)

    numerator   = lplus - M[0,0]
    denominator = ( 2 * lplus ) - tr
    argument    = numerator / denominator

    if l1 != l2:
        theta = np.sign(M[0,1]) * np.sqrt( np.arcsin( argument ) )
    else:
        theta = 0.0

    T = np.sqrt( np.abs(lplus + lmoin) ) / np.sqrt(2)

    return T, theta

### TEST ###

if __name__ == '__main__':

    # --- Path to your ENVI image (.hdr or .bin) ---
    hdr_path = filedialog.askopenfilename(title="HDR file of your image")
    bin_path = filedialog.askopenfilename(title="BIN file of your image")

    # --- Open the image ---
    img = envi.open(hdr_path, bin_path)

    # --- Load the image data as a NumPy array ---
    I = img.load()

    print(I.shape)
    width, height, channels = I.shape

    Gram = np.eye(channels, channels, dtype= float)

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

    for k in range( I.shape[2] ):
        k_band = I[:, :, k].reshape((width, height))
        Gx[:, :, k] = convolve(k_band, sobel_x)[:,:,np.newaxis]
        Gy[:, :, k] = convolve(k_band, sobel_y)[:,:,np.newaxis]

    Magnitude = np.zeros( ( width, height ) )
    Direction = np.zeros( ( width, height ) )

    for x in tqdm(range(I.shape[0])):
        for y in range(I.shape[1]):
            Magnitude[x,y], Direction[x,y] = Compute_Magnitude_and_Direction( (Gx, Gy), Gram, (x,y) )

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from matplotlib.colors import hsv_to_rgb
    from matplotlib import rc

    rc('font', size=15)          # Default text size
    rc('axes', titlesize=20)     # Title font size
    rc('axes', labelsize=15)     # Axis label font size
    rc('xtick', labelsize=12)    # X-tick label font size
    rc('ytick', labelsize=12)    # Y-tick label font size
    rc('legend', fontsize=12)    # Legend font size

    # Normalize Direction and Magnitude
    hue = (Direction + np.pi / 2) / (np.pi)     # Normalize Direction to [0, 1]
    value = Magnitude / np.max(Magnitude)       # Normalize Magnitude to [0, 1]

    # Create HSV and convert to RGB
    hsv = np.stack((hue, np.ones_like(hue), value), axis=-1)  # (H, S=1, V)
    rgb_hue_magnitude = hsv_to_rgb(hsv)

    # Création de la figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    # Image 1 : Direction
    im1 = axes[0].imshow( Direction, cmap='hsv', origin='upper', vmax = np.pi / 2, vmin = -np.pi / 2)
    axes[0].set_title('Direction')
    fig.colorbar(im1, ax=axes[0], label='Angle')

    # Image 2 : Magnitude
    im2 = axes[1].imshow(Magnitude, cmap='viridis', origin='upper')
    axes[1].set_title('Magnitude')
    fig.colorbar(im2, ax=axes[1], label='Magnitude')

    plt.show()
    plt.close()

    ###########################################################################################################################

    # Création de la figure
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))  # 2 ligne, 1 colonnes

    N = 128

    # Histogramme 2D pour les tableaux 1 et 2
    h1 = ax.hist2d(Direction.flatten(), Magnitude.flatten(), bins=N, cmap='turbo', norm = colors.LogNorm())
    ax.set_title('Direction / Magnitude')
    ax.set_xlabel('Direction')
    ax.set_ylabel('Magnitude')
    fig.colorbar(h1[3], ax=ax, label='Densité')

    # Ajustement et affichage
    plt.tight_layout()
    plt.show()