import numpy as np
import spectral as spy
import os
from glob import glob
from spectral import envi
from scipy.ndimage import convolve
from ghost import Gram_Matrix, Compute_Magnitude_and_Direction
from HSI_distances import dist_KLPD


DATASET_REFLEC_DIR = 'dataset_reflec_test'

# Constantes globales nécessaires à l'ensemble du script
L_min = 708      # Nombre de lignes minimum (pour le redimensionnement spatial)
BAND_START = 30  # Indice de bande pour 410 nm
BAND_END = 465   # Indice de bande pour couper l'extrême fin (avant 1006 nm)
DATASET_HIST_DIR = 'dataset_hist2'
LOG_FILE = 'ghost_hist_processed_log.txt' # Nouveau log pour cette étape
CHECKPOINT_FILE = 'ghost_hist_checkpoint.npz' # Nouveau checkpoint

# --- À placer dans votre fichier data_processor.py ---

# NOTE: J'ai ajouté min_lines (L_min) à la signature
def process_single_image_and_save(hdr_file, dark_ref_hdr, white_ref_hdr, label, min_lines):
    """
    WORKER : Calibre un fichier, redimensionne le cube de réflectance 
    et le sauvegarde immédiatement en NPY.
    """
    
    reflectance_cube = load_and_calibrate(hdr_file, dark_ref_hdr, white_ref_hdr)
    reflectance_cube = reflectance_cube[:, :, BAND_START:BAND_END]
    
    if reflectance_cube is not None:
        
        # 1. RÉDUCTION DU NOMBRE DE LIGNES
        # Le cube de réflectance a la forme (Lignes, Colonnes, Bandes). 
        # Nous prenons les L_min premières lignes.
        current_lines = reflectance_cube.shape[0]
        
        if current_lines >= min_lines:
            reflectance_cube = reflectance_cube[:min_lines, :, :]
            print(f"Redimensionnement de {current_lines} lignes à {min_lines} pour {os.path.basename(hdr_file)}")
        else:
            # Ce cas ne devrait pas arriver si L_min est bien le minimum global.
            print(f"Avertissement: Lignes actuelles ({current_lines}) < L_min ({min_lines}). Image ignorée.")
            return None 

        # 2. Sauvegarde du cube 3D (L_min, Colonnes, Bandes)
        file_base_name = os.path.basename(hdr_file).replace('.hdr', '')
        npy_path = os.path.join(DATASET_REFLEC_DIR, f"{file_base_name}_class{label}.npy")
        
        np.save(npy_path, reflectance_cube)
        
        return True # Succès
    else:
        return None

def load_and_calibrate(image_path_hdr, dark_path_hdr, white_path_hdr):
    """Charge une image ENVI et la convertit en réflectance."""
    # Cette fonction doit rester la même que précédemment
    try:
        img = spy.envi.open(image_path_hdr, image_path_hdr.replace('.hdr', '.bin'))
        dark = spy.envi.open(dark_path_hdr, dark_path_hdr.replace('.hdr', '.bin'))
        white = spy.envi.open(white_path_hdr, white_path_hdr.replace('.hdr', '.bin'))
        
        img_data = img.load()
        dark_data = dark.load()
        white_data = white.load()
        
        dark_mean_spectrum = np.mean(dark_data, axis=(0, 1))
        white_mean_spectrum = np.mean(white_data, axis=(0, 1))
        
        numerator = img_data - dark_mean_spectrum 
        denominator = white_mean_spectrum - dark_mean_spectrum + np.finfo(float).eps
        reflectance_data = numerator / denominator
        
        return reflectance_data.astype(np.float32)
    
    except Exception as e:
        # Il est utile d'afficher l'erreur pour identifier quel fichier pose problème
        print(f"Erreur de calibration dans un processus enfant ({image_path_hdr}): {e}")
        return None

    
 

def extract_ghost_features_maps (npy_path):

    # Fonction pour générer une gaussienne
    def generate_gaussian(waves, center, fwhm, amplitude):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return amplitude * np.exp(-0.5 * ((waves - center) / sigma) ** 2)

    # --- Open the image ---
    # img = envi.open(hdr_file, hdr_file.replace('.hdr', '.bin')  )
    # npy_path = filedialog.askopenfilename(title="NPY file of your image")

    # --- Load the image data as a NumPy array ---
    
    #I = img.load()
    I = np.load(npy_path)
    I = I[:708, :, :]  # Ensure L_min lines and band selection
    # wavelengths_list = img.bands.centers[BAND_START:BAND_END]
    wavelengths_list = np.arange(BAND_START, BAND_END)
    wavelengths = np.array(wavelengths_list)

    width, height, channels = I.shape

    # bands_sensitivity = []

    # for k in range( len(wavelengths) ):
    #     bands_sensitivity.append( generate_gaussian(np.arange(start = 400, stop= 1000, step= 1), wavelengths[k], 5, 1.0) )

    Gram = np.eye(channels, channels, dtype=np.float32)

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

    #Specral comparison
    ref = generate_gaussian(wavelengths, 700, 40, 0.5) + 0.1
    G, W = dist_KLPD( ref, I.reshape( ( width * height,  channels) ) )

    G = G.reshape((width, height))
    W = W.reshape((width, height))

    # Sélection d'une bande pour Gx et Gy (ex: la bande centrale)
    band_idx = channels // 2

    # for x in range(I.shape[0]):
    #     for y in range(I.shape[1]):
    #         Magnitude[x,y], Direction[x,y] = Compute_Magnitude_and_Direction( (Gx, Gy), Gram, (x,y) )

    # --- Affichage de Gx ---
    percentile = np.percentile(Gx2, 99)
    Gx2 = np.log(Gx2)
    norm_Gx = (Gx2 - np.min(Gx2)) / (np.max(Gx2) - np.min(Gx2))
    Gy2 = np.log(Gy2)
    norm_Gy = (Gy2 - np.min(Gy2)) / (np.max(Gy2) - np.min(Gy2))
    

    # --- Affichage de Gy ---
    

    # --- Affichage de G (Distance/Similitude) ---
    norm_G = (G - np.min(G)) / (np.max(G) - np.min(G))
    

    # --- Affichage de W ---
    norm_W = (W - np.min(W)) / (np.max(W) - np.min(W))
    

    return norm_Gx, norm_Gy,norm_G, norm_W


def compute_hist_features (npy_path, label):
    """
    Calcule et concatène les histogrammes des cartes de caractéristiques.
    """

    Gx, Gy, G, W = extract_ghost_features_maps (npy_path)

    #Loi de Sturges pour le nombre de bins
    # num_bins = int( np.ceil( 1 + 3.322*np.log10(T.size) ) )
    # hist_T, _ = np.histogram(T, bins=num_bins, range=(np.min(T), np.max(T)), density=True)
    # hist_theta, _ = np.histogram(theta, bins=num_bins, range=(np.min(theta), np.max(theta)), density=True)
    # hist_G, _ = np.histogram(G, bins=num_bins, range=(np.min(G), np.max(G)), density=True)
    # hist_W, _ = np.histogram(W, bins=num_bins, range=(np.min(W), np.max(W)), density=True)
    



    # empiler les histogrammes suivant la dimension 3
    feature_vector = np.dstack((Gx, Gy, G, W))
    

    # Sauvegarde individuelle du vecteur d'histogramme (X_image, y_label) en NPY
    file_base_name = os.path.basename(npy_path).replace('.npy', '')
    npy_path_hist = os.path.join(DATASET_HIST_DIR, f"{file_base_name}_hist_class{label}.npy")
    
    # Sauvegarde du vecteur de caractéristiques NPY (1D array)
    np.save(npy_path_hist, feature_vector)
    
    return feature_vector