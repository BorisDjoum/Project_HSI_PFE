# Imports #

import numpy as np
#import tifffile as tiff

# Functions #

#########################################################################################################################################################

def resample_HSI(hsi, wavelengths, definition=1.0, low_bound=360.0, high_bound=760.0):
    """
    Resamples the given spectrum based on the provided definition (interval size between points)
    and wavelength bounds using linear interpolation.

    Parameters:
    - spectra: numpy array with 2 columns (wavelength, intensity)
    - definition: step size or resolution for the new wavelength sampling (default 1.0 nm)
    - low_bound: lower bound of the wavelength range (default 360.0 nm)
    - high_bound: upper bound of the wavelength range (default 760.0 nm)

    Returns:
    - resampled_curve: numpy array with 2 columns (wavelength, resampled intensity)
    """

    # Generate new wavelength range based on the definition and bounds
    new_wavelengths = np.arange(low_bound, high_bound + definition, definition)

    new_hsi = np.zeros(  ( hsi.shape[0], hsi.shape[1], new_wavelengths.shape[0] ) )
    
    for i in range(hsi.shape[0]):
        for j in range(hsi.shape[1]):
            # Use numpy's interp function to perform linear interpolation
            new_hsi[i,j,:] = np.interp(new_wavelengths, np.array(wavelengths).flatten(), hsi[i,j,:].flatten())
    
    return new_hsi, new_wavelengths



def resample_spectra(spectrum, wavelengths, definition=1.0, low_bound=360.0, high_bound=760.0):
    """
    Resamples a single spectrum based on the provided definition (interval size between points)
    and wavelength bounds using linear interpolation.

    Parameters:
    - spectrum: 1D numpy array of shape (n_bands,), intensities of the spectrum
    - wavelengths: 1D array-like of shape (n_bands,), corresponding wavelengths
    - definition: step size or resolution for the new wavelength sampling (default 1.0 nm)
    - low_bound: lower bound of the wavelength range (default 360.0 nm)
    - high_bound: upper bound of the wavelength range (default 760.0 nm)

    Returns:
    - resampled_spectrum: 1D numpy array of shape (n_new_bands,)
    - new_wavelengths: 1D numpy array of shape (n_new_bands,)
    """

    # Generate new wavelength range
    new_wavelengths = np.arange(low_bound, high_bound + definition, definition)

    # Perform linear interpolation
    resampled_spectrum = np.interp(new_wavelengths, np.array(wavelengths).flatten(), np.array(spectrum).flatten())

    return resampled_spectrum, new_wavelengths

#########################################################################################################################################################

def open_tiff_file(file_path):
    global rgb_image, hsi_cube, img_label_rgb, img_label_band
    
    if file_path:
        with tiff.TiffFile(file_path) as tif:
            # Load the RGB image (first IFD)
            rgb_image = tif.pages[0].asarray()          
            
            # Load the hyperspectral cube (remaining IFDs)
            hsi_bands = [tif.pages[i].asarray() for i in range(1, len(tif.pages))]
            hsi_cube = np.stack(hsi_bands, axis=-1)

            return rgb_image, hsi_cube
        
#########################################################################################################################################################

def split_HSI_patches(HSI, patch_size=128, train_ratio=0.8):
    """
    Divide the HSI into patches, then split them into train and test sets.

    Parameters:
        HSI (np.ndarray): Resampled hyperspectral image (w, h, c).
        patch_size (int): Size of the smaller patches (128 by default).
        train_ratio (float): Ratio of patches to use for the training set (between 0 and 1).

    Returns:
        train_patches (np.ndarray): Array of training patches.
        test_patches (np.ndarray): Array of test patches.
    """
    w, h, _ = HSI.shape
    patches = []
    
    # Divide the image into patches of size (patch_size, patch_size)
    for i in range(0, w, patch_size):
        for j in range(0, h, patch_size):
            patch = HSI[i:i + patch_size, j:j + patch_size, :]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)

    # Randomly shuffle the patches
    np.random.shuffle(patches)
    
    # Calculate the number of patches for the training set based on the ratio
    total_patches = len(patches)
    num_train = int(total_patches * train_ratio)
    
    # Split into training and test sets
    train_patches = np.array(patches[:num_train])
    test_patches = np.array(patches[num_train:])
    
    return train_patches, test_patches

#########################################################################################################################################################

def quantize_data(dict_data, max_values, min_values, n_levels=256):
    """
    Quantize the arrays in the dictionary using column-wise max and min values.
    
    Parameters:
        dict_data (dict): Dictionary containing numpy arrays to quantize.
        max_values (np.ndarray): Array of max values for each column (shape: (8,)).
        min_values (np.ndarray): Array of min values for each column (shape: (8,)).
        n_levels (int): Number of quantization levels (default: 256).

    Returns:
        dict: Dictionary with quantized arrays (same structure as input).
    """
    quantized_data = {}

    # Iterate over each key and array in the dictionary
    for key, array in dict_data.items():
        # Normalize the data to [0, 1]
        normalized = (array - min_values) / (max_values - min_values)
        normalized = np.clip(normalized, 0, 1)  # Ensure values are within [0, 1]

        # Quantize the normalized data
        quantized = np.floor(normalized * (n_levels - 1)).astype(np.int32)
        
        # Store the quantized array
        quantized_data[key] = quantized

    return quantized_data

#########################################################################################################################################################

def quantize_array(array, max_values, min_values, n_levels=256):
    """
    Quantize the arrays in the dictionary using column-wise max and min values.
    
    Parameters:
        array (np.ndarray): Array containing numpy arrays to quantize.
        max_values (np.ndarray): Array of max values for each column
        min_values (np.ndarray): Array of min values for each column
        n_levels (int): Number of quantization levels (default: 256).

    Returns:
        dict: Dictionary with quantized arrays (same structure as input).
    """
    quantized_data = {}

    # Normalize the data to [0, 1]
    normalized = (array - min_values) / (max_values - min_values)
    normalized = np.clip(normalized, 0, 1)  # Ensure values are within [0, 1]

    # Quantize the normalized data
    quantized = np.floor(normalized * (n_levels - 1)).astype(np.int32)

    return quantized