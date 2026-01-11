# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 14:30:02 2017
    HSI_library.py
    
    Purpose : Library of distance and divergense functions and spectral statistics
              Process the distance between 2 spectral sets S1 and S2 (must be with the same size)
              Establish the spectral statistics from a median spectrum (and an external reference for global stats)


    Functions :
        div_KL                  : Kullback-Leibler measure of information 
        pseudo_div_KL           : Kullback-Leibler pseudo-divergence : total divergence
        pseudo_div_KL2          : Kullback-Leibler pseudo-divergence returning shape and intensity differences
        div_Hellinger           : 
        dist_Minkowski          : order-p Minkowski distance
        dist_euclidienne_cum : Euclidean distance between cumulated spectra
        dist_euclidienne_cum_derive : Sum of Euclidean distances between cumulated spectra cumulated in the 2 spectral directions
        dist_SAM                : Spectral Angle Mapper
        dist_SID                : Spectral Information Divergence
        dist_SGA                : Spectral Gradient Angle
        dist_SCA                : Spectral Correlation Angle
        dist_chi_square         : order n chi square measure
        dist_bhatta             : Bhattacharyya Distance
        div_Csiszar             : Csiszar measure of information defined by bhatia(2013)
        pseudo_div_Csiszar2r    : new Csiszar divergence as defined by Bathia(2013) adapted to the KLPD framework
        
        GlobalSpectralStats     : standard deviation, Dissymetry and Flattening moments of a spectral set (global moments)
        StructSpectralStats     : variance-covariance Matrix of a spectral set from a BHSD relative to a Median spectrum (structural moment)

    status : Ok (28/09/2017)
             To add/modify      :   to modify each distance function in order to transform the sums into integrals
                                    to embedd preprocessing to allow distance processing between 1 reference and a spectral set
                                    to embedd Dynamic Time Warping approach for comparison purposes.

    date   : 28/09/2017
    
@author: Hilda Deborah, Noël Richard, Martin Tamisier, Yu-Jung Chen
"""
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt  ## for the hist function (in SpectralEntropy)
from matplotlib.colors     import LogNorm

##============================================================================
## zone de déclaration des fonctions utilisées
##============================================================================

# KLPD
def dist_KLPD(S1, S2):
    
    return KLPD(S1,S2)

def div_KL(S1,S2,resolution=1.0):
    """
     Divergence de Kullback-Leibler (KL) avec prise en compte de la résolution spectrale

     Parameters:
     ===========
        :param P: spectrum 1
        :type P: float numpy array
        :param Q: spectrum 2
        :type Q: float numpy array
        :param R: resolution
        :type R: float numpy array


    Returns:
    ========
        :return: Divergence de Kullback-Leibler
        :rtype: numpy.float
    """

    Q = np.copy(S2)
    P = np.copy(S1)
    Q[Q< 10**(-32)] = 10**(-32)
    P[P< 10**(-32)] = 10**(-32)
    #A = P/Q
    #return np.trapz(P*np.log(A)), dx=resolution, axis=0)

    return np.trapz(P*(np.log(P) - np.log(Q)), dx=resolution, axis=0)


def pseudo_div_KL(H1, H2, resolution=1.0):
    """
     Pseudo-Divergence spectrale de Kullback-Leibler .

     Parameters:
     ===========
        :param H1: spectrum 1 or spectra list nb lines(nb spectra) x nb_col (wavelength)
        :type H1: float numpy array
        :param H2: spectrum 2
        :type H2: float numpy array
    Returns:
    ========
        :return: Pseudo-Divergence spectrale de Kullback-Leibler
        :rtype: numpy.float
    """
    if (H1.shape[0] == H1.size):
        nbwaves   = H1.size
        nbsamples = 1
    else :
        [nbsamples, nbwaves] = H1.shape

    k1 = np.trapz(H1,dx = resolution, axis=0).reshape(nbsamples,1)
    k2 = np.trapz(H2,dx = resolution, axis=0).reshape(nbsamples,1)


    return ( k1*div_KL( H1/k1, H2/k2, resolution).reshape(nbsamples,1)
           + k2*div_KL(H2/k2, H1/k1, resolution).reshape(nbsamples,1)
           + (k1-k2)*np.log(k1/k2)
           )

##============================================================================
def pseudo_div_KL2(H1, H2, resolution=1.0):
    """
     Pseudo-Divergence spectrale de Kullback-Leibler ,
     nécessite la valeur de la résolution spectrale (dx) et la spécification
     de la méthode d'intégration.
     
     2 possible cases : distance between 2 spectral sets of same size
                    or  distance between 1 reference and a spectral set
                        (require to duplicate the reference)

     Parameters:
     ===========
        :param H1: spectrum 1
        :type H1: float numpy array[nb samples, nb wavelength channels]
        
        :param H2: spectrum 2
        :type H2: float numpy array[nb samples, nb wavelength channels]
        
        :param resolution: spectral resolution in nm
        :type H2: float 
    Returns:
    ========
        :return: Pseudo-Divergence spectrale de Kullback-Leibler
        :rtype: numpy.float
    """
    ## extract the number of samples and number of spectra for each spectral set
    if (H1.ndim == 1):
        nbwaves1   = H1.size
        nbsamples1 = 1
        H1=H1.reshape((1,nbwaves1))
    else:
        [nbsamples1, nbwaves1] = H1.shape

    if (H2.ndim == 1):
        nbwaves2   = H2.size
        nbsamples2 = 1
        H2=H2.reshape((1,nbwaves1))
    else:
        print(H2.shape)
        [nbsamples2, nbwaves2] = H2.shape
    
    ## Verify that the spectral resolution are in accordance    
    if (nbwaves1 != nbwaves2):
        print("H1 and H2 don't have the same spectral band count")
        return (-1.0)
    
    ## Duplicate if necessary one of the two spectral set (being the reference)
    if (nbsamples1 == nbsamples2):
        nbsamples = nbsamples1
        nbwaves   = nbwaves1
    elif (nbsamples1 == 1):
        ## duplicate H1
        nbsamples = nbsamples2
        nbwaves   = nbwaves2
        H1 = np.tile(H1, nbsamples).reshape(nbsamples, nbwaves )     
    elif (nbsamples2 == 1):
        ## duplicate H2
        nbsamples = nbsamples1
        nbwaves   = nbwaves1
        H2 = np.tile(H2, nbsamples).reshape(nbsamples, nbwaves )
    else:
        print("H1 and H2 must have the same number of samples, or just 1(as reference)")
        print('nbsamples 1 :' + str(nbsamples1) + '   nbsamples 2 : ' + str(nbsamples2))
        
    ## Spectral integration using trapzale rule
    k1 = np.trapz(H1,dx = resolution, axis=0).reshape(nbsamples,1)
    k2 = np.trapz(H2,dx = resolution, axis=0).reshape(nbsamples,1)

    ## distance computation splitting the spectral difference in 2 sub-parts
    dist_shape  = k1*div_KL( H1/k1,H2/k2, resolution).reshape(nbsamples,1) \
                + k2*div_KL(H2/k2, H1/k1,resolution).reshape(nbsamples,1)
    dist_energy = (k1-k2)*np.log(k1/k2)

    dist = np.concatenate ( (dist_shape, dist_energy),axis=0 )
    dist = dist.reshape( nbsamples , 2)
    return (dist_shape, dist_energy, dist)


import numpy as np

def KLPD_SUM_xp(S1, S2, xp, debug=False):
    """
    GPU/CPU-compatible version of KLPD_SUM using xp (NumPy or CuPy).
    S1: (..., B), S2: (B,) or (..., B)
    """
    ep = 1e-6
    min_val = 1e-12

    # Replace zeros to avoid division by zero
    S1 = xp.where(S1 == 0, ep, S1)
    S2 = xp.where(S2 == 0, ep, S2)

    # Compute integrals
    k1 = xp.trapz(S1, axis=-1)
    k2 = xp.trapz(S2, axis=-1)

    # Normalize spectra
    if S1.ndim > 1:
        k1_exp = k1[..., xp.newaxis]
        k2_exp = k2[..., xp.newaxis]
    else:
        k1_exp = k1
        k2_exp = k2

    N1 = S1 / k1_exp + ep
    N2 = S2 / k2_exp + ep

    # Ratio with clipping
    ratio_N = xp.clip(N1 / N2, min_val, None)
    ratio_N_inv = xp.clip(N2 / N1, min_val, None)
    ratio_k = xp.clip(k1 / k2, min_val, None)

    if debug:
        if xp.any((N1 / N2) < min_val):
            print("Debug: Clipping applied on N1/N2")
        if xp.any((N2 / N1) < min_val):
            print("Debug: Clipping applied on N2/N1")
        if xp.any((k1 / k2) < min_val):
            print("Debug: Clipping applied on k1/k2")

    p1 = N1 * xp.log(ratio_N)
    p2 = N2 * xp.log(ratio_N_inv)

    G = k1 * xp.trapz(p1, axis=-1) + k2 * xp.trapz(p2, axis=-1)
    W = (k1 - k2) * xp.log(ratio_k)

    return G + W

def KLPD_SUM(S1, S2, debug=False):
    """
    Calcule G et W à partir de deux spectres S1 et S2 en utilisant la divergence symétrique de Kullback-Leibler.
    
    Paramètres :
      - S1, S2 : Tableaux numpy représentant les spectres.
      - debug : booléen (False par défaut). Si True, affiche des messages de débogage lorsque le clipping est appliqué.
    
    Renvoie :
      - G, W : Valeurs calculées à partir des spectres.
    
    Remarque :
      Les erreurs "invalid value encountered in log" proviennent du fait que np.log est appliqué sur des valeurs ≤ 0.
      Pour éviter cela, nous utilisons np.clip afin de garantir que les arguments de np.log soient toujours strictement positifs.
    """
    
    # Remplacer les zéros pour éviter la division par zéro
    S1 = np.where(S1 == 0, 1e-6, S1)
    S2 = np.where(S2 == 0, 1e-6, S2)

    # Calcul des intégrales avec la règle des trapèzes
    k1 = np.trapz(S1, axis=-1)
    k2 = np.trapz(S2, axis=-1)

    # Normalisation des spectres avec un offset pour éviter les zéros exacts
    ep = 1e-6
    if S1.ndim > 1:
        N1 = (S1 / k1[:, np.newaxis]) + ep
        N2 = (S2 / k2[:, np.newaxis]) + ep
    else:
        N1 = (S1 / k1) + ep
        N2 = (S2 / k2) + ep

    # Définir une valeur minimale pour éviter de prendre le log de 0
    min_val = 1e-12

    # Calculer les rapports et les clipper pour qu'ils soient ≥ min_val
    ratio_N = np.clip(N1 / N2, min_val, None)
    ratio_N_inv = np.clip(N2 / N1, min_val, None)
    ratio_k = np.clip(k1 / k2, min_val, None)

    # Affichage en mode debug si un clipping a été nécessaire
    if debug:
        if np.any((N1 / N2) < min_val):
            print("Debug : Clipping appliqué sur N1/N2")
        if np.any((N2 / N1) < min_val):
            print("Debug : Clipping appliqué sur N2/N1")
        if np.any((k1 / k2) < min_val):
            print("Debug : Clipping appliqué sur k1/k2")

    # Calcul des composantes p1 et p2
    p1 = N1 * np.log(ratio_N)
    p2 = N2 * np.log(ratio_N_inv)

    # Calcul de G et W
    G = k1 * np.trapz(p1, axis=-1) + k2 * np.trapz(p2, axis=-1)
    W = (k1 - k2) * np.log(ratio_k)

    return G + W

def KLPD(S1, S2, debug=False):
    """
    Calcule G et W à partir de deux spectres S1 et S2 en utilisant la divergence symétrique de Kullback-Leibler.
    
    Paramètres :
      - S1, S2 : Tableaux numpy représentant les spectres.
      - debug : booléen (False par défaut). Si True, affiche des messages de débogage lorsque le clipping est appliqué.
    
    Renvoie :
      - G, W : Valeurs calculées à partir des spectres.
    
    Remarque :
      Les erreurs "invalid value encountered in log" proviennent du fait que np.log est appliqué sur des valeurs ≤ 0.
      Pour éviter cela, nous utilisons np.clip afin de garantir que les arguments de np.log soient toujours strictement positifs.
    """
    
    # Remplacer les zéros pour éviter la division par zéro
    S1 = np.where(S1 == 0, 1e-6, S1)
    S2 = np.where(S2 == 0, 1e-6, S2)

    # Calcul des intégrales avec la règle des trapèzes
    k1 = np.trapezoid(S1, axis=-1)
    k2 = np.trapezoid(S2, axis=-1)

    # Normalisation des spectres avec un offset pour éviter les zéros exacts
    ep = 1e-6
    if S1.ndim > 1:
        N1 = (S1 / k1[:, np.newaxis]) + ep
    else:
        N1 = (S1 / k1) + ep

    if S2.ndim > 1:
        N2 = (S2 / k2[:, np.newaxis]) + ep
    else:
        N2 = (S2 / k2) + ep

    # Définir une valeur minimale pour éviter de prendre le log de 0
    min_val = 1e-12

    # Calculer les rapports et les clipper pour qu'ils soient ≥ min_val
    ratio_N = np.clip(N1 / N2, min_val, None)
    ratio_N_inv = np.clip(N2 / N1, min_val, None)
    ratio_k = np.clip(k1 / k2, min_val, None)

    # Affichage en mode debug si un clipping a été nécessaire
    if debug:
        if np.any((N1 / N2) < min_val):
            print("Debug : Clipping appliqué sur N1/N2")
        if np.any((N2 / N1) < min_val):
            print("Debug : Clipping appliqué sur N2/N1")
        if np.any((k1 / k2) < min_val):
            print("Debug : Clipping appliqué sur k1/k2")

    # Calcul des composantes p1 et p2
    p1 = N1 * np.log(ratio_N)
    p2 = N2 * np.log(ratio_N_inv)

    # Calcul de G et W
    G = k1 * np.trapz(p1, axis=-1) + k2 * np.trapz(p2, axis=-1)
    W = (k1 - k2) * np.log(ratio_k)

    return G, W

def KLPD_Shape(S1, S2, debug=False):
    """
    Calcule G et W à partir de deux spectres S1 et S2 en utilisant la divergence symétrique de Kullback-Leibler.
    
    Paramètres :
      - S1, S2 : Tableaux numpy représentant les spectres.
      - debug : booléen (False par défaut). Si True, affiche des messages de débogage lorsque le clipping est appliqué.
    
    Renvoie :
      - G, W : Valeurs calculées à partir des spectres.
    
    Remarque :
      Les erreurs "invalid value encountered in log" proviennent du fait que np.log est appliqué sur des valeurs ≤ 0.
      Pour éviter cela, nous utilisons np.clip afin de garantir que les arguments de np.log soient toujours strictement positifs.
    """
    
    # Remplacer les zéros pour éviter la division par zéro
    S1 = np.where(S1 == 0, 1e-6, S1)
    S2 = np.where(S2 == 0, 1e-6, S2)

    # Calcul des intégrales avec la règle des trapèzes
    k1 = np.trapz(S1, axis=-1)
    k2 = np.trapz(S2, axis=-1)

    # Normalisation des spectres avec un offset pour éviter les zéros exacts
    ep = 1e-6
    if S1.ndim > 1:
        N1 = (S1 / k1[:, np.newaxis]) + ep
        N2 = (S2 / k2[:, np.newaxis]) + ep
    else:
        N1 = (S1 / k1) + ep
        N2 = (S2 / k2) + ep

    # Définir une valeur minimale pour éviter de prendre le log de 0
    min_val = 1e-12

    # Calculer les rapports et les clipper pour qu'ils soient ≥ min_val
    ratio_N = np.clip(N1 / N2, min_val, None)
    ratio_N_inv = np.clip(N2 / N1, min_val, None)
    ratio_k = np.clip(k1 / k2, min_val, None)

    # Affichage en mode debug si un clipping a été nécessaire
    if debug:
        if np.any((N1 / N2) < min_val):
            print("Debug : Clipping appliqué sur N1/N2")
        if np.any((N2 / N1) < min_val):
            print("Debug : Clipping appliqué sur N2/N1")
        if np.any((k1 / k2) < min_val):
            print("Debug : Clipping appliqué sur k1/k2")

    # Calcul des composantes p1 et p2
    p1 = N1 * np.log(ratio_N)
    p2 = N2 * np.log(ratio_N_inv)

    # Calcul de G et W
    G = k1 * np.trapz(p1, axis=-1) + k2 * np.trapz(p2, axis=-1)
    W = (k1 - k2) * np.log(ratio_k)

    return G

def KLPD_Energy(S1, S2, debug=False):
    """
    Calcule G et W à partir de deux spectres S1 et S2 en utilisant la divergence symétrique de Kullback-Leibler.
    
    Paramètres :
      - S1, S2 : Tableaux numpy représentant les spectres.
      - debug : booléen (False par défaut). Si True, affiche des messages de débogage lorsque le clipping est appliqué.
    
    Renvoie :
      - G, W : Valeurs calculées à partir des spectres.
    
    Remarque :
      Les erreurs "invalid value encountered in log" proviennent du fait que np.log est appliqué sur des valeurs ≤ 0.
      Pour éviter cela, nous utilisons np.clip afin de garantir que les arguments de np.log soient toujours strictement positifs.
    """
    
    # Remplacer les zéros pour éviter la division par zéro
    S1 = np.where(S1 == 0, 1e-6, S1)
    S2 = np.where(S2 == 0, 1e-6, S2)

    # Calcul des intégrales avec la règle des trapèzes
    k1 = np.trapz(S1, axis=-1)
    k2 = np.trapz(S2, axis=-1)

    # Normalisation des spectres avec un offset pour éviter les zéros exacts
    ep = 1e-6
    if S1.ndim > 1:
        N1 = (S1 / k1[:, np.newaxis]) + ep
        N2 = (S2 / k2[:, np.newaxis]) + ep
    else:
        N1 = (S1 / k1) + ep
        N2 = (S2 / k2) + ep

    # Définir une valeur minimale pour éviter de prendre le log de 0
    min_val = 1e-12

    # Calculer les rapports et les clipper pour qu'ils soient ≥ min_val
    ratio_N = np.clip(N1 / N2, min_val, None)
    ratio_N_inv = np.clip(N2 / N1, min_val, None)
    ratio_k = np.clip(k1 / k2, min_val, None)

    # Affichage en mode debug si un clipping a été nécessaire
    if debug:
        if np.any((N1 / N2) < min_val):
            print("Debug : Clipping appliqué sur N1/N2")
        if np.any((N2 / N1) < min_val):
            print("Debug : Clipping appliqué sur N2/N1")
        if np.any((k1 / k2) < min_val):
            print("Debug : Clipping appliqué sur k1/k2")

    # Calcul des composantes p1 et p2
    p1 = N1 * np.log(ratio_N)
    p2 = N2 * np.log(ratio_N_inv)

    # Calcul de G et W
    G = k1 * np.trapz(p1, axis=-1) + k2 * np.trapz(p2, axis=-1)
    W = (k1 - k2) * np.log(ratio_k)

    return W

##============================================================================
##============================================================================

def div_Hellinger(H1,H2,resolution=1.0):
    '''
    Hellinger.
	Adapted to change the sum into an integral using the trapzale rule.

        :param H1: Spectral set 1 
        :type H1: float numpy array[nb Sample, nb Wavelengths]
        :param H2: Spectral set 2
        :type H2: float numpy array [nb Sample, nb Wavelengths]
        
		:param resolution: spectral sampling in nm
        :type resolution: numpy.float 
        :return: distance
        :rtype: numpy.float

        :Example: D=dist_euclienne(H1,H2,2, 1.0)
    '''

    return ( 0.5* np.trapz( (np.sqrt(H1) - np.sqrt(H2))**2, dx= resolution, axis=0))


##============================================================================

def dist_Minkowski(H1,H2,p=2,resolution=1.0):
    '''
    Minkowski distance Fonction with p order (p=2: Euclidean).
	Adapted to change the sum into an integral using the trapzale rule.

        :param H1: Spectral set 1 
        :type H1: float numpy array[nb Sample, nb Wavelengths]
        :param H2: Spectral set 2
        :type H2: float numpy array [nb Sample, nb Wavelengths]
        :param p: Minkowski order
        :type p: numpy.float 
		:param resolution: spectral sampling in nm
        :type resolution: numpy.float 
        :return: distance
        :rtype: numpy.float

        :Example: D=dist_euclienne(H1,H2,2, 1.0)
    '''

    return ( np.trapz((np.abs(H1-H2))**p, dx= resolution, axis=0))**(1.0/p)


##============================================================================
def dist_euclidienne_cum_derive(H1,H2):
    '''
     function which provide de sum of two
	 Euclidean of cumulative spectrum distance function (ECS)
	 for the right and left directions.

        :param H1: spectrum 1
        :type H1: float numpy array
        :param H2: spectrum 2
        :type H2: float numpy array
        :return: distance
        :rtype: numpy.float

        :Example: D=dist_euclienne_cum(H1,H2)
    '''
    H1_cum = np.cumsum( H1 )
    H2_cum = np.cumsum( H2 )
    A = np.sqrt( np.sum( ( np.abs( H1_cum - H2_cum ) )**2.0))


    H1_cum = np.cumsum( H1[::-1] )
    H2_cum = np.cumsum( H2[::-1] )
    B = np.sqrt( np.sum( ( np.abs( H1_cum - H2_cum ) )**2.0 ) )

    return A+B


##============================================================================

def dist_euclidienne_cum(H1,H2, sens=0):
   '''
   Euclidean of cumulative spectum distance function (ECS).

       :param H1: spectrum 1
       :type H1: float numpy array
       :param H2: spectrum 2
       :type H2: float numpy array
       :return: distance
       :rtype: numpy.float

       :Example: D=dist_euclienne_cum(H1,H2)
   '''
   if sens == 1:
       H1=H1[::-1]
       H2=H2[::-1]
   H1_cum = np.cumsum(H1)
   H2_cum = np.cumsum(H2)
   return np.sqrt( np.sum( ( np.abs( H1_cum - H2_cum ) )**2.0) )


##============================================================================
def dist_SAM(H1, H2, resolution = 1.0):

    """
     Spectral Angle Mapper distance function (SAM).
	 Adapted to transform the sum into an integral using the trapzal rule

        :param H1: spectrum 1
        :type H1: float numpy array
        :param H2: spectrum 2
        :type H2: float numpy array
        :return: distance
        :rtype: numpy.float
    """

    # print((H1*H2).shape)
    # print((H1**2).shape)
    # print((H2**2).shape)


    A = np.trapz(H1*H2, dx= resolution, axis=0)/((np.trapz(H1**2, dx= resolution, axis=0)**(1.0/2.0))*(np.trapz(H2**2.0, dx= resolution, axis=0)**(1.0/2.0)))

#    if A>1.0 : A = 1.0
    A=np.where(A>1.0, 1.0, A)

    return np.arccos(A)
##============================================================================

def dist_SID(S1,S2):

    """
     Spectral information divergence distance function (SID).

        :param H1: spectrum 1
        :type H1: float numpy array
        :param H2: spectrum 2
        :type H2: float numpy array
        :return: distance
        :rtype: numpy.float
    """

    S1[S1<10.0**-9] = 10**-9
    S2[S2<10.0**-9] = 10**-9

    k1 = np.trapz(S1)
    k2 = np.trapz(S2) 

    H1 = S1 / k1
    H2 = S2 / k2

    H11 = np.copy(H1)
    H22 = np.copy(H2)

    H11[H11<10.0**-9] = 10**-9
    H22[H22<10.0**-9] = 10**-9

    return ( ( ( H11 / H11.sum() ) - ( H22 / H22.sum() ) ) * ( np.log( H11 / H11.sum() ) - np.log( H22 / H22.sum() ) ) ).sum()


##============================================================================

def dist_SGA(H1, H2):
    """
     Spectral gradient angle distance function (SGA).

        :param H1: spectrum 1
        :type H1: float numpy array
        :param H2: spectrum 2
        :type H2: float numpy array
        :return: distance
        :rtype: numpy.float
    """

    n = len(H1)

    SG_H1 = H1[1:]-H1[:n-1]
    SG_H2 = H2[1:]-H2[:n-1]

    return dist_SAM(np.abs(SG_H1), np.abs(SG_H2))

##============================================================================

def dist_SCA(H1,H2):

    """
    Spectral Corelation Angle distance function (SCA).

        :param H1: spectrum 1
        :type H1: float numpy array
        :param H2: spectrum 2
        :type H2: float numpy array
        :return: distance
        :rtype: numpy.float
    """
    H1n = H1-np.mean(H1)
    H2n = H2-np.mean(H2)

    c = (H1n*H2n).sum()/(((((H1n)**2).sum())**(1.0/2.0))*((((H2n)**2).sum())**(1.0/2.0)))
    if c>1.0: c = 1.0
    return np.arccos((c+1.0)/2.0)



##============================================================================
def dist_chi_square(H1,H2,n):
    """
    chi square  distance function, order n.

       :param H1: spectrum 1
       :type H1: float numpy array
       :param H2: spectrum 2
       :type H2: float numpy array
       :param n: order
       :type H1: numpy.float
       :return: distance
       :rtype: numpy.float
    """
    return  (1/n) * np.sum( ( ( ( H1 - H2 )**2 ) / ( ( H1 + H2 )**(2/n) ) ) )


def dist_bhatta(H1, H2):
    """
    Bhattacharyya distance function.

       :param H1: spectrum 1
       :type H1: float numpy array
       :param H2: spectrum 2
       :type H2: float numpy array
       :return: distance
       :rtype: numpy.float
    """
    return -1*np.log(np.sum((H1*H2)**(1/2)))

##============================================================================
def dist_SAM_L2(H1, H2, resolution=1.0):
    """
     Process the SAM and L2 measures in order to construct histogram of Spectral
     Difference (unadapted alternative to KL-pseudo-divergence )
     
     2 possible cases : distance between 2 spectral sets of same size
                    or  distance between 1 reference and a spectral set
                        (require to duplicate the reference)

     Parameters:
     ===========
        :param H1: spectrum 1
        :type H1: float numpy array[nb samples, nb wavelength channels]
        
        :param H2: spectrum 2
        :type H2: float numpy array[nb samples, nb wavelength channels]
        
        :param resolution: spectral resolution in nm
        :type H2: float 
    Returns:
    ========
        :return: Pseudo-Divergence spectrale de Kullback-Leible
        :rtype: numpy.float
    """
    ## extract the number of samples and number of spectra for each spectral set
    if (H1.shape[0] == H1.size):
        nbwaves1   = H1.size
        nbsamples1 = 1
    else:
        [nbsamples1, nbwaves1] = H1.shape

    if (H2.shape[0] == H2.size):
        nbwaves2   = H2.size
        nbsamples2 = 1
    else:
        [nbsamples2, nbwaves2] = H2.shape
    
    ## Verify that the spectral resolution are in accordance    
    if (nbwaves1 != nbwaves2):
        print("H1 and H2 don't have the same spectral band count")
        return (-1.0)
    
    ## Duplicate if necessary one of the two spectral set (being the reference)
    if (nbsamples1 == nbsamples2):
        nbsamples = nbsamples1
        nbwaves   = nbwaves1
    elif (nbsamples1 == 1):
        ## duplicate H1
        nbsamples = nbsamples2
        nbwaves   = nbwaves2
        H1 = np.tile(H1, nbsamples).reshape(nbsamples, nbwaves )     
    elif (nbsamples2 == 1):
        ## duplicate H2
        nbsamples = nbsamples1
        nbwaves   = nbwaves1
        H2 = np.tile(H2, nbsamples).reshape(nbsamples, nbwaves )
    else:
        print("H1 and H2 must have the same number of samples, or just 1(as reference)")
        
    ## construct a simili distance with a shape and an intensity spectral difference
    dist_shape  = dist_SAM(H1, H2, resolution).reshape(nbsamples,1)
    dist_energy = dist_Minkowski(H1, H2, 2, resolution).reshape(nbsamples,1)
    
    # print(dist_shape.shape)
    # print(dist_energy.shape)

    dist = np.concatenate ( (dist_shape, dist_energy), axis=0 )
    dist = dist.reshape( nbsamples , 2)
    return dist
##============================================================================

##============================================================================
def GlobalSpectralStats(Spectral_Set, Median_Spectrum, Spectral_Ref, Resolution=1.0):
    """
     Process the global spectral moments of a spectral set :
         Standard Deviation    
         Dissymetry = skewness, 
         Flattening= Kurthosis
    These moments are processed from the total KLPD established between
    each spectrum and the Median spectrum of the set. A spectral reference 
    is required in order to obtain algebraic distances.

     Parameters:
     ===========
        :param Spectral_Set
        :type H1: float numpy array[nb samples, nb wavelength channels]
        
        :param Median_Spectrum
        :type H2: float numpy array[nb wavelength channels]
        
        :param Spectral_Ref
        :type H2: float numpy array[nb wavelength channels]
        
        :param resolution: spectral resolution in nm
        :type H2: float 

    Returns:
    ========
        :return: [StdDev, Dissymetry, Flattening]
        :rtype: numpy.float.array ( (1,3) )
    """
    
    count = Spectral_Set.shape[0] ## nb of spectra in the spectral set
    
    Dist2Median = pseudo_div_KL2(Spectral_Set, Median_Spectrum,  Resolution ).sum(axis=0).reshape(count,1)
    Dist2Ref    = pseudo_div_KL2(Spectral_Set, Spectral_Ref,  Resolution ).sum(axis=0).reshape(count,1)

    Dist_Ref2Median = pseudo_div_KL2(Spectral_Ref, Median_Spectrum,  Resolution).sum(axis=0)
    Dist_Ref2Median = np.tile(Dist_Ref2Median, count).reshape(count, 1)  

    Tau_table  = np.sign( Dist2Ref - Dist_Ref2Median  )
    
    ## the algebraic distance
    Algebraic_Dist = Tau_table * Dist2Median
    
    ## the global statistics
    StdDev  = np.sqrt(np.sum( (Algebraic_Dist)**2 ) / count)
    Dissym  = (np.sum( (Algebraic_Dist)**3 ) / count) / (StdDev**3)
    Flatng  = (np.sum( (Algebraic_Dist)**4 ) / count) / (StdDev**4)
    
    return [StdDev, Dissym, Flatng]
##============================================================================

##============================================================================
def StructSpectralStats(Spectral_Set, Median_Spectrum, Resolution=1.0):
    """
     Process the spectral structural moments of a spectral set, so the variance-covariance matrix
     obtained from a BHSD processed from the Median spectrum of the set.

     Parameters:
     ===========
        :param Spectral_Set
        :type H1: float numpy array[nb samples, nb wavelength channels]
        
        :param Median_Spectrum
        :type H2: float numpy array[nb wavelength channels]
        
        :param resolution: spectral resolution in nm
        :type H2: float 

    Returns:
    ========
        :return: the variance-covariance matrix
        :rtype: numpy.float.array ( (2,2) )
    """

    count = Spectral_Set.shape[0]    
    Spectral_Dist2Median = pseudo_div_KL2(Spectral_Set, Median_Spectrum,  Resolution )
    
    ## the elements of the variance-covariance matrix
    alpha_ShSh = np.sum( Spectral_Dist2Median[:,0]*Spectral_Dist2Median[:,0] ) / count
    alpha_II   = np.sum( Spectral_Dist2Median[:,1]*Spectral_Dist2Median[:,1] ) / count
    alpha_ShI  = np.sum( Spectral_Dist2Median[:,0]*Spectral_Dist2Median[:,1] ) / count
    
    Gamma = np.array( [ [alpha_ShSh, alpha_ShI] , [alpha_ShI, alpha_II] ])
    
    return(Gamma)
##============================================================================

##============================================================================
def SpectralEntropy(SpecImg, reference_spec, Resolution, BHSDrange=None, BHSD_bins=75):
    """
     Process the spectral entropy from a BHSD processed from a reference spectrum (to define).
     The Entropy would be usefull to select the right reference (hypothesis)

     Parameters:
     ===========
        :param Spectral_Set - samples
        :type H1: float numpy array[nb samples, nb wavelength channels]
        
        :param Median_Spectrum - reference - median for now
        :type H2: float numpy array[nb wavelength channels]
        
        :param resolution: spectral resolution in nm
        :type H2: float 
        
        :param BHSDrange: hist2D range :  [[xmin, xmax], [ymin, ymax]]
        :type H2: array like 

        :param BHSD_bins: number of bins per axis (so histogram of nb_bins x nb_bins )
        :type H2: integer numpy (by default 75) 

    Returns:
    ========
        :return: entropy value, unit as bits
        :rtype: numpy.float
    """
    #epsilon = 1e-6
#    hist_value = np.where(hist_value != 0 ,  hist_value, np.delete)
    if (SpecImg.ndim ==3):
        NbLg, NbCol, NbWaves = SpecImg.shape
    else: #SpecImg.ndim = 2
        NbLg, NbWaves = SpecImg.shape
        NbCol =1
        
    SpecList = SpecImg.reshape(NbLg*NbCol, NbWaves)
    SpecDist = pseudo_div_KL2(SpecList, reference_spec,  Resolution )
    
    BHSD = np.histogram2d(SpecDist[:,0], SpecDist[:,1], range = BHSDrange, bins = BHSD_bins, normed= True)
        
    hist_value = BHSD[0]    
    NonNullValueLocation = np.where(hist_value != 0)
    hist_percentage = hist_value / np.sum(hist_value)
    
    entropy = -np.sum(np.dot(hist_percentage[NonNullValueLocation[0], NonNullValueLocation[1]], np.log2(hist_percentage[NonNullValueLocation[0], NonNullValueLocation[1]]) ))
 
    return (entropy)
##============================================================================

##============================================================================
def div_Csiszar(S1,S2, alpha= 1.0, resolution=1.0):
    """
     Divergence de Csiszar, as defined by Bathia(2013) avec prise en compte de la résolution spectrale

     Parameters:
     ===========
        :param S1: spectrum 1
        :type S1: float numpy array
        :param S2: spectrum 2
        :type S2: float numpy array
        :param S2: alpha (alpha >0)
        :type S2: float numpy 
        
        :param : resolution
        :type R: float numpy array


    Returns:
    ========
        :return: Csiszar Measure of information
        :rtype: numpy.float
    """

    if ( alpha<= 0):
        print("Csiszar measure of information : alpha parameter must be > 0")
        return(-1)

    Q = np.copy(S2)
    P = np.copy(S1)
    Q[Q< 10**(-32)] = 10**(-32)
    P[P< 10**(-32)] = 10**(-32)
    #A = P/Q
    #return np.trapz(P*np.log(A)), dx=resolution, axis=0)

#    return np.trapz(P*np.sinh(alpha * (np.log(P) - np.log(Q)) ) / np.sinh(alpha), dx=resolution, axis=0)
    return ( np.sum(P*np.sinh(alpha * (np.log(P) - np.log(Q)) ) / np.sinh(alpha),  axis=0) +
             np.sum(Q*np.sinh(alpha * (np.log(Q) - np.log(P)) ) / np.sinh(alpha),  axis=0) )
##============================================================================

##============================================================================
def dist_Csiszar(S1,S2, alpha= 1.0, resolution=1.0):
    """
     New Distance de Csiszar, as defined by Bathia(2013) avec prise en compte de la résolution spectrale

     Parameters:
     ===========
        :param S1: spectrum 1
        :type S1: float numpy array
        :param S2: spectrum 2
        :type S2: float numpy array
        :param S2: alpha (alpha >0)
        :type S2: float numpy 
        
        :param : resolution
        :type R: float numpy array


    Returns:
    ========
        :return: Csiszar Measure of information
        :rtype: numpy.float
    """

    if ( alpha<= 0):
        print("Csiszar measure of information : alpha parameter must be > 0")
        return(-1)

    Q = np.copy(S2)
    P = np.copy(S1)
    Q[Q< 10**(-32)] = 10**(-32)
    P[P< 10**(-32)] = 10**(-32)
    #A = P/Q
    #return np.trapz(P*np.log(A)), dx=resolution, axis=0)

#    return np.trapz(P*np.sinh(alpha * (np.log(P) - np.log(Q)) ) / np.sinh(alpha), dx=resolution, axis=0)
    return ( np.trapz(P*np.sinh(alpha * (np.log(P) - np.log(Q)) ) / np.sinh(alpha), dx=resolution,  axis=0) +
             np.trapz(Q*np.sinh(alpha * (np.log(Q) - np.log(P)) ) / np.sinh(alpha), dx=resolution, axis=0) )

#    return ( np.sum(P*np.sinh(alpha * (np.log(P) - np.log(Q)) ) / np.sinh(alpha),  axis=0) +
#             np.sum(Q*np.sinh(alpha * (np.log(Q) - np.log(P)) ) / np.sinh(alpha),  axis=0) )
 
 
##============================================================================

##============================================================================
def pseudo_div_Csiszar2(H1, H2, alpha =1.0, resolution=1.0):
    """
     new Csiszar pseudo-divergence from Bathia (2013) adapted to the KLPD framework
     (producing a spectral shape and intensity difference).
     The theoretical developement stays to finish
     
     Classical alpha values : 0.5 ; 1.0
     
     The logarithm of the distance seems more usefull
     
     2 possible cases : distance between 2 spectral sets of same size
                    or  distance between 1 reference and a spectral set
                        (require to duplicate the reference)

     Parameters:
     ===========
        :param H1: spectrum 1
        :type H1: float numpy array[nb samples, nb wavelength channels]
        
        :param H2: spectrum 2
        :type H2: float numpy array[nb samples, nb wavelength channels]
        
        :param H3: alpha (alpha >0)
        :type H3: float numpy 
        
        :param resolution: spectral resolution in nm
        :type H2: float 
    Returns:
    ========
        :return: Spectral Pseudo-Divergence of Csiszar  (delta shape, delta intensity)
        :rtype: numpy.float
    """
     
    if (alpha <= 0):
        print("Csiszar measure of information : alpha parameter must be > 0")
        return(-1)
    
    ## extract the number of samples and number of spectra for each spectral set
    if (H1.ndim == 1):
        nbwaves1   = H1.size
        nbsamples1 = 1
        H1=H1.reshape((1,nbwaves1))
    else:
        [nbsamples1, nbwaves1] = H1.shape

    if (H2.ndim == 1):
        nbwaves2   = H2.size
        nbsamples2 = 1
        H2=H2.reshape((1,nbwaves1))
    else:
        [nbsamples2, nbwaves2] = H2.shape
    
    ## Verify that the spectral resolution are in accordance    
    if (nbwaves1 != nbwaves2):
        print("H1 and H2 don't have the same spectral band count")
        return (-1.0)
    
    ## Duplicate if necessary one of the two spectral set (being the reference)
    if (nbsamples1 == nbsamples2):
        nbsamples = nbsamples1
        nbwaves   = nbwaves1
    elif (nbsamples1 == 1):
        ## duplicate H1
        nbsamples = nbsamples2
        nbwaves   = nbwaves2
        H1 = np.tile(H1, nbsamples).reshape(nbsamples, nbwaves )     
    elif (nbsamples2 == 1):
        ## duplicate H2
        nbsamples = nbsamples1
        nbwaves   = nbwaves1
        H2 = np.tile(H2, nbsamples).reshape(nbsamples, nbwaves )
    else:
        print("H1 and H2 must have the same number of samples, or just 1(as reference)")
        print('nbsamples 1 :' + str(nbsamples1) + '   nbsamples 2 : ' + str(nbsamples2))
        
    ## Spectral integration using trapzale rule
    k1 = np.trapz(H1,dx = resolution, axis=0).reshape(nbsamples,1)
    k2 = np.trapz(H2,dx = resolution, axis=0).reshape(nbsamples,1)

    ## distance computation splitting the spectral difference in 2 sub-parts
    dist_shape  = k1*div_Csiszar( H1/k1, H2/k2, alpha, resolution).reshape(nbsamples,1) \
                + k2*div_Csiszar( H2/k2, H1/k1, alpha, resolution).reshape(nbsamples,1)
     
    dist_energy = (k1-k2)*np.log(k1/k2)

    dist = np.concatenate ( (dist_shape, dist_energy),axis=0 )
    dist = dist.reshape( nbsamples , 2)
    return dist