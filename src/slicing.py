from scipy import special
import numpy as np

# gaussian charge density
def rho(z): # [1/m] z = z_k/sigma_z normalized by bunch length in the frame where the slicing takes place
    #return np.exp(-z**2/(2)) / (np.sqrt(2*np.pi)*sigma_z)
    return np.exp(-z**2/(2)) / (np.sqrt(2*np.pi))  # dmitry also doesnt use sigma_z in slice.f90

def unicharge(n_slices):
    """
    Uniform charge slicing.
    """

    # these are units of sigma_z
    z_k_arr_unicharge = np.zeros(n_slices)  # should be n_slices long, ordered from + to -
    l_k_arr_unicharge = np.zeros(n_slices+1)  # bin edges, n_slices+1 long
    w_k_arr_unicharge = np.zeros(n_slices)  # bin weights, used for bunch intensity normalization
    half = int((n_slices + 1) / 2)
    n_odd = n_slices % 2
    w_k_arr_unicharge[:half] = 1 / n_slices  # fill up initial values, e.g. n_slices=300-> fill up elements [0,149]; 301: [0,150]
    l_k_arr_unicharge[0] = -5  # leftmost bin edge
    w_k_sum = 0 # sum of weights: integral of gaussian up to l_k
    rho_upper = 0 # start from top of distribution (positive end, l_upper=inf)
    
    # go from bottom end toward 0 (=middle of Gaussian)
    for j in range(half):
    
        w_k_sum += 2*w_k_arr_unicharge[j] # integrate rho up to and including bin j

        # get bin center
        if n_odd and j == half-1:  # center bin (z_c=0)
            z_k_arr_unicharge[j] = 0
        else:  # all other bins
            rho_lower = rho_upper

            # get upper bin boundary
            arg = w_k_sum - 1
            l_upper = np.sqrt(2)*special.erfinv(arg)
            l_k_arr_unicharge[j+1] = l_upper
            rho_upper = rho(l_upper) 
            
            # get z_k: center of momentum
            z_k_arr_unicharge[j] = (rho_upper - rho_lower) / w_k_arr_unicharge[j]
    
    # mirror for positive half
    z_k_arr_unicharge[half:] = -z_k_arr_unicharge[n_slices-half-1::-1]  # bin centers
    w_k_arr_unicharge[half:] =  w_k_arr_unicharge[n_slices-half-1::-1]  # bin weights, used for bunch intensity normalization
    l_k_arr_unicharge[half:] = -l_k_arr_unicharge[n_slices-half::-1]  # bin edges
    dz_k_arr_unicharge       = np.diff(l_k_arr_unicharge)  # for beamstrahlung
    l_k_arr_unicharge        = l_k_arr_unicharge[::-1]

    return z_k_arr_unicharge, l_k_arr_unicharge, w_k_arr_unicharge, dz_k_arr_unicharge


def unibin(n_slices):
    """
    Uniform bin slicing.
    """

    # these are units of sigma_z
    z_k_list_unibin = []  # should be n_slices long, ordered from + to -

    m = 1 if not n_slices%2 else 0

    # dmitry goes from +n_slices/2 to -n_slices/2-1 (50-(-51) for 101 slices); hirata goes from n_slices to 0
    for k in range(int(n_slices/2), -int(n_slices/2)-(1-m), -1):
    
        # slices extend from -N*sigma to +N*sigma
        N = 5
        z_k = (2*k - m) / (n_slices - 1) * N * special.erf(np.sqrt(n_slices / 6))
        z_k_list_unibin.append(z_k)

    z_k_arr_unibin = np.array(z_k_list_unibin)  # bin centers
    w_k_arr_unibin = np.exp(-z_k_arr_unibin**2/2) # proportional, but these are not yet not normalized
    w_k_arr_unibin = w_k_arr_unibin / np.sum(w_k_arr_unibin) # bin weights, used for bunch intensity normalization
    dz_i = -np.diff(z_k_arr_unibin)[0]
    l_k_arr_unibin = np.hstack([z_k_arr_unibin+dz_i/2, z_k_arr_unibin[-1]-dz_i/2])
    dz_k_array_unibin = np.ones(n_slices)*dz_i
    return z_k_arr_unibin, l_k_arr_unibin, w_k_arr_unibin, dz_k_array_unibin


def improved(n_slices):
    """
    This method is a mix between uniform bin and charge. It finds the slice centers by iteration.
    """

    # these are units of sigma_z
    z_k_arr_improved = np.zeros(n_slices)  # should be n_slices long, ordered from + to -
    l_k_arr_improved = np.zeros(n_slices+1)  # bin edges, n_slices+1 long
    w_k_arr_improved = np.zeros(n_slices)  # bin weights, used for bunch intensity normalization
    half = int((n_slices + 1) / 2)
    n_odd = n_slices % 2
    w_k_arr_improved[:half] = 1 / n_slices  # fill up initial values, e.g. n_slices=300-> fill up elements [0,149]; 301: [0,150]
    l_k_arr_improved[0] = -5  # leftmost bin edge

    k_max = min(1000, 20*n_slices)  # max iterations for l_k
    
    for i in range(k_max+1):
        w_k_sum = 0 # sum of weights: integral of gaussian up to l_k
        rho_upper = 0 # start from top of distribution (positive end, l_upper=inf)
        
        # go from bottom toward 0 (=middle of Gaussian)
        for j in range(half):
        
            w_k_sum += 2*w_k_arr_improved[j] # integrate rho up to including current bin
    
            # get z_k
            if n_odd and j == half-1:  # center bin (z_c=0)
                z_k_arr_improved[j] = 0
            else:  # all other bins
                rho_lower = rho_upper
    
                arg = w_k_sum - 1
                l_upper = np.sqrt(2)*special.erfinv(arg)
    
                l_k_arr_improved[j+1] = l_upper
                
                rho_upper = rho(l_upper)  # to cancel 1/sigma_z in rho
                
                # get z_k: center of momentum
                z_k_arr_improved[j] = (rho_upper - rho_lower) / w_k_arr_improved[j]
                
            # get w_k
            if i < k_max:
                w_k_arr_improved[j] = np.exp( -z_k_arr_improved[j]**2 / 4 )
        
        # renormalize w_k
        if i < k_max:
            w_int = 2*np.sum(w_k_arr_improved[:half]) - n_odd * w_k_arr_improved[half-1]
            w_k_arr_improved[:half] = w_k_arr_improved[:half] / w_int
    
    # mirror for negative half
    z_k_arr_improved[half:] = -z_k_arr_improved[n_slices-half-1::-1]  # bin centers
    w_k_arr_improved[half:] =  w_k_arr_improved[n_slices-half-1::-1]  # bin weights, used for bunch intensity normalization
    l_k_arr_improved[half:] = -l_k_arr_improved[n_slices-half::-1]  # bin edges
    dz_k_arr_improved       = np.diff(l_k_arr_improved)  # for beamstrahlung
    l_k_arr_improved        = l_k_arr_improved[::-1]
    
    return z_k_arr_improved, l_k_arr_improved, w_k_arr_improved, dz_k_arr_improved

