"""Filtering routines."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
#from numpy.core.numeric import newaxis
from scipy import signal
from kwiklib.utils.six.moves import range

from IPython import embed 

# -----------------------------------------------------------------------------
# Signal processing functions
# -----------------------------------------------------------------------------
def bandpass_filter(**prm):
    """Bandpass filtering."""
    rate = prm['sample_rate']
    order = prm['filter_butter_order']
    low = prm['filter_low']
    high = prm['filter_high']
    return signal.butter(order,
                         (low/(rate/2.), high/(rate/2.)),
                         'pass')

def apply_filter(x, filter=None):
    if x.shape[0] == 0:
        return x
    b, a = filter
#    print b, a, x
    try:
        out_arr = signal.filtfilt(b, a, x, axis=0)
    except TypeError:   
        out_arr = np.zeros_like(x)
        for i_ch in range(x.shape[1]):
            out_arr[:, i_ch] = signal.filtfilt(b, a, x[:, i_ch]) 
    return out_arr
    
def decimate(x):
    q = 16
    n = 50
    axis = 0
    
    b = signal.firwin(n + 1, 1. / q, window='hamming')
    a = 1.

    y = signal.lfilter(b, a, x, axis=axis)
    
    sl = [slice(None)] * y.ndim
    sl[axis] = slice(n // 2, None, q)
    
    return y[sl]


# -----------------------------------------------------------------------------
# Whitening
# -----------------------------------------------------------------------------
"""
  * Get the first chunk of data
  * Detect spikes the usual way
  * Compute mean on each channel on non-spike data
  * For every pair of channels:
      * estimate the covariance on non-spike data
  * Get the covariance matrix
  * Get its square root C' (sqrtm)
  * Get u*C' + (1-u)*s*Id, where u is a parameter, s the std of non-spike data 
    across all channels
  * Option to save or not whitened data in FIL
  * All spike detection is done on whitened data
  
"""

#def get_excludepairs_timesample(lst):
    #pairs = []
    #for i in range(len(lst)):
        #for j in range(i+1, len(lst)):
            #if (lst[i],lst[j]) not in pairs:
                #pairs.append((lst[i],lst[j]))
    #return pairs      

def cov_with_specified_meanvector(m,mean=None,bias=0,ddof=None):
    #Compute the covariance matrix - user specified mean vector
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")
      
    X = np.array(m, ndmin=2, dtype=float)
    axis = 1
    tup = (np.newaxis, slice(None))
    if mean is not None:
        X -= mean
    else:
        X -= X.mean(axis=1-axis)[tup]
       
    if ddof is None:
	if bias == 0:
	    ddof = 1
	else:
	    ddof = 0
	    
    N = X.shape[0]
    fact = float(N - ddof)
    return (np.dot(X.T, X.conj()) / fact).squeeze()
    

def get_noise_cov(chunk_fil,components):
    chunksize = chunk_fil.shape[0] # 20000
    chunk_binary = np.zeros_like(chunk_fil)
    for k,indlist in enumerate(components):
	indtemparray = np.array(indlist.items)
	chunk_binary[indtemparray[:,0],indtemparray[:,1]] = 1#all the samples that are not in a spike component are 0, rest 1
	
    chunk_binary_not = np.logical_not(chunk_binary) #all the samples that are not in a spike component are 1, rest 0
    chunk_fil_nospikes = np.multiply(chunk_fil, chunk_binary_not) # samples are zero in the spike component
    #cov_nospikes_naive = np.cov(chunk_fil_nospikes, rowvar = 0, ddof = 0)# ddof gives normalization, N-ddof
    nospikesmean = chunk_fil_nospikes.mean(axis = 0)
    
    contributing_pairs = np.einsum('ij, ik->ijk',chunk_binary_not,chunk_binary_not)
    normaliz = np.sum(contributing_pairs, axis = 0) #Normalization factor for each pair jk 
    
    chunk_binary_mean = np.einsum('mn,n->mn',chunk_binary,nospikesmean)
    chunk_filnospikes_plus_mean = chunk_fil_nospikes + chunk_binary_mean
    
    ##get covariance matrix of the filtered data, where the spikes have been replaced with the noise mean
    #cov_nospikes_almost = np.cov(chunk_filnospikes_plus_mean, rowvar = 0, ddof = chunksize-1)# ddof gives normalization, N-ddof
    cov_nospikes_almost = cov_with_specified_meanvector(chunk_filnospikes_plus_mean,mean = nospikesmean,ddof = chunksize-1)
    ##required normalization
    noisecov = np.divide(cov_nospikes_almost, normaliz)
    
    return noisecov
  
#
#In [47]: chunk_binary[936,:]
#Out[47]: 
#array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,
        #1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        #0.,  0.,  0.,  0.,  0.,  0.])

#In [48]: np.nonzero(chunk_binary[936,:])
#Out[48]: (array([10, 11, 12, 13]),)
#In [55]: get_excludepairs_timesample(list(np.nonzero(chunk_binary[936,:])[0])
   #....: )
#Out[55]: [(10, 11), (10, 12), (10, 13), (11, 12), (11, 13), (12, 13)]



def get_whitening_matrix(chunk_fil,components,epsilon_fudge=1E-18):
    noisecov = get_noise_cov(chunk_fil,components)
    d, V = np.linalg.eigh(noisecov)
    D = np.diag(1. / np.sqrt(d+epsilon_fudge))
    Whitening = np.dot(D, V.T)
    print np.dot(np.dot(Whitening,noisecov),Whitening.T) # sanity check
    
    return Whitening, noisecov
    

def whiten(X, whiteningmatrix):
    #if whiteningmatrix is None:
    #    whiteningmatrix = get_whitening_matrix(X)
    #embed()
    X_whitened = np.dot(X,whiteningmatrix)

    #X_whitened = np.dot(whiteningmatrix,X)   
    return X_whitened
    
    
    


