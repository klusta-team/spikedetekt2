"""PCA routines."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from scipy import signal
from spikedetekt2.utils.six.moves import range


# -----------------------------------------------------------------------------
# PCA functions
# -----------------------------------------------------------------------------
def compute_pcs(x, npcs=None):
    """Compute the PCs of an array x, where each row is an observation.
    x can be a 2D or 3D array. In the latter case, the PCs are computed
    and concatenated iteratively along the last axis."""
    # If x is a 3D array, compute the PCs by iterating over the last axis.
    if x.ndim == 3:
        return np.dstack([compute_pcs(x[..., i], npcs=npcs) 
                              for i in range(x.shape[-1])])
    # Now, we assume x is a 2D array.
    assert x.ndim == 2
    # Take the covariance matrix.
    cov_ss = np.cov(x.astype(np.float64), rowvar=0)
    # Compute the eigenelements.
    vals, vecs = np.linalg.eigh(cov_ss)
    pcs = vecs.T.astype(np.float32)[np.argsort(vals)[::-1]]
    # Take the first npcs components.
    if npcs is not None:
        return pcs[:npcs,...]
    else:
        return pcs
    
def project_pcs(x, pcs):
    """Project data points onto principal components.
    
    Arguments:
      * x: a 2D array.
      * pcs: the PCs as returned by `compute_pcs`.
    
    """
    x_proj = np.einsum('ijk,jk->ki', pcs, x)  # Notice the transposition.
    x_proj *= 100.
    return x_proj

    