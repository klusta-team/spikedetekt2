"""PCA routines."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from scipy import signal
from kwiklib.utils.six.moves import range


# -----------------------------------------------------------------------------
# PCA functions
# -----------------------------------------------------------------------------
def compute_pcs(x, npcs=None, masks=None):
    """Compute the PCs of an array x, where each row is an observation.
    x can be a 2D or 3D array. In the latter case, the PCs are computed
    and concatenated iteratively along the last axis."""
    # If x is a 3D array, compute the PCs by iterating over the last axis.
    if x.ndim == 3:
        if masks is None:
            return np.dstack([compute_pcs(x[..., i], npcs=npcs, masks=None)
                              for i in range(x.shape[-1])])
        else:
            # We pass the masks to compute_pcs for each row of the 3D array.
            assert isinstance(masks, np.ndarray)
            assert masks.ndim == 2
            assert masks.shape[0] == x.shape[0]  # number of spikes
            assert masks.shape[1] == x.shape[-1]  # number of channels
            return np.dstack([compute_pcs(x[..., i], npcs=npcs,
                                          masks=masks[..., i])
                              for i in range(x.shape[-1])])
    # Now, we assume x is a 2D array.
    assert x.ndim == 2
    # Check the masks.
    if masks is not None:
        assert masks.ndim == 1
        assert masks.shape[0] == x.shape[0]  # number of spikes
        # Only select those rows in x that are *unmasked* (mask>0).
        x = np.compress(masks>0, x, axis=0)
    if len(x) <= 1:
        return np.zeros((npcs, x.shape[-1]), dtype=np.float32)
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


