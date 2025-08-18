import numpy as np

class PCAFromScratch:
    def __init__(self, n_components=None, keep_variance=None, whiten=False):
        
        assert (n_components is None) ^ (keep_variance is None), "Set exactly one of n_components or keep_variance."
        self.n_components = n_components
        self.keep_variance = keep_variance
        self.whiten = whiten
        # learned params
        self.mean_ = None
        self.components_ = None      # shape (n_comp, D)
        self.eigvals_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
      
        # center
        self.mean_ = X.mean(axis=0, keepdims=True)   # (1, D)
        Xc = X - self.mean_

        # use SVD (stable): Xc = U S Vt, PCs = rows of Vt
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        eigvals = (S**2) / (X.shape[0] - 1)         # eigenvalues of covariance
        total_var = eigvals.sum()
        ratio = eigvals / total_var

        if self.keep_variance is not None:
            cumsum = np.cumsum(ratio)
            k = np.searchsorted(cumsum, self.keep_variance) + 1
        else:
            k = int(self.n_components)

        self.components_ = Vt[:k, :]                # (k, D)
        self.eigvals_ = eigvals[:k]
        self.explained_variance_ratio_ = ratio[:k]
        return self

    def transform(self, X):
        Xc = X - self.mean_
        Z = Xc @ self.components_.T                 # (N, k)
        if self.whiten:
            Z = Z / np.sqrt(self.eigvals_ + 1e-12)
        return Z

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @property
    def n_comp_(self):
        return 0 if self.components_ is None else self.components_.shape[0]