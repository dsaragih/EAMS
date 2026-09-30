from __future__ import annotations

import numpy as np
import scipy
import trimesh
import logging

import os
import math
from pathlib import Path
from typing import Any, Optional

from .laplace import *


DEFAULT_SIGNATURE_APPROX = "mesh"
DEFAULT_SIGNATURE_BASIS = 128
DEFAULT_SIGNATURE_CACHE_VERSION = "v1"


def _sample_cache_stem(data: Any) -> str:
    mesh_id = getattr(data, "mesh_id", None)
    if isinstance(mesh_id, str) and mesh_id:
        return Path(mesh_id).stem

    filename = getattr(data, "filename", None)
    if isinstance(filename, str) and filename:
        return Path(filename).stem

    num_nodes = int(getattr(data, "num_nodes", len(getattr(data, "pos", []))))
    return f"sample_{num_nodes}_nodes"


def _signature_cache_root(data: Any) -> Optional[Path]:
    cache_dir = getattr(data, "signature_cache_dir", None)
    if not cache_dir:
        return None
    return Path(str(cache_dir)).expanduser().resolve()


def get_signature_cache_path(
    data: Any,
    *,
    approx: str = DEFAULT_SIGNATURE_APPROX,
    n_basis: int = DEFAULT_SIGNATURE_BASIS,
    cache_version: str = DEFAULT_SIGNATURE_CACHE_VERSION,
) -> Optional[Path]:
    cache_root = _signature_cache_root(data)
    if cache_root is None:
        return None

    sample_stem = _sample_cache_stem(data)
    file_name = f"{sample_stem}_{approx}_{int(n_basis)}_{cache_version}.npz"
    return cache_root / file_name


def trimesh_from_data(data: Any) -> trimesh.Trimesh:
    face = getattr(data, "face", None)
    faces = None
    if face is not None:
        faces = face.T.detach().cpu() if hasattr(face, "T") else face
    return trimesh.Trimesh(
        vertices=data.pos.detach().cpu(),
        faces=faces,
        process=False,
    )


def load_or_build_signature_extractor(
    data: Any,
    *,
    approx: str = DEFAULT_SIGNATURE_APPROX,
    n_basis: int = DEFAULT_SIGNATURE_BASIS,
    cache_version: str = DEFAULT_SIGNATURE_CACHE_VERSION,
    cache_if_possible: bool = True,
    uncached_warning: bool = True,
) -> SignatureExtractor:
    cache_path = get_signature_cache_path(
        data,
        approx=approx,
        n_basis=n_basis,
        cache_version=cache_version,
    )

    if cache_path is not None and cache_path.exists():
        return SignatureExtractor(path=str(cache_path))
    elif cache_path is not None and cache_path.parent.exists():
        if uncached_warning:
            logging.warning(
                "Signature cache file not found at %s. Computing signature extractor from mesh. This may take a while.",
                cache_path,
            )
    else:
        if uncached_warning:
            logging.warning(
                "Signature cache directory not found at %s. Computing signature extractor from mesh without caching. This may take a while.",
                cache_path.parent if cache_path is not None else None,
            )

    extractor = SignatureExtractor(
        trimesh_from_data(data),
        n_basis=n_basis,
        approx=approx,
    )

    if cache_if_possible and cache_path is not None:
        extractor.save(str(cache_path))

    return extractor


class SignatureExtractor(object):
    def __init__(self, mesh=None, n_basis=-1, approx="cotangens", path=None):
        self._initialized = False
        if path is not None:
            self.load(path)
        elif n_basis > 0 and mesh is not None:
            self.initialize(mesh, n_basis, approx)

    def initialize(self, mesh: trimesh.Trimesh, n: int, approx="cotangens"):
        """Initializes the SignatureExtractor by computing eigenvectors and eigenvalues
        of the discrete laplace operator.

        Args:
            mesh (trimesh.Trimesh): Mesh to extract signatures from
            n (int): Number of eigenvalues and eigenvectors to compute.
            approx (str, optional): Laplace operator approximation to use.
                                    Must be in ['beltrami', 'cotangens', 'mesh', 'fem']. Defaults to 'cotangens'.
        """
        self.W, self.M = get_laplace_operator_approximation(mesh, approx)
        self.n_basis = min(len(mesh.vertices) - 1, n)

        sigma = -0.01
        shifted = (self.W - sigma * self.M).tocsc()
        try:
            from sksparse.cholmod import cholesky

            chol = cholesky(shifted)
            op_inv = scipy.sparse.linalg.LinearOperator(
                matvec=chol, shape=self.W.shape, dtype=self.W.dtype
            )
            logging.info(
                "Using CHOLMOD (scikit-sparse) for efficient eigen decomposition"
            )
        except Exception as e:
            # CHOLMOD may be missing OR may fail factorization (e.g. not positive definite).
            # Fall back to SciPy's sparse factorization.
            logging.info(
                "CHOLMOD unavailable or failed (%s); falling back to scipy.sparse.linalg.factorized",
                type(e).__name__,
            )
            solve = scipy.sparse.linalg.factorized(shifted)
            op_inv = scipy.sparse.linalg.LinearOperator(
                matvec=solve, shape=self.W.shape, dtype=self.W.dtype
            )

        self.evals, self.evecs = scipy.sparse.linalg.eigsh(
            self.W, self.n_basis, self.M, sigma=sigma, OPinv=op_inv
        )

        self._initialized = True

    def heat_signatures(self, dim: int, return_times=False, times=None):
        """Compute the heat signature for all vertices

        Args:
            dim (int): Dimensionality (timesteps) of the signature.
            return_times (bool, optional): If True the function returns a tuple (signature, timesteps)
                                           otherwise only the signature is returned. Defaults to False.
            times (arraylike, optional): Timesteps used for signature computation.
                                         If None the times are spaced logarithmically. Defaults to None.

        Note:
            This signature is based on 'A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion'
            by Jian Sun et al (http://www.lix.polytechnique.fr/~maks/papers/hks.pdf)

        Returns:
            Returns an array of shape (#vertices, dim) containing the heat signatures of every vertex.
            If return_times is True this function returns a tuple (Signature, timesteps).
        """
        assert self._initialized, "Signature extractor was not initialized"

        evals = np.asarray(self.evals).reshape(-1)
        if evals.size < 3:
            raise RuntimeError(
                f"Not enough eigenvalues for HKS (need >=3, got {evals.size})"
            )

        # Numerical robustness:
        # - In theory evals should be >= 0, but in practice we can see tiny negatives
        #   (or outright invalid values) from degenerate meshes / near-singular mass.
        # - HKS requires strictly positive time samples.
        eps = 1e-12
        evals_pos = np.maximum(np.abs(evals), eps)

        if times is None:
            # Use the standard heuristic times based on extremal eigenvalues.
            # Use evals_pos to ensure positivity.
            lam_max = float(evals_pos[-1])
            lam_2 = float(evals_pos[1])

            # Fallback if spectrum is pathological
            if (
                not (np.isfinite(lam_max) and np.isfinite(lam_2))
                or lam_max <= 0.0
                or lam_2 <= 0.0
            ):
                tmin, tmax = 1e-2, 1e2
            else:
                tmin = 4.0 * math.log(10.0) / lam_max
                tmax = 4.0 * math.log(10.0) / lam_2

            # Ensure increasing and valid
            if (
                not (np.isfinite(tmin) and np.isfinite(tmax))
                or tmin <= 0.0
                or tmax <= 0.0
            ):
                tmin, tmax = 1e-2, 1e2
            if tmax < tmin:
                tmin, tmax = tmax, tmin

            times = np.geomspace(tmin, tmax, dim)
        else:
            times = np.array(times).flatten()
            assert (
                len(times) == dim
            ), f"Requested feature dimension and time steps array do not match: {dim} and {len(times)}"

        # Make sure we don't accidentally feed non-positive times (geomspace requires > 0).
        times = np.asarray(times, dtype=np.float64)
        times = np.maximum(times, eps)

        phi2 = np.square(self.evecs[:, 1:])

        # Compute exp(-lambda * t) safely. With evals_pos >= eps, exponent is <= 0.
        exponent = -evals_pos[1:, None] * times[None]
        # Clip just in case upstream produces very large magnitude values.
        exponent = np.clip(exponent, -700.0, 50.0)
        exp = np.exp(exponent)
        s = np.sum(phi2[..., None] * exp[None], axis=1)
        heat_trace = np.sum(exp, axis=0)
        heat_trace = np.maximum(heat_trace, eps)
        s = s / heat_trace[None]

        # Guarantee finite output
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

        if return_times:
            return s, times
        else:
            return s

    def wave_signatures(
        self, dim: int, variance: float = 6.0, return_energies=False, energies=None
    ):
        """Compute the wave signature for all vertices

        Args:
            dim (int): Dimensionality (energy spectra) of the signature.
            variance (float): variance of the WKS gaussian (wih respect to the difference of the two first eigenvalues).
                          For easy or precision tasks (eg. matching with only isometric deformations) you can take it smaller
            return_energies (bool, optional): If True the function returns a tuple (signature, energies)
                                              otherwise only the signature is returned. Defaults to False.
            energies (arraylike, optional): Energie spectra used for signature computation.
                                            If None the energy is linearly spaced. Defaults to None.

        Note:
            This signature is based on 'The Wave Kernel Signature: A Quantum Mechanical Approach to Shape Analysis'
            by Mathieu Aubry et al (https://vision.informatik.tu-muenchen.de/_media/spezial/bib/aubry-et-al-4dmod11.pdf)

        Returns:
            Returns an array of shape (#vertices, dim) containing the heat signatures of every vertex.
            If return_times is True this function returns a tuple (Signature, timesteps).
        """

        assert self._initialized, "Signature extractor was not initialized"

        log_e = np.log(np.maximum(np.abs(self.evals), 1e-6))

        if energies is None:
            energies = np.linspace(log_e[1], log_e[-1] / 1.02, dim)
        else:
            energies = np.array(energies).flatten()
            assert (
                len(energies) == dim
            ), f"Requested featrue dimension and energies array do not match: {dim} and {len(energies)}"

        sigma = variance * (energies[1] - energies[0])
        phi2 = np.square(self.evecs)
        exp = np.exp(-np.square(energies[None] - log_e[:, None])) / (
            2.0 * sigma * sigma
        )
        s = np.sum((phi2[:, :, None] * exp[None]), axis=1)
        energy_trace = np.sum(exp, axis=0)
        s = s / energy_trace[None]

        if return_energies:
            return s, energies
        else:
            return s

    def signatures(
        self, dim: int, kernel: str, return_x_ticks=False, x_ticks=None, **kwargs
    ):
        """Computes a signature for each vertex

        Args:
            dim (int): Dimensionality (energy spectra) of the signature.
            kernel (str): Feature kernel used must be in ['heat', 'wave'].
            return_x_ticks (bool, optional): If True the function returns a tuple (signature, x_ticks)
                                            otherwise only the signature is returned. Defaults to False.
            x_ticks (arraylike, optional): Variable used for the feature dimension. Defaults to None.

        Returns:
            Returns an array of shape (#vertices, dim) containing the mesh signatures of every vertex.
            If return_x_ticks is True this function returns a tuple (signature, x_ticks).
        """
        assert kernel in [
            "heat",
            "wave",
        ], f"Invalid kernel type '{kernel}'. Must be in ['heat', 'wave']"

        if kernel == "heat":
            return self.heat_signatures(dim, return_x_ticks, x_ticks)
        else:
            return self.wave_signatures(
                dim, return_energies=return_x_ticks, energies=x_ticks, **kwargs
            )

    def heat_distances(
        self, query, dim: int, return_signature=False, times=None, cutoff=1.0
    ):
        """Compute distances of all vertices to vertices in query based on heat signature

        Note:
            We use the L2 norm of the feature vectors.

        Args:
            query (int, arrylike): queried indices
            dim (int): target index
            return_signature (bool, optional): If True the function returns a tuple (distances, signatures). Defaults to False
            times (None, arraylike, optional): Time steps used for signature computation. Defaults to None
            cutoff (float, optional): Fraction of dimensionality to use for distance computation. Defaults to 1.0

        Returns:
            An array of shape (#vertices, len(query)) holding the heat signature distance
            of each vertex to the queried vertices. If return_signature is True this function returns a
            tuple (distances, signature).
        """

        e = max(0, int(cutoff * dim)) + 1
        a = self.heat_signatures(dim, times=times)[..., :e]
        b = np.atleast_2d(a[np.array(query, dtype=np.int64)])
        a_dim = a.ndim
        b_dim = b.ndim
        if a_dim == 1:
            a = a.reshape(1, 1, a.shape[0])
        if a_dim >= 2:
            a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
        if b_dim > 2:
            b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
        diff = a - b
        dist_arr = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        dist_arr = np.squeeze(dist_arr)
        if return_signature:
            return dist_arr, a[:, 0, :]
        else:
            return dist_arr

    def wave_distance(
        self, query, dim: int, return_signatures=False, energies=None, cutoff=1.0
    ):
        """Compute distances of all vertices to vertices in query based on the wave signature

        Note:
            We use the L1 norm of the feature vectors.

        Args:
            query (int, arrylike): queried indices
            dim (int): target index
            return_signature (bool): If True the function returns a tuple (distances, signatures). Defaults to False
            times (None, arraylike): Time steps used for signature computation. Defaults to None
            cutoff (float, optional): Fraction of dimensionality to use for distance computation. Defaults to 1.0

        Returns:
            An array of shape (#vertices, len(query)) holding the wave signature distance
            of each vertex to the queried vertices. If return_signature is True this function returns a
            tuple (distances, signature).
        """
        e = max(0, int(dim * cutoff)) + 1
        a = self.wave_signatures(dim, energies=energies)[..., :e]
        b = np.atleast_2d(a[np.array(query, dtype=np.int64)])
        a_dim = a.ndim
        b_dim = b.ndim
        if a_dim == 1:
            a = a.reshape(1, 1, a.shape[0])
        if a_dim >= 2:
            a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
        if b_dim > 2:
            b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
        diff = a - b
        dist_arr = np.sum(np.abs(diff), axis=-1)
        dist_arr = np.squeeze(dist_arr)
        if return_signatures:
            return dist_arr, a[:, 0, :]
        else:
            return dist_arr

    def feature_distance(
        self,
        query,
        dim: int,
        kernel: str,
        return_signatures=False,
        x_ticks=None,
        cutoff=1.0,
    ):
        """Compute distances of all vertices to vertices in query based on a mesh signature

        Args:
            query (int, arrylike): queried indices
            dim (int): target index
            kernel (str): Signature type to use. Must be in ['heat', 'wave']
            return_signature (bool): If True the function returns a tuple (distances, signatures). Defaults to False
            x_ticks (None, arraylike): x_ticks used for signature computation. Defaults to None
            cutoff (float, optional): Fraction of dimensionality to use for distance computation. Defaults to 1.0

        Returns:
            An array of shape (#vertices, len(query)) holding the signature distance
            of each vertex to the queried vertices. If return_signature is True this function returns a
            tuple (distances, signature).
        """
        assert kernel in [
            "heat",
            "wave",
        ], f"Invalid kernel type '{kernel}'. Must be in ['heat', 'wave']"

        if kernel == "heat":
            return self.heat_distances(query, dim, return_signatures, x_ticks, cutoff)
        else:
            return self.wave_distance(query, dim, return_signatures, x_ticks, cutoff)

    def save(self, path: str):
        """Save the laplace spectrum (eigenvalues and eigenvectors) to a .npz file

        Args:
            path (str): Output filename.
        """
        assert self._initialized, "Signature extractor was not initialized"

        folder, _ = os.path.split(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.savez_compressed(path, evals=self.evals, evecs=self.evecs)

    def load(self, path):
        """Initialize by loading the laplace spectrum (eigenvalues and eigenvectors) from a .npz file

        Args:
            path (str): Input filename.
        """
        assert os.path.exists(path), "File not found"

        data = np.load(path)
        assert "evals" in data and "evecs" in data, "Error loading file"

        self.evals = data["evals"]
        self.evecs = data["evecs"]
        self.n_basis = len(self.evals)
        self._initialized = True

    @property
    def stiffness(self):
        if not self._initialized:
            raise RuntimeError("Extractor not initialized")
        return self.W

    @property
    def mass(self):
        if not self._initialized:
            raise RuntimeError("Extractor not initialized")
        return self.M

    @property
    def spectrum(self):
        if not self._initialized:
            raise RuntimeError("Extractor not initialized")
        return self.evals
