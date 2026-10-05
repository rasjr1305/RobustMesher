from pathlib import Path
import numpy as np
import segyio


def read_segy_velocity_model(fname):
    """Read a velocity model array from a SEGY file.

    Parameters
    ----------
    fname : str
        Path to the SEGY filename.

    Returns
    -------
    tuple
        A tuple containing:
        - vp (ndarray): 2D Velocity model array of shape (n_samples, n_traces).
        - n_traces (int): Number of traces (columns).
        - n_samples (int): Number of samples per trace (rows).
    """

    print(f"Reading SEGY file: {fname}")

    # Open SEGY file
    with segyio.open(fname, 'r', ignore_geometry=True) as segy:

        n_traces = len(segy.trace)
        n_samples = len(segy.samples)

        # Read traces directly into array
        vp = np.zeros((n_samples, n_traces))
        for i in range(n_traces):
            vp[:, i] = segy.trace[i]
    print(f"Final velocity range: {vp.min():.1f} - {vp.max():.1f}")
    return vp, n_samples, n_traces


def _read_velocity_cube(fname, nz, nx, ny, dtype_string, order):
    """Memory-map a binary velocity model as a ``(z, x, y)`` cube.

    Parameters
    ----------
    fname : str or pathlib.Path
        Path to the binary velocity model.
    nz : int
        Number of velocity samples in the z direction.
    nx : int
        Number of velocity samples in the x direction.
    ny : int
        Number of velocity samples in the y direction.
    dtype_string : str
        NumPy dtype string including the required byte order.
    order : {"C", "F"}
        Array reshape order used for the binary velocity model.

    Returns
    -------
    tuple
        Memory-mapped velocity cube and validated ``nz``, ``nx``, and ``ny`` sizes.
    """
    try:
        nz, nx, ny = int(nz), int(nx), int(ny)
    except Exception as exc:
        raise RuntimeError("nz, nx, and ny must be valid integer arguments.") from exc

    path = Path(fname)
    if not path.exists():
        raise FileNotFoundError(f"Binary not found: {path}")

    mm = np.memmap(path, dtype=np.dtype(dtype_string), mode="r")
    expected = nz * nx * ny
    if mm.size != expected:
        raise ValueError(
            f"File size mismatch for {path}:\n"
            f"  got {mm.size} floats, expected {expected} for "
            f"shape (nz,nx,ny)=({nz},{nx},{ny})"
        )

    return mm.reshape((nz, nx, ny), order=order), nz, nx, ny
