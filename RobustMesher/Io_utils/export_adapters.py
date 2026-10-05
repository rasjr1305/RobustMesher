from pathlib import Path
import numpy as np
import meshio


def create_bin_from_grid(velocity, filename, byte_order="big",
                         axes_order_sort="F", dtype="float32"):
    """Write a 2D/3D array using binary operations.
    """
    if byte_order not in ("big", "little"):
        raise ValueError("byte_order must be 'big' or 'little'.")
    if axes_order_sort not in ("C", "F"):
        raise ValueError("axes_order_sort must be 'C' or 'F'.")
    vp = np.asarray(velocity)
    if vp.ndim not in (2, 3):
        raise ValueError("velocity must be a 2D or 3D array.")
    binary_dtype = np.dtype(dtype).newbyteorder(">" if byte_order == "big" else "<")
    np.asarray(vp, dtype=binary_dtype).ravel(order=axes_order_sort).tofile(filename)
    return str(filename)


def export_mesh(mesh, filename, file_format=None):
    """Export a loaded mesh; generation already writes the original Gmsh file."""
    if file_format is None and Path(filename).suffix.lower() == ".msh":
        file_format = "gmsh22"
    meshio.write(filename, mesh, file_format=file_format)
    return str(filename)
