from contextvars import ContextVar

import meshio

_prepared_sizing = ContextVar("RobustMesher_prepared_sizing", default=None)


def create_sizing_function(**kwargs):
    prepared = _prepared_sizing.get()
    if prepared is not None:
        return prepared
    from ..sizing_utils.sizing2d import create_sizing_function as original
    return original(**kwargs)


def create_sizing_function3D(**kwargs):
    prepared = _prepared_sizing.get()
    if prepared is not None:
        return prepared
    from ..sizing_utils.sizing3d import create_sizing_function3D as original
    return original(**kwargs)


def read_mesh(filename, comm=None):
    """Load the exported Gmsh mesh with its original physical groups."""
    return meshio.read(filename)
