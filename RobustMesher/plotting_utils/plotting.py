from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def boundary_faces_from_cells(cells, local_faces):
    faces = np.concatenate([cells[:, face] for face in local_faces], axis=0)
    keys = np.sort(faces, axis=1)
    _, first_index, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    return faces[first_index[counts == 1]]


def _as_mesh(mesh):
    """Accept either a meshio Mesh or an exported mesh filename."""
    if isinstance(mesh, (str, Path)):
        import meshio

        mesh = meshio.read(mesh)
    if not hasattr(mesh, "points") or not hasattr(mesh, "cells"):
        raise TypeError("mesh must be a meshio Mesh or an exported mesh filename.")
    if len(mesh.points) == 0:
        raise ValueError("The mesh contains no points.")
    return mesh


def _cell_blocks(mesh):
    for block in mesh.cells:
        # meshio CellBlocks expose attributes; tuples also work for simple callers.
        if hasattr(block, "type"):
            yield block.type, np.asarray(block.data, dtype=np.int64)
        else:
            yield block[0], np.asarray(block[1], dtype=np.int64)


def mesh_boundary_faces(mesh):
    """Return exterior triangular and quadrilateral faces of a volume mesh.
    """
    mesh = _as_mesh(mesh)
    triangle_blocks, quad_blocks = [], []
    explicit_triangles, explicit_quads = [], []
    has_volumes = False

    for cell_type, cells in _cell_blocks(mesh):
        if not len(cells):
            continue
        if cell_type.startswith("tetra"):
            has_volumes = True
            for face in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]:
                triangle_blocks.append(cells[:, face])
        elif cell_type.startswith("hexahedron"):
            has_volumes = True
            for face in [
                [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
            ]:
                quad_blocks.append(cells[:, face])
        elif cell_type.startswith("wedge"):
            has_volumes = True
            for face in [[0, 1, 2], [3, 4, 5]]:
                triangle_blocks.append(cells[:, face])
            for face in [[0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]]:
                quad_blocks.append(cells[:, face])
        elif cell_type.startswith("pyramid"):
            has_volumes = True
            for face in [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]:
                triangle_blocks.append(cells[:, face])
            quad_blocks.append(cells[:, [0, 1, 2, 3]])
        elif cell_type.startswith("triangle"):
            explicit_triangles.append(cells[:, :3])
        elif cell_type.startswith("quad"):
            explicit_quads.append(cells[:, :4])

    if not has_volumes:
        triangle_blocks = explicit_triangles
        quad_blocks = explicit_quads

    def exterior(blocks, width):
        if not blocks:
            return np.empty((0, width), dtype=np.int64)
        faces = np.vstack(blocks)
        return boundary_faces_from_cells(faces, [list(range(width))])

    triangles = exterior(triangle_blocks, 3)
    quads = exterior(quad_blocks, 4)
    if not len(triangles) and not len(quads):
        raise ValueError("No supported surface or volume cells were found.")
    return triangles, quads


def plot_mesh(mesh, dimension=2, ax=None, show=True):
    """Plot a meshio Mesh, or an exported mesh file, and return its axes.
    """
    mesh = _as_mesh(mesh)
    coordinates = np.asarray(mesh.points, dtype=float)
    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3.")
    if coordinates.ndim != 2 or coordinates.shape[1] < dimension:
        raise ValueError("The mesh point coordinates do not match dimension.")

    if dimension == 2:
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 8), dpi=150)
        triangles, quads = [], []
        for cell_type, cells in _cell_blocks(mesh):
            if cell_type.startswith("triangle"):
                triangles.append(cells[:, :3])
            elif cell_type.startswith("quad"):
                quads.append(cells[:, :4])
        if not triangles and not quads:
            raise ValueError("No triangular or quadrilateral cells were found.")
        triangles = np.vstack(triangles) if triangles else np.empty((0, 3), int)
        quads = np.vstack(quads) if quads else np.empty((0, 4), int)

        x = coordinates[:, 1]
        z = -coordinates[:, 0]
        xy = np.c_[x, z]

        if len(triangles):
            ax.triplot(x, z, triangles, linewidth=0.8)

        if len(quads):
            p = xy[quads]
            lines = np.concatenate([
                p[:, [0, 1]],
                p[:, [1, 2]],
                p[:, [2, 3]],
                p[:, [3, 0]],
            ])
            ax.add_collection(LineCollection(lines, linewidths=0.8))

        ax.set(
            xlabel="x (m)",
            ylabel="depth z (m)",
            title="Spyro 2D mesh",
            xlim=(x.min(), x.max()),
            ylim=(z.max(), z.min()),
        )
        ax.set_aspect("equal")
    else:
        if ax is None:
            fig = plt.figure(figsize=(12, 8), dpi=150)
            ax = fig.add_subplot(111, projection="3d")
        elif not hasattr(ax, "add_collection3d"):
            raise ValueError("For dimension=3, ax must have projection='3d'.")
        triangles, quads = mesh_boundary_faces(mesh)

        # Spyro order: (z, x, y) -> plot order: (x, y, z)
        xyz = coordinates[:, [1, 2, 0]]

        if len(triangles):
            triangle_surface = Poly3DCollection(
                xyz[triangles],
                facecolors="white",
                edgecolors="black",
                linewidths=0.18,
                alpha=1.0,
            )
            ax.add_collection3d(triangle_surface)

        if len(quads):
            quad_surface = Poly3DCollection(
                xyz[quads],
                facecolors="white",
                edgecolors="black",
                linewidths=0.18,
                alpha=1.0,
            )
            ax.add_collection3d(quad_surface)

        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        span = np.maximum(maxs - mins, 1.0)

        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_box_aspect(span)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title("Spyro 3D mesh")
        ax.view_init(elev=24, azim=-55)

    ax.figure.tight_layout()
    if show:
        plt.show()
    return ax
