from collections import defaultdict, deque
import numpy as np
from .gmsh3d_helpers import (
    boundary_faces_of_volume,
    create_hyperellipsoid_volume,
    generate_rectangular_padding_no_water,
    generate_structured_rectangular_padding_water,
    generate_water_interface_volumes,
)


def align_water_columns3d(gmsh, padding_type=None):
    """Align linear, layered water hexes in place; return their Gmsh node tags."""
    if padding_type not in (None, "rectangular"):
        raise ValueError("Water alignment supports None or rectangular padding.")
    suffix = "_with_padding" if padding_type == "rectangular" else ""
    names = ("Water" + suffix, "Subsurface" + suffix)
    groups = {
        gmsh.model.getPhysicalName(dim, tag): tag
        for dim, tag in gmsh.model.getPhysicalGroups(3)
    }
    if any(name not in groups for name in names):
        raise ValueError(f"Water alignment requires physical groups {names}.")
    volumes, boundaries = [], []
    for name in names:
        vols = gmsh.model.getEntitiesForPhysicalGroup(3, groups[name])
        volumes.append(vols)
        boundaries.append(set(gmsh.model.getBoundary(
            [(3, int(v)) for v in vols], combined=True, oriented=False,
        )))
    interface = set()
    for dim, face in boundaries[0] & boundaries[1]:
        tags, _, _ = gmsh.model.mesh.getNodes(dim, face, True, False)
        interface.update(map(int, tags))
    adjacency = defaultdict(set)
    for volume in volumes[0]:
        if set(gmsh.model.mesh.getElementTypes(3, int(volume))) != {5}:
            raise ValueError("Water alignment requires only 8-node hexahedra.")
        edges = gmsh.model.mesh.getElementEdgeNodes(5, int(volume), primary=True)
        for a, b in np.asarray(edges).reshape(-1, 2):
            a, b = int(a), int(b)
            adjacency[a].add(b)
            adjacency[b].add(a)
    if not interface or not interface.issubset(adjacency):
        raise ValueError("No valid shared water/subsurface interface was found.")
    owner = {node: (node, 0) for node in interface}
    queue = deque(interface)
    while queue:
        node = queue.popleft()
        root, layer = owner[node]
        for other in adjacency[node]:
            if other not in owner:
                owner[other] = (root, layer + 1)
                queue.append(other)
            elif owner[other][1] == layer + 1 and owner[other][0] != root:
                raise ValueError("Water topology does not define unique columns.")
    if len(owner) != len(adjacency):
        raise ValueError("Some water nodes are disconnected from the interface.")
    tags, coordinates = gmsh.model.mesh.getNodesForPhysicalGroup(3, groups[names[0]])
    coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 3)
    points = {int(tag): point for tag, point in zip(tags, coordinates)}
    if set(points) != set(owner) or not np.isfinite(coordinates).all():
        raise ValueError("Invalid water-node coordinates or connectivity.")
    for node, (root, _) in owner.items():
        if not np.array_equal(points[node][:2], points[root][:2]):
            point = points[node].copy()
            point[:2] = points[root][:2]
            gmsh.model.mesh.setNode(node, point.tolist(), [])
    return set(owner)


def define_winslow_points_3d(
    gmsh,
    points_3d,
    tag_to_index,
    length_x,
    length_y,
    depth_z,
    padding_type,
    padding_x,
    padding_y,
    padding_z,
    water_interface,
    tol=2.0,
):
    """Classify structured-mesh nodes by the coordinates they may move during Winslow smoothing.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    points_3d : numpy.ndarray
        Mesh-node coordinates with shape ``(N, 3)``.
    tag_to_index : dict
        Mapping from Gmsh node tags to zero-based point indices.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth; the model occupies negative z.
    padding_type : str or None
        Padding geometry, either ``None`` or ``"rectangular"`` for Winslow.
    padding_x : float
        Padding thickness in the x direction.
    padding_y : float
        Padding thickness in the y direction.
    padding_z : float
        Bottom padding thickness in the z direction.
    water_interface : bool
        Whether the water/subsurface interface is geometrically delimited.
    tol : float
        Coordinate tolerance used to identify boundary planes.

    Returns
    -------
    dict
        Sets of locked, directionally movable, fully movable, water, and union movable
        nodes.

    Notes
    -----
    Water nodes are locked when a delimited water volume is present. Boundary
    intersections are constrained so corners are fixed, edges move tangentially, faces
    move in-plane, and interior nodes move freely.
    """
    points_3d = np.asarray(points_3d, dtype=float)
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3).")

    if padding_type not in (None, "rectangular"):
        raise ValueError(
            "3-D Winslow point selection supports only padding_type=None "
            "or padding_type='rectangular'; 'hyperelliptical' is not "
            "supported for structured Winslow smoothing."
        )

    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("Winslow point-selection tolerance must be positive.")

    locked = set()
    move_X_only = set()
    move_Y_only = set()
    move_Z_only = set()
    move_all = set()
    water_nodes = set()

    if water_interface:
        water_group_name = (
            "Water_with_padding"
            if padding_type == "rectangular"
            else "Water"
        )

        group_found = False
        for dim, physical_tag in gmsh.model.getPhysicalGroups(dim=3):
            if gmsh.model.getPhysicalName(dim, physical_tag) != water_group_name:
                continue

            group_found = True
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, physical_tag)
            for entity_tag in entities:
                node_tags, _, _ = gmsh.model.mesh.getNodes(
                    dim,
                    entity_tag,
                    includeBoundary=True,
                )
                for node_tag in node_tags:
                    node_index = tag_to_index.get(int(node_tag))
                    if node_index is not None:
                        water_nodes.add(node_index)

        if not group_found:
            raise RuntimeError(
                f"Physical volume group {water_group_name!r} was not found "
                "while selecting 3-D Winslow nodes."
            )

        if not water_nodes:
            raise RuntimeError(
                f"Physical volume group {water_group_name!r} contains no "
                "mesh nodes for 3-D Winslow smoothing."
            )

        if len(water_nodes) == len(points_3d):
            raise RuntimeError(
                f"Every mesh node was classified as {water_group_name}. "
                "No subsurface nodes remain available for Winslow smoothing."
            )

    x_planes = [0.0, float(length_x)]
    y_planes = [0.0, float(length_y)]
    z_planes = [-abs(float(depth_z))]

    if padding_type == "rectangular":
        x_planes.extend(
            [
                -float(padding_x),
                float(length_x) + float(padding_x),
            ]
        )
        y_planes.extend(
            [
                -float(padding_y),
                float(length_y) + float(padding_y),
            ]
        )
        z_planes.append(
            -abs(float(depth_z)) - float(padding_z)
        )

    if not water_interface:
        z_planes.append(0.0)

    def on_any_plane(value, planes):
        """Test whether a coordinate lies on any constrained plane within tolerance.

        Parameters
        ----------
        value : float
            Coordinate to classify.
        planes : sequence of float
            Candidate plane coordinates.

        Returns
        -------
        bool
            ``True`` when the coordinate is within tolerance of any supplied plane.
        """
        return any(abs(value - plane) < tol for plane in planes)

    for node_index, (x_coord, y_coord, z_coord) in enumerate(points_3d):
        if node_index in water_nodes:
            locked.add(node_index)
            continue

        on_x_plane = on_any_plane(x_coord, x_planes)
        on_y_plane = on_any_plane(y_coord, y_planes)
        on_z_plane = on_any_plane(z_coord, z_planes)

        contact_count = int(on_x_plane) + int(on_y_plane) + int(on_z_plane)

        if contact_count == 3:
            locked.add(node_index)
            continue

        if contact_count == 2:
            if on_x_plane and on_y_plane:
                move_Z_only.add(node_index)
            elif on_x_plane and on_z_plane:
                move_Y_only.add(node_index)
            elif on_y_plane and on_z_plane:
                move_X_only.add(node_index)
            continue

        if contact_count == 1:
            if on_x_plane:
                move_Y_only.add(node_index)
                move_Z_only.add(node_index)
            elif on_y_plane:
                move_X_only.add(node_index)
                move_Z_only.add(node_index)
            elif on_z_plane:
                move_X_only.add(node_index)
                move_Y_only.add(node_index)
            continue

        move_all.add(node_index)

    movable_nodes = (
        move_all
        | move_X_only
        | move_Y_only
        | move_Z_only
    )

    if not movable_nodes:
        raise RuntimeError(
            "No movable nodes were found for 3-D Winslow smoothing."
        )

    return {
        "locked": locked,
        "move_all": move_all,
        "move_X_only": move_X_only,
        "move_Y_only": move_Y_only,
        "move_Z_only": move_Z_only,
        "water_nodes": water_nodes,
        "movable_nodes": movable_nodes,
    }


def build_gmsh_geometry_and_groups3D(
    gmsh,
    fname,
    length_x,
    length_y,
    depth_z,
    padding_type,
    padding_x,
    padding_y,
    padding_z,
    hyper_n,
    water_interface,
    water_search_value,
    structured_mesh,
    nz,
    nx,
    ny,
    dz,
    dx,
    dy,
    byte_order="big",
    axes_order=(0, 1, 2),
    axes_order_sort="F",
    dtype="float32",
    *,
    comm=None,
    parallel_print,
):
    """Build the three-dimensional Gmsh geometry and physical groups for all supported padding and water cases.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    fname : str or pathlib.Path
        Path to the binary velocity model.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth; the model occupies negative z.
    padding_type : str or None
        Padding geometry, either ``None`` or ``"rectangular"`` for Winslow.
    padding_x : float
        Padding thickness in the x direction.
    padding_y : float
        Padding thickness in the y direction.
    padding_z : float
        Bottom padding thickness in the z direction.
    hyper_n : float
        Hyperellipsoid exponent used by the padding geometry and sizing law.
    water_interface : bool
        Whether the water/subsurface interface is geometrically delimited.
    water_search_value : float
        Velocity value used to identify the water layer.
    structured_mesh : bool
        Whether a structured hexahedral mesh is requested.
    minElementSize : float or None
        Minimum element-size input retained by the geometry API.
    nz : int
        Number of velocity samples in the z direction.
    nx : int
        Number of velocity samples in the x direction.
    ny : int
        Number of velocity samples in the y direction.
    dz : float
        Velocity-model spacing in the z direction.
    dx : float
        Velocity-model spacing in the x direction.
    dy : float
        Velocity-model spacing in the y direction.
    byte_order : {"big", "little"}
        Byte order of the velocity-model binary file.
    axes_order : tuple of int
        Permutation mapping binary axes to Spyro ``(z, x, y)`` order.
    axes_order_sort : {"C", "F"}
        Memory order used to reshape the binary velocity data.
    dtype : str or numpy.dtype
        Numeric data type stored in the velocity-model file.
    comm : mpi4py.MPI.Comm or None
        MPI communicator forwarded to rank-aware output.
    parallel_print : callable
        Rank-aware print function accepting a ``comm`` keyword argument.

    Returns
    -------
    dict
        Geometry metadata and volume tags required by mesh generation and smoothing.

    Notes
    -----
    Supported padding types are ``None``, ``"rectangular"``, and ``"hyperelliptical"``.
    Hyperelliptical padding is restricted to unstructured meshing.
    """

    if padding_type not in (None, "rectangular", "hyperelliptical"):
        raise ValueError(
            "padding_type must be None, 'rectangular', "
            "or 'hyperelliptical'."
        )

    # Hyperelliptical padding
    if padding_type == "hyperelliptical" and structured_mesh:
        raise ValueError(
            "Hyperelliptical 3-D padding currently supports only "
            "structured_mesh=False. Winslow smoothing is not available "
            "for this geometry."
        )

    if water_interface and tuple(axes_order) != (0, 1, 2):
        raise ValueError(
            "The supplied water-interface geometry reads the binary directly "
            "in (z, x, y) order; axes_order must therefore be (0, 1, 2)."
        )

    box_xmin = 0.0
    box_xmax = length_x
    box_ymin = 0.0
    box_ymax = length_y
    box_zmin = -abs(depth_z)
    box_zmax = 0.0
    ellipse_a = length_x / 2.0 + padding_x
    ellipse_b = length_y / 2.0 + padding_y
    ellipse_c = abs(depth_z) / 2.0 + padding_z
    xc = length_x / 2.0
    yc = length_y / 2.0
    zc = -abs(depth_z) / 2.0

    z_min, z_max = depth_z, 0.0
    x_min, x_max = 0.0, length_x
    y_min, y_max = 0.0, length_y

    if not water_interface and padding_type is None:
        # No padding
        parallel_print(
            "Generating undelimited water+subsurface rectangular domain (no padding)...",
            comm=comm,
        )
        occ = gmsh.model.occ
        vol_subsoil = occ.addBox(
            x_min,
            y_min,
            z_min,
            x_max - x_min,
            y_max - y_min,
            z_max - z_min,
        )
        occ.synchronize()

        gmsh.model.addPhysicalGroup(3, [vol_subsoil], name="SubsurfaceAndWater")
        vol_water = None

    elif not water_interface and padding_type == "rectangular":
        parallel_print(
            "Generating undelimited water+subsurface domain "
            "with rectangular padding...",
            comm=comm,
        )
        vol_subsoil, vol_water = generate_rectangular_padding_no_water(
            gmsh,
            length_x,
            length_y,
            depth_z,
            padding_x,
            padding_y,
            padding_z,
        )

    elif not water_interface and padding_type == "hyperelliptical":
        # Hyperelliptical padding
        parallel_print(
            "Generating undelimited water+subsurface domain with "
            "hyperelliptical padding...",
            comm=comm,
        )
        occ = gmsh.model.occ

        cube_volume_tag = occ.addBox(
            box_xmin,
            box_ymin,
            box_zmin,
            box_xmax - box_xmin,
            box_ymax - box_ymin,
            box_zmax - box_zmin,
        )
        ellipsoid_tag = create_hyperellipsoid_volume(
            gmsh=gmsh,
            a=ellipse_a,
            b=ellipse_b,
            c=ellipse_c,
            n=hyper_n,
            xc=xc,
            yc=yc,
            zc=zc,
            comm=comm,
            parallel_print=parallel_print,
        )
        if ellipsoid_tag is None:
            raise RuntimeError("Could not create the hyperelliptical outer volume.")

        occ.synchronize()
        _fragment_result, fragment_map = occ.fragment(
            [(3, ellipsoid_tag)],
            [(3, cube_volume_tag)],
            removeObject=True,
            removeTool=True,
        )
        occ.synchronize()

        ellipsoid_fragments = {tag for dim, tag in fragment_map[0] if dim == 3}
        cube_fragments = {tag for dim, tag in fragment_map[1] if dim == 3}

        if not cube_fragments:
            raise RuntimeError(
                "The internal water+subsurface rectangle was lost during "
                "the hyperellipsoid fragment operation."
            )

        padding_fragments = sorted(ellipsoid_fragments - cube_fragments)
        core_fragments = sorted(cube_fragments)

        if not padding_fragments:
            raise RuntimeError(
                "The hyperellipsoid fragment produced no padding volume."
            )

        vol_subsoil = core_fragments[0]
        vol_water = None

        # both water and subsurface.
        gmsh.model.addPhysicalGroup(3, core_fragments, name="SubsurfaceAndWater")
        gmsh.model.addPhysicalGroup(3, padding_fragments, name="Padding")

        ellipsoid_faces = set()
        for volume_tag in padding_fragments:
            ellipsoid_faces.update(boundary_faces_of_volume(gmsh, volume_tag))

        for surface_tag in ellipsoid_faces:
            gmsh.model.mesh.setAlgorithm(2, int(surface_tag), 1)

    else:
        # Cases 4-6: water delimitation enabled.
        vol_subsoil, vol_water = generate_water_interface_volumes(
            gmsh=gmsh,
            fname=fname,
            water_search_value=water_search_value,
            nz=nz,
            nx=nx,
            ny=ny,
            dz=dz,
            dx=dx,
            dy=dy,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
            byte_order=byte_order,
            order=axes_order_sort,
            dtype=dtype,
            comm=comm,
            parallel_print=parallel_print,
        )

        if padding_type is None:

            gmsh.model.addPhysicalGroup(3, [vol_subsoil], name="Subsurface")
            gmsh.model.addPhysicalGroup(3, [vol_water], name="Water")
        if padding_type == "hyperelliptical":
            cube_volume_tag = gmsh.model.occ.addBox(
                box_xmin,
                box_ymin,
                box_zmin,  # x, y, z of corner
                box_xmax - box_xmin,  # width in x
                box_ymax - box_ymin,  # width in y
                box_zmax - box_zmin,  # width in z
            )

            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()
            ellipsoid_tag = create_hyperellipsoid_volume(
                gmsh,
                a=ellipse_a,
                b=ellipse_b,
                c=ellipse_c,
                n=hyper_n,
                xc=xc,
                yc=yc,
                zc=zc,
                comm=comm,
                parallel_print=parallel_print,
            )
            gmsh.model.occ.synchronize()
            gmsh.model.occ.fragment(
                [
                    (3, ellipsoid_tag),
                    (3, cube_volume_tag),
                    (3, vol_subsoil),
                    (3, vol_water),
                ],
                [],  # Tool volumes (empty for self-fragmentation)
                removeObject=True,
                removeTool=False,
            )
            gmsh.model.occ.synchronize()
            volumes = gmsh.model.getEntities(3)
            volume_tags = [tag for dim, tag in volumes if dim == 3]

            water_volumes = [volume_tags[1]] if len(volume_tags) > 0 else []
            ellipsoid_volumes = [volume_tags[2]] if len(volume_tags) > 1 else []
            cube_volumes = [volume_tags[0]] if len(volume_tags) > 2 else []

            gmsh.model.addPhysicalGroup(3, cube_volumes, name="Subsurface")
            gmsh.model.addPhysicalGroup(3, water_volumes, name="Water")
            gmsh.model.addPhysicalGroup(3, ellipsoid_volumes, name="Padding")

            ellipsoid_faces = []
            for v in ellipsoid_volumes:
                ellipsoid_faces.extend(boundary_faces_of_volume(gmsh, v))
            ellipsoid_faces = set(ellipsoid_faces)
            algo_id = 1.0
            for surf_tag in ellipsoid_faces:
                gmsh.model.mesh.setAlgorithm(2, int(surf_tag), int(algo_id))
        if padding_type == "rectangular":
            generate_structured_rectangular_padding_water(
                gmsh=gmsh,
                vol_water=vol_water,
                vol_subsoil=vol_subsoil,
                padding_x=padding_x,
                padding_y=padding_y,
                padding_z=padding_z,
                comm=comm,
                parallel_print=parallel_print,
            )

    gmsh.model.occ.synchronize()
    return {
        "vol_subsoil": vol_subsoil,
        "vol_water": vol_water,
        "padding_type3D": padding_type,
        "water_interface": bool(water_interface),
        "water_delimited": bool(water_interface),
        "contains_water_from_velocity_model": True,
        "structured_mesh_supported": padding_type != "hyperelliptical",
        "winslow_supported": padding_type != "hyperelliptical",
        "ellipse_a": ellipse_a,
        "ellipse_b": ellipse_b,
        "ellipse_c": ellipse_c,
        "xc": xc,
        "yc": yc,
        "zc": zc,
    }
