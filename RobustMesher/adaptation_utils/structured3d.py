import numpy as np
from ..geometry_utils.geometry3d import align_water_columns3d, define_winslow_points_3d
from .winslow3d import run_selected_winslow


def apply_structured_winslow_smoothing3D(
    gmsh,
    comm,
    geom_params,
    length_x,
    length_y,
    depth_z,
    padding_type,
    padding_x,
    padding_y,
    padding_z,
    water_interface,
    hyper_n,
    winslow_implementation,
    apply_winslow,
    winslow_iterations,
    winslow_omega,
    n_samples,
    n_traces_x,
    n_traces_y,
    domain_xmin,
    domain_xmax,
    domain_ymin,
    domain_ymax,
    domain_zmin,
    domain_zmax,
    ef_segy3,
    parallel_print,
):
    """Apply the configured Winslow smoother to a structured three-dimensional Gmsh mesh.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    comm : mpi4py.MPI.Comm or None
        MPI communicator forwarded to rank-aware output.
    geom_params : dict
        Geometry metadata returned by the Gmsh geometry builder.
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
    hyper_n : float
        Hyperellipsoid exponent used by the padding geometry and sizing law.
    winslow_implementation : str or None
        Requested Winslow implementation.
    apply_winslow : bool
        Whether Winslow smoothing should be applied.
    winslow_iterations : int
        Number of Winslow smoothing iterations.
    winslow_omega : float
        Winslow relaxation coefficient.
    n_samples : int
        Velocity-model sample count retained for API compatibility.
    n_traces_x : int
        Velocity-model x-trace count retained for API compatibility.
    n_traces_y : int
        Velocity-model y-trace count retained for API compatibility.
    domain_xmin : float
        Minimum model x coordinate retained for API compatibility.
    domain_xmax : float
        Maximum model x coordinate retained for API compatibility.
    domain_ymin : float
        Minimum model y coordinate retained for API compatibility.
    domain_ymax : float
        Maximum model y coordinate retained for API compatibility.
    domain_zmin : float
        Minimum model z coordinate retained for API compatibility.
    domain_zmax : float
        Maximum model z coordinate retained for API compatibility.
    ef_segy3 : callable
        Three-dimensional mesh-sizing function in ``(z, x, y)`` coordinates.
    parallel_print : callable
        Rank-aware print function accepting a ``comm`` keyword argument.

    Returns
    -------
    None
        Water columns are aligned in place when delimited. Non-water nodes
        are smoothed only when smoothing is enabled.
    """
    aligned_water_tags = set()
    if water_interface and padding_type in (None, "rectangular"):
        aligned_water_tags = align_water_columns3d(gmsh, padding_type)
        parallel_print(
            f"Aligned 3D water columns ({len(aligned_water_tags)} water nodes).",
            comm=comm,
        )

    if not apply_winslow:
        parallel_print("Skipping 3D Winslow smoothing.", comm=comm)
        return

    if padding_type == "hyperelliptical":
        raise ValueError(
            "3-D Winslow smoothing is not available for hyperelliptical padding."
        )
    if padding_type not in (None, "rectangular"):
        raise ValueError(
            "3-D Winslow smoothing supports only padding_type=None "
            "or padding_type='rectangular'."
        )

    selected_winslow = (
        "numba"
        if winslow_implementation is None
        else str(winslow_implementation).strip().lower()
    )
    if selected_winslow != "numba":
        raise ValueError(
            "Only winslow_implementation='numba' is currently available "
            "for 3-D structured smoothing. Received "
            f"{winslow_implementation!r}."
        )

    geometry_name = (
        ("Water Interface" if water_interface else "No Water Interface")
        + ", "
        + ("Rectangular Padding" if padding_type == "rectangular" else "No Padding")
    )
    parallel_print(f"Extracting nodes for 3D smoothing ({geometry_name})...", comm=comm)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    points_3d = np.asarray(node_coords, dtype=float).reshape(-1, 3)

    elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    all_hex_nodes = []
    for element_index, element_type in enumerate(elem_types):
        if element_type == 5:  # Gmsh type 5 = 8-node hexahedron
            all_hex_nodes.extend(elem_node_tags[element_index])

    if not all_hex_nodes:
        raise ValueError(
            "No hexahedra found. 3-D Winslow smoothing requires "
            "an 8-node hexahedral mesh."
        )

    tag_to_index = {
        int(tag): index
        for index, tag in enumerate(node_tags)
    }
    hexes = np.asarray(
        [tag_to_index[int(tag)] for tag in all_hex_nodes],
        dtype=np.int64,
    ).reshape(-1, 8)

    winslow_points = define_winslow_points_3d(
        gmsh=gmsh,
        points_3d=points_3d,
        tag_to_index=tag_to_index,
        length_x=length_x,
        length_y=length_y,
        depth_z=depth_z,
        padding_type=padding_type,
        padding_x=padding_x,
        padding_y=padding_y,
        padding_z=padding_z,
        water_interface=water_interface,
        tol=2.0,
    )

    if aligned_water_tags:
        water_nodes = {tag_to_index[tag] for tag in aligned_water_tags}
        for key in (
            "move_all", "move_X_only", "move_Y_only", "move_Z_only", "movable_nodes",
        ):
            winslow_points[key].difference_update(water_nodes)
        winslow_points["locked"].update(water_nodes)
        winslow_points["water_nodes"].update(water_nodes)

    move_all = winslow_points["move_all"]
    move_X_only = winslow_points["move_X_only"]
    move_Y_only = winslow_points["move_Y_only"]
    move_Z_only = winslow_points["move_Z_only"]
    movable_nodes = winslow_points["movable_nodes"]

    parallel_print("Applying 3D Winslow smoothing...", comm=comm)
    smoothed_points_3d = run_selected_winslow(
        points=points_3d,
        hexes=hexes,
        move_all=move_all,
        move_X_only=move_X_only,
        move_Y_only=move_Y_only,
        move_Z_only=move_Z_only,
        ef_segy=ef_segy3,
        length_x=length_x,
        length_y=length_y,
        depth_z=depth_z,
        winslow_iterations=winslow_iterations,
        winslow_omega=winslow_omega,
        selected_winslow=selected_winslow,
        comm=comm,
        parallel_print=parallel_print,
    )

    parallel_print("Updating Winslow-smoothed nodes back into Gmsh...", comm=comm)
    for node_index, node_tag in enumerate(node_tags):
        if node_index in movable_nodes:
            gmsh.model.mesh.setNode(
                int(node_tag),
                smoothed_points_3d[node_index].tolist(),
                [],
            )

