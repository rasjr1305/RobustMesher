import numpy as np
from .winslow2d import winslow_smooth_numba, winslow_smooth_vectorized, winslow_smooth_default
from ..geometry_utils.geometry2d import (
    get_surface_entities_by_physical_name, get_nodes_on_surface_entities,
    get_water_interface_node_indices, align_water_columns_to_interface_x,
    intersect, make_arc,
)


def apply_structured_winslow_smoothing2d(
    gmsh, comm, geom_params, length_x, depth_z, padding_type,
    water_interface, hyper_n, winslow_implementation, winslow_iterations,
    winslow_omega, n_samples, n_traces, domain_xmin, domain_xmax,
    domain_zmin, domain_zmax, ef_segy2, parallel_print,
    z_water_L, z_water_R, pad_x_min, pad_x_max, pad_z_min, a_val, b_val, xc, zc, apply_winslow
):
    """
    Extracts nodes and elements from a structured Gmsh model, calculates physical
    boundaries, applies Winslow smoothing, and updates the Gmsh node coordinates.
    """
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    points_3d = np.asarray(node_coords, dtype=float).reshape(-1, 3)
    points_2d = points_3d[:, :2]

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    all_quad_nodes = []
    for i, t_type in enumerate(elem_types):
        if t_type == 3:
            all_quad_nodes.extend(elem_node_tags[i])

    if not all_quad_nodes:
        raise ValueError("No quadrilaterals found! Winslow requires quads.")

    tag_to_index = {tag: idx for idx, tag in enumerate(node_tags)}
    quads = np.asarray([tag_to_index[tag] for tag in all_quad_nodes], dtype=np.int64).reshape(-1, 4)

    if water_interface:
        water_surface_entities = get_surface_entities_by_physical_name("WaterSurface")
        water_surface_nodes = get_nodes_on_surface_entities(tag_to_index, water_surface_entities)
        interface_nodes = get_water_interface_node_indices(
            tag_to_index=tag_to_index, water_surface_entities=water_surface_entities, length_x=length_x, tol=1e-8
        )
        parallel_print("Aligning water columns with water spline X positions...", comm=comm)
        points_2d, n_snapped, n_cols = align_water_columns_to_interface_x(
            points_2d=points_2d, water_surface_nodes=water_surface_nodes, interface_nodes=interface_nodes, quads=quads
        )
        parallel_print(f"Snapped {n_snapped} water-surface nodes onto {n_cols} spline-X columns.", comm=comm)
    else:
        interface_nodes = set()
        water_surface_nodes = set()

    move_all = set()
    move_X_only = set()
    move_Z_only = set()
    move_hyperellipse = set()
    locked = set()

    def get_nodes(dim, tags):
        nodes = set()
        for t in tags:
            if t is None:
                continue
            n, _, _ = gmsh.model.mesh.getNodes(dim, t, includeBoundary=True)
            if len(n) > 0:
                nodes.update([tag_to_index[node] for node in n])
        return nodes

    if padding_type == "hyperelliptical":
        if water_interface:
            locked_surfs = [geom_params.get("water_surface"), geom_params.get("surf_pad_TL"), geom_params.get("surf_pad_TR")]
            locked_surface_nodes = get_nodes(2, locked_surfs)
            outer_middle_arcs = [geom_params.get(k) for k in ["arc_WL_ML", "arc_ML_BL45", "arc_BL45_BML", "arc_BML_BMR", "arc_BMR_BR45", "arc_BR45_MR", "arc_MR_WR"]]
            z_slide_curves = [geom_params.get(k) for k in ["rock_L_lower", "rock_L_upper", "rock_R_lower", "rock_R_upper"]]
            x_slide_curves = [geom_params.get(k) for k in ["rock_B_left", "rock_B_mid", "rock_B_right"]]
            diag_rays = [geom_params.get("ray_BL_45"), geom_params.get("ray_BR_45")]
        else:
            locked_surface_nodes = set()

            outer_middle_arcs = [
                geom_params.get(k) for k in ["arc_TL_BL", "arc_BL_BML", "arc_BML_BMR", "arc_BMR_BR", "arc_BR_TR"]
            ]

            z_slide_curves = [geom_params.get("line_left"), geom_params.get("line_right")]
            x_slide_curves = [geom_params.get("line_bot_left"), geom_params.get("line_bot_mid"), geom_params.get("line_bot_right")]
            diag_rays = [geom_params.get("ray_BL"), geom_params.get("ray_BR")]

        outer_arc_nodes = get_nodes(1, outer_middle_arcs)
        z_slide_nodes = get_nodes(1, z_slide_curves)
        x_slide_nodes = get_nodes(1, x_slide_curves)
        locked_diag_nodes = get_nodes(1, diag_rays)

        corner_indices = set()
        tol = 1e-3
        x_O_TL = geom_params.get("x_O_TL")
        x_O_TR = geom_params.get("x_O_TR")

        for i, pt in enumerate(points_2d):
            x, z = pt
            if water_interface:
                if (abs(x - 0.0) < tol and abs(z - depth_z) < tol) or \
                   (abs(x - length_x) < tol and abs(z - depth_z) < tol) or \
                   (abs(x - 0.0) < tol and abs(z - z_water_L) < tol) or \
                   (abs(x - length_x) < tol and abs(z - z_water_R) < tol):
                    corner_indices.add(i)
            else:
                if (abs(x - 0.0) < tol and abs(z - depth_z) < tol) or \
                   (abs(x - length_x) < tol and abs(z - depth_z) < tol) or \
                   (abs(x - 0.0) < tol and abs(z - 0.0) < tol) or \
                   (abs(x - length_x) < tol and abs(z - 0.0) < tol) or \
                   (x_O_TL is not None and abs(x - x_O_TL) < tol and abs(z - 0.0) < tol) or \
                   (x_O_TR is not None and abs(x - x_O_TR) < tol and abs(z - 0.0) < tol):
                    corner_indices.add(i)

        for i in range(len(points_2d)):
            if i in corner_indices:
                locked.add(i)
            elif i in locked_diag_nodes:
                locked.add(i)
            elif i in locked_surface_nodes:
                if i in x_slide_nodes:
                    move_X_only.add(i)
                elif i in z_slide_nodes:
                    move_Z_only.add(i)
                else:
                    locked.add(i)
            elif not water_interface and abs(points_2d[i][1] - 0.0) < tol:
                move_X_only.add(i)  # Lock top boundary Z axis, slide in X
            elif i in outer_arc_nodes:
                move_hyperellipse.add(i)
            elif i in x_slide_nodes:
                move_X_only.add(i)
            elif i in z_slide_nodes:
                move_Z_only.add(i)
            else:
                move_all.add(i)

    elif padding_type == "rectangular":
        tol = 1e-3
        if water_interface:
            corners_to_lock = [
                (0.0, depth_z), (length_x, depth_z), (pad_x_min, depth_z), (pad_x_max, depth_z),
                (0.0, pad_z_min), (length_x, pad_z_min), (pad_x_min, pad_z_min), (pad_x_max, pad_z_min)
            ]
        else:
            corners_to_lock = [
                (0.0, 0.0), (length_x, 0.0), (pad_x_min, 0.0), (pad_x_max, 0.0),
                (0.0, depth_z), (length_x, depth_z), (pad_x_min, depth_z), (pad_x_max, depth_z),
                (0.0, pad_z_min), (length_x, pad_z_min), (pad_x_min, pad_z_min), (pad_x_max, pad_z_min)
            ]

        for i, pt in enumerate(points_2d):
            x, z = pt
            is_locked = False
            for cx, cz in corners_to_lock:
                if abs(x - cx) < tol and abs(z - cz) < tol:
                    locked.add(i)
                    is_locked = True
                    break
            if is_locked:
                continue

            if water_interface:
                if (x <= 0.0 + tol and z >= z_water_L - tol) or (x >= length_x - tol and z >= z_water_R - tol):
                    locked.add(i)
                    continue
                if i in water_surface_nodes:
                    locked.add(i)
                    continue

            if abs(x - pad_x_min) < tol or abs(x - pad_x_max) < tol or abs(x - 0.0) < tol or abs(x - length_x) < tol:
                move_Z_only.add(i)
                continue

            if abs(z - pad_z_min) < tol or abs(z - depth_z) < tol or (not water_interface and abs(z - 0.0) < tol):
                move_X_only.add(i)
                continue

            move_all.add(i)

    elif padding_type is None:
        tol = 1e-3
        corners_to_lock = [(0.0, 0.0), (length_x, 0.0), (0.0, depth_z), (length_x, depth_z)]

        for i, pt in enumerate(points_2d):
            x, z = pt
            is_locked = False
            for cx, cz in corners_to_lock:
                if abs(x - cx) < tol and abs(z - cz) < tol:
                    locked.add(i)
                    is_locked = True
                    break
            if is_locked:
                continue

            if water_interface:
                if (x <= 0.0 + tol and z >= z_water_L - tol) or (x >= length_x - tol and z >= z_water_R - tol):
                    locked.add(i)
                    continue
                if i in water_surface_nodes:
                    locked.add(i)
                    continue

            if abs(x - 0.0) < tol or abs(x - length_x) < tol:
                move_Z_only.add(i)
                continue
            if abs(z - 0.0) < tol or abs(z - depth_z) < tol:
                move_X_only.add(i)
                continue

            move_all.add(i)

    parallel_print(f"Nodes Breakdown | Total: {len(points_2d)}", comm=comm)
    parallel_print(f"Move All: {len(move_all)} | X-Slide: {len(move_X_only)} | Z-Slide: {len(move_Z_only)} | hyperellipse: {len(move_hyperellipse)} | Locked: {len(locked)}", comm=comm)

    if apply_winslow:
        parallel_print("Applying Winslow smoothing...", comm=comm)
        if winslow_implementation in ("fast", "numba"):
            nx_grid, nz_grid = n_samples, n_traces
            segy_grid_x = np.linspace(domain_xmin, domain_xmax, nx_grid)
            segy_grid_z = np.linspace(domain_zmin, domain_zmax, nz_grid)
            X_grid, Z_grid = np.meshgrid(segy_grid_x, segy_grid_z, indexing='ij')
            sizes_flat = ef_segy2(X_grid.flatten(), Z_grid.flatten())
            segy_grid_vals = sizes_flat.reshape((nx_grid, nz_grid))

            if winslow_implementation == "fast":
                smoothed_points_2d = winslow_smooth_vectorized(
                    points=points_2d, quads=quads, segy_grid_x=segy_grid_x,
                    segy_grid_z=segy_grid_z, segy_grid_vals=segy_grid_vals,
                    move_all=move_all, move_X_only=move_X_only, move_Z_only=move_Z_only,
                    move_hyperellipse=move_hyperellipse, hyperellipse_params=(a_val, b_val, xc, zc, hyper_n),
                    iterations=winslow_iterations, omega=winslow_omega
                )
            elif winslow_implementation == "numba":
                smoothed_points_2d = winslow_smooth_numba(
                    points=points_2d, quads=quads, segy_grid_x=segy_grid_x,
                    segy_grid_z=segy_grid_z, segy_grid_vals=segy_grid_vals,
                    move_all=move_all, move_X_only=move_X_only, move_Z_only=move_Z_only,
                    move_hyperellipse=move_hyperellipse, hyperellipse_params=(a_val, b_val, xc, zc, hyper_n),
                    iterations=winslow_iterations, omega=winslow_omega
                )
        else:
            smoothed_points_2d = winslow_smooth_default(
                points=points_2d, quads=quads, sizing_fn=ef_segy2,
                move_all=move_all, move_X_only=move_X_only, move_Z_only=move_Z_only,
                move_hyperellipse=move_hyperellipse, hyperellipse_params=(a_val, b_val, xc, zc, hyper_n),
                iterations=winslow_iterations, omega=winslow_omega
            )

        smoothed_points_3d = np.zeros_like(points_3d)
        smoothed_points_3d[:, :2] = smoothed_points_2d

        parallel_print("Updating nodes in Gmsh...", comm=comm)
        for i, tag in enumerate(node_tags):
            gmsh.model.mesh.setNode(int(tag), smoothed_points_3d[i].tolist(), [])
    else:
        parallel_print("Skipping Winslow smoothing...", comm=comm)

