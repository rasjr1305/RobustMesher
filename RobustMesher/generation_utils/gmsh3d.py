import numpy as np
from ..geometry_utils.gmsh3d_helpers import checked_mesh_size


def configure_gmsh_mesh_size3D(
    gmsh,
    ef_segy3,
    bbox,
    structured_mesh,
    parallel,
    extend_segy,
    padding_type,
    padding_x,
    padding_y,
    padding_z,
    hyper_n,
    h_padding,
    length_x,
    length_y,
    depth_z,
    nz,
    nx,
    ny,
    *,
    gmsh_num_threads=1,
    comm=None,
    parallel_print,
):
    """Configure serial or parallel Gmsh mesh sizing from the velocity-model sizing function.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    ef_segy3 : callable
        Three-dimensional mesh-sizing function in ``(z, x, y)`` coordinates.
    bbox : sequence of float
        Bounding box ordered as ``(zmin, zmax, xmin, xmax, ymin, ymax)``.
    structured_mesh : bool
        Whether a structured hexahedral mesh is requested.
    parallel : bool
        Whether to build a point-cloud background field for parallel meshing.
    extend_segy : bool
        Whether velocity-model sizing is extended by nearest-edge projection.
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
    h_padding : float
        Target element size at the outer padding boundary.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth; the model occupies negative z.
    nz : int
        Number of velocity samples in the z direction.
    nx : int
        Number of velocity samples in the x direction.
    ny : int
        Number of velocity samples in the y direction.
    gmsh_num_threads : int
        Number of threads used by parallel 3-D Gmsh/HXT meshing. Default is 1.
    comm : mpi4py.MPI.Comm or None
        MPI communicator forwarded to rank-aware output.
    parallel_print : callable
        Rank-aware print function accepting a ``comm`` keyword argument.

    Returns
    -------
    None
        The Gmsh sizing callback or background field is configured in place.
    """
    ef_segy2 = ef_segy3

    if structured_mesh is False:
        if parallel is False:
            callback_calls = 0

            def mesh_size_callback3D(dim, tag, x, y, z, lc):
                """Evaluate the serial Gmsh mesh-size callback at one mesh point.

                Parameters
                ----------
                dim : int
                    Dimension of the Gmsh entity requesting the size.
                tag : int
                    Tag of the Gmsh entity requesting the size.
                x : float
                    X coordinate of the Gmsh sizing query.
                y : float
                    Y coordinate of the Gmsh sizing query.
                z : float
                    Z coordinate of the Gmsh sizing query.
                lc : float
                    Element size proposed by Gmsh before applying the callback.

                Returns
                -------
                float
                    Element size assigned to the queried mesh point.
                """
                nonlocal callback_calls
                callback_calls += 1

                if extend_segy:
                    # Edge-extended sizing

                    coords = np.array([[z, x, y]], dtype=float)
                    return checked_mesh_size(
                        ef_segy2,
                        coords,
                        "extend_segy=True",
                    )
                else:
                    (
                        z_min_segy,
                        z_max_segy,
                        x_min_segy,
                        x_max_segy,
                        y_min_segy,
                        y_max_segy,
                    ) = bbox

                    in_x = (x >= x_min_segy) and (x <= x_max_segy)
                    in_y = (y >= y_min_segy) and (y <= y_max_segy)
                    in_z = (z >= z_min_segy) and (z <= z_max_segy)

                    if in_x and in_y and in_z:
                        coords = np.array([[z, x, y]], dtype=float)
                        return checked_mesh_size(
                            ef_segy2,
                            coords,
                            "inside velocity model",
                        )

                    else:

                        x_proj = min(max(x, x_min_segy), x_max_segy)
                        y_proj = min(max(y, y_min_segy), y_max_segy)
                        z_proj = min(max(z, z_min_segy), z_max_segy)

                        coords_proj = np.array(
                            [[z_proj, x_proj, y_proj]],
                            dtype=float,
                        )
                        base_size = checked_mesh_size(
                            ef_segy2,
                            coords_proj,
                            "projected padding boundary",
                        )

                        dx = abs(x - x_proj)
                        dy = abs(y - y_proj)
                        dz = abs(z - z_proj)

                        tx = dx / padding_x if padding_x > 0 else 0.0
                        ty = (
                            dy / padding_y if padding_y > 0 else 0.0
                        )
                        tz = dz / padding_z if padding_z > 0 else 0.0

                        if padding_type == "hyperelliptical":
                            t = (tx**hyper_n + ty**hyper_n + tz**hyper_n) ** (
                                1.0 / hyper_n
                            )
                        else:
                            t = max(tx, ty, tz)

                        t = min(t, 1.0)  # Clamp between 0.0 and 1.0
                        max_padding_size = h_padding
                        graded_size = base_size + t * (max_padding_size - base_size)

                        return float(graded_size)

            gmsh.model.mesh.setSizeCallback(mesh_size_callback3D)

        if parallel:
            num_threads = gmsh_num_threads
            if num_threads < 1:
                raise ValueError("gmsh_num_threads must be a positive integer.")

            parallel_print(
                f"Computing mesh size callback for parallel meshing "
                f"with {num_threads} Gmsh thread(s)...",
                comm=comm,
            )

            x_core = np.linspace(bbox[2], bbox[3], nx)
            y_core = np.linspace(bbox[4], bbox[5], ny)
            z_core = np.linspace(bbox[0], bbox[1], nz)

            XX_c, YY_c, ZZ_c = np.meshgrid(x_core, y_core, z_core, indexing="ij")
            coords_c = np.column_stack((ZZ_c.ravel(), XX_c.ravel(), YY_c.ravel()))

            size_core = ef_segy2(coords_c).ravel()

            sp_data_core = np.empty((XX_c.size, 4), dtype=np.float64)
            sp_data_core[:, 0] = XX_c.ravel()
            sp_data_core[:, 1] = YY_c.ravel()
            sp_data_core[:, 2] = ZZ_c.ravel()
            sp_data_core[:, 3] = size_core

            xmin_pad = -padding_x
            xmax_pad = length_x + padding_x
            ymin_pad = -padding_y
            ymax_pad = length_y + padding_y
            zmin_pad = -abs(depth_z) - padding_z
            zmax_pad = 0.0

            dx_core = (bbox[3] - bbox[2]) / max(nx - 1, 1)
            dy_core = (bbox[5] - bbox[4]) / max(ny - 1, 1)
            dz_core = (bbox[1] - bbox[0]) / max(nz - 1, 1)

            nx_pad = max(2, int(np.ceil((xmax_pad - xmin_pad) / dx_core)) + 1,)
            ny_pad = max(2, int(np.ceil((ymax_pad - ymin_pad) / dy_core)) + 1,)
            nz_pad = max(2, int(np.ceil((zmax_pad - zmin_pad) / dz_core)) + 1,)

            x_pad = np.linspace(xmin_pad, xmax_pad, nx_pad)
            y_pad = np.linspace(ymin_pad, ymax_pad, ny_pad)
            z_pad = np.linspace(zmin_pad, zmax_pad, nz_pad)

            XX_p, YY_p, ZZ_p = np.meshgrid(x_pad, y_pad, z_pad, indexing="ij")

            eps = 1e-8
            out_mask = ~(
                (XX_p >= bbox[2] - eps)
                & (XX_p <= bbox[3] + eps)
                & (YY_p >= bbox[4] - eps)
                & (YY_p <= bbox[5] + eps)
                & (ZZ_p >= bbox[0] - eps)
                & (ZZ_p <= bbox[1] + eps)
            )

            XX_out = XX_p[out_mask]
            YY_out = YY_p[out_mask]
            ZZ_out = ZZ_p[out_mask]

            # Padding sizing
            if extend_segy:

                coords_p = np.column_stack((ZZ_out, XX_out, YY_out))
                size_pad = ef_segy2(coords_p).ravel()
            else:

                XX_proj = np.clip(XX_out, bbox[2], bbox[3])
                YY_proj = np.clip(YY_out, bbox[4], bbox[5])
                ZZ_proj = np.clip(ZZ_out, bbox[0], bbox[1])

                coords_p = np.column_stack((ZZ_proj, XX_proj, YY_proj))
                base_size_pad = ef_segy2(coords_p).ravel()

                tx = (
                    np.abs(XX_out - XX_proj) / padding_x
                    if padding_x > 0
                    else np.zeros_like(XX_out)
                )
                ty = (
                    np.abs(YY_out - YY_proj) / padding_y
                    if padding_y > 0
                    else np.zeros_like(YY_out)
                )
                tz = (
                    np.abs(ZZ_out - ZZ_proj) / padding_z
                    if padding_z > 0
                    else np.zeros_like(ZZ_out)
                )

                if padding_type == "hyperelliptical":
                    t = (tx**hyper_n + ty**hyper_n + tz**hyper_n) ** (1.0 / hyper_n)
                else:
                    t = np.maximum(np.maximum(tx, ty), tz)

                t = np.clip(t, 0.0, 1.0)
                size_pad = base_size_pad + t * (h_padding - base_size_pad)

            sp_data_pad = np.empty((XX_out.size, 4), dtype=np.float64)
            sp_data_pad[:, 0] = XX_out
            sp_data_pad[:, 1] = YY_out
            sp_data_pad[:, 2] = ZZ_out
            sp_data_pad[:, 3] = size_pad

            sp_data_combined = np.vstack((sp_data_core, sp_data_pad))
            total_points = len(sp_data_combined)

            sp_list = sp_data_combined.ravel().tolist()

            parallel_print(f"Loading {total_points} total points into Gmsh PostView...", comm=comm)
            view_tag = gmsh.view.add("background_size")

            gmsh.view.addListData(view_tag, "SP", total_points, sp_list)

            # Apply background field
            gmsh.model.mesh.field.add("PostView", 1)
            gmsh.model.mesh.field.setNumber(1, "ViewIndex", view_tag)
            gmsh.model.mesh.field.setAsBackgroundMesh(1)
            # Parallel HXT in 3D
            gmsh.option.setNumber("General.NumThreads", np.int64(num_threads))
            gmsh.option.setNumber("Mesh.MaxNumThreads3D", np.int64(num_threads))
            # Serialize the boundary mesh generation for deterministic results
            gmsh.option.setNumber("Mesh.MaxNumThreads1D", 1)
            gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
