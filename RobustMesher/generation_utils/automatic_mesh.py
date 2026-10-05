import numpy as np
import gmsh
from .support import parallel_print
from ..Io_utils.segy_io import create_segy_from_grid
from ..geometry_utils.geometry2d import build_gmsh_geometry_and_groups
from ..geometry_utils.geometry3d import build_gmsh_geometry_and_groups3D
from .gmsh2d import configure_gmsh_2d_meshing_mode
from .gmsh3d import configure_gmsh_mesh_size3D
from ..adaptation_utils.structured2d import apply_structured_winslow_smoothing2d
from ..adaptation_utils.structured3d import apply_structured_winslow_smoothing3D
from .bindings import create_sizing_function, create_sizing_function3D, read_mesh as FireMeshReader


class AutomaticMesh:
    """Gmsh-only extraction of Spyro's AutomaticMesh; no wave solver required."""


    def __init__(
        self, mesh_parameters=None
    ):
        """
        Initialize the MeshingFunctions class.

        Parameters
        ----------
        comm : MPI communicator, optional
            MPI communicator. The default is None.
        mesh_parameters : dict, optional
            Dictionary containing the mesh parameters. The default is None.

        Raises
        ------
        ValueError
            If `abc_pad_length` is negative.

        Notes
        -----
        The `mesh_parameters` dictionary should contain the following keys:
        - 'dimension' : int, optional. Dimension of the mesh. The default is 2.
        - 'length_z' : float, optional. Length of the mesh in the z-direction.
        - 'length_x' : float, optional. Length of the mesh in the x-direction.
        - 'length_y' : float, optional. Length of the mesh in the y-direction.
        - 'cell_type' : str, optional. Type of the mesh cells.
        - 'mesh_type' : str, optional. Type of the mesh.

        For mesh with absorbing layer only:
        - 'abc_pad_length' : float, optional. Length of the absorbing boundary condition padding.

        For Firedrake mesh only:
        - 'dx' : float, optional. Mesh element size.
        - 'periodic' : bool, optional. Whether the mesh is periodic.
        - 'edge_length' : float, optional. Length of the mesh edges.

        For velocity-based sizing:
        - 'cells_per_wavelength' : float, optional. Number of cells per wavelength.
        - 'source_frequency' : float, optional. Frequency of the source.
        - 'minimum_velocity' : float, optional. Minimum velocity.
        - 'velocity_model_file' : str, optional. File containing the velocity model.
        - 'edge_length' : float, optional. Length of the mesh edges.
        """
        self.dimension = mesh_parameters.dimension
        self.length_z = mesh_parameters.length_z
        self.length_x = mesh_parameters.length_x
        self.length_y = mesh_parameters.length_y
        self.quadrilateral = mesh_parameters.quadrilateral
        self.comm = mesh_parameters.comm
        self.mesh_type = mesh_parameters.mesh_type
        self.edge_length = mesh_parameters.edge_length
        self.edge_length_z = getattr(mesh_parameters, "edge_length_z", None)
        self.edge_length_x = getattr(mesh_parameters, "edge_length_x", None)
        self.edge_length_y = getattr(mesh_parameters, "edge_length_y", None)
        self.abc_pad = mesh_parameters.abc_pad_length
        self.mesh_parameters = mesh_parameters

        # Firedrake mesh only parameters

        self.periodic = mesh_parameters.periodic

        # Velocity-based sizing parameters
        self.cpw = mesh_parameters.cells_per_wavelength
        self.source_frequency = mesh_parameters.source_frequency
        self.minimum_velocity = mesh_parameters.minimum_velocity
        self.lbda = None
        self.velocity_model = mesh_parameters.velocity_model
        self.output_file_name = mesh_parameters.output_filename


    def _resolved_edge_length(self, axis):
        axis_value = getattr(self, f"edge_length_{axis}", None)
        if axis_value is not None:
            return axis_value
        return self.edge_length


    def create_gmsh_2D_mesh(self):
        """
        Creates a 2D mesh using Gmsh with optional water interface,
        hyperelliptical/rectangular padding, structured Winslow smoothing,
        or unstructured full-quadrilateral meshing.

        Returns
        -------
        mesh : Firedrake Mesh
            The loaded Firedrake mesh object.
        """
        if gmsh is None:
            raise ImportError("gmsh is not available. Please install it.")

        if self.mesh_parameters.segy_velocity_model is not None:
            self.velocity_model = self.mesh_parameters.segy_velocity_model

        elif self.mesh_parameters.velocity_model is not None:
            velocity_model = self.mesh_parameters.velocity_model

            if not isinstance(velocity_model, dict):
                raise TypeError(
                    "velocity_model must be a grid dictionary when "
                    "segy_velocity_model is not provided."
                )

            if "vp_values" not in velocity_model:
                raise ValueError(
                    "Grid velocity_model must contain the key 'vp_values'."
                )

            vp = np.asarray(velocity_model["vp_values"])

            if vp.ndim != 2:
                raise ValueError(
                    "Grid velocity_model['vp_values'] must be a 2-D array, "
                    f"but received shape {vp.shape}."
                )

            vp_for_gmsh = np.ascontiguousarray(vp[::-1, :])

            filename = "tmp_velocity_model.segy"
            create_segy_from_grid(vp_for_gmsh, filename)

            self.velocity_model = filename

        else:
            raise ValueError(
                "Gmsh meshing requires either 'segy_velocity_model' "
                "or a grid 'velocity_model'."
            )

        if self.comm is None or self.comm.ensemble_comm.rank == 0:
            parallel_print("Generating Gmsh mesh...", comm=self.comm)

            depth_z = -abs(self.length_z)
            length_x = self.length_x
            padding_z = self.mesh_parameters.padding_z
            padding_x = self.mesh_parameters.padding_x
            hyper_n = self.mesh_parameters.hyper_n
            fname = self.velocity_model
            hmin_segy = self.mesh_parameters.hmin_segy
            wl = self.cpw
            freq = self.source_frequency
            grade = self.mesh_parameters.grade
            water_search_value = self.mesh_parameters.water_search_value
            padding_type = self.mesh_parameters.padding_type
            output_file = self.output_file_name
            water_interface = self.mesh_parameters.water_interface
            vp_water = self.mesh_parameters.vp_water
            structured_mesh = self.mesh_parameters.structured_mesh
            unstructured_quad_mesh = self.mesh_parameters.unstructured_quad_mesh
            minElementSize = self.mesh_parameters.min_element_size
            winslow_implementation = self.mesh_parameters.winslow_implementation
            apply_winslow = self.mesh_parameters.apply_winslow
            winslow_iterations = self.mesh_parameters.winslow_iterations
            winslow_omega = self.mesh_parameters.winslow_omega
            extend_segy = self.mesh_parameters.extend_segy
            h_padding = self.mesh_parameters.h_padding
            segy_bbox = (depth_z, 0, 0, length_x)

            # Calculating domain bounding box according to selected padding
            box_xmax = length_x
            box_zmin = depth_z
            xc = box_xmax / 2.0
            zc = box_zmin / 2.0

            if padding_type is None:
                domain_xmin, domain_xmax = 0.0, length_x
                domain_zmax, domain_zmin = depth_z, 0.0

            if padding_type == "hyperelliptical":
                hyper_a = (box_zmin / 2.0) - padding_z
                hyper_b = (box_xmax / 2.0) + padding_x
                domain_zmin = zc + hyper_a
                domain_zmax = 0.0
                domain_xmin = xc - hyper_b
                domain_xmax = xc + hyper_b

            if padding_type == "rectangular":
                domain_zmin = box_zmin - padding_z
                domain_zmax = 0.0
                domain_xmin = 0.0 - padding_x
                domain_xmax = box_xmax + padding_x

            # Interpolating sizing function from segy file
            ef_segy2, _, _, n_samples, n_traces = create_sizing_function(
                fname=fname, hmin=hmin_segy, bbox=segy_bbox, wl=wl, freq=freq,
                pad_type=padding_type, pad_size_x=padding_x, pad_size_z=padding_z,
                grade=grade, vp_water=vp_water
            )

            gmsh.initialize()
            gmsh.option.setNumber("Geometry.Tolerance", 1e-16)
            gmsh.model.add("seismic_model")

            geom_params = build_gmsh_geometry_and_groups(
                gmsh=gmsh, fname=fname, length_x=length_x, depth_z=depth_z,
                padding_type=padding_type, padding_x=padding_x, padding_z=padding_z,
                hyper_n=hyper_n, water_interface=water_interface,
                water_search_value=water_search_value, structured_mesh=structured_mesh,
                minElementSize=minElementSize
            )

            # Standard Params
            z_water_L = geom_params.get("z_water_L")
            z_water_R = geom_params.get("z_water_R")
            pad_x_min = geom_params.get("pad_x_min")
            pad_x_max = geom_params.get("pad_x_max")
            pad_z_min = geom_params.get("pad_z_min")
            a_val = geom_params.get("a_val")
            b_val = geom_params.get("b_val")
            xc = geom_params.get("xc")
            zc = geom_params.get("zc")

            def mesh_size_callback(dim, tag, x, y, z, lc):
                if extend_segy:
                    return float(ef_segy2(np.array([x]), np.array([y]))[0])
                else:
                    z_min_segy, z_max_segy, x_min_segy, x_max_segy = segy_bbox
                    if (x_min_segy <= x <= x_max_segy) and (z_min_segy <= y <= z_max_segy):
                        return float(ef_segy2(np.array([x]), np.array([y]))[0])
                    else:
                        x_proj, y_proj = min(max(x, x_min_segy), x_max_segy), min(max(y, z_min_segy), z_max_segy)
                        base_size = float(ef_segy2(np.array([x_proj]), np.array([y_proj]))[0])
                        tx = abs(x - x_proj) / padding_x if padding_x > 0 else 0.0
                        ty = abs(y - y_proj) / padding_z if padding_z > 0 else 0.0
                        t = min((tx**hyper_n + ty**hyper_n)**(1.0 / hyper_n) if padding_type == "hyperelliptical" else max(tx, ty), 1.0)
                        return float(base_size + t * (h_padding - base_size))

            gmsh.model.mesh.setSizeCallback(mesh_size_callback)
            gmsh.option.setNumber("Mesh.SaveWithoutOrphans", 1)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

            configure_gmsh_2d_meshing_mode(
                gmsh=gmsh,
                structured_mesh=structured_mesh,
                unstructured_quad_mesh=unstructured_quad_mesh,
                padding_type=padding_type,
                min_element_size=minElementSize,
            )

            gmsh.model.mesh.generate(2)
            if unstructured_quad_mesh:
                gmsh.model.mesh.optimize(
                    "QuadQuasiStructured",
                    force=False,
                    niter=50,
                )

            if structured_mesh:
                apply_structured_winslow_smoothing2d(
                    gmsh=gmsh, comm=self.comm, geom_params=geom_params,
                    length_x=length_x, depth_z=depth_z, padding_type=padding_type,
                    water_interface=water_interface, hyper_n=hyper_n,
                    winslow_implementation=winslow_implementation,
                    winslow_iterations=winslow_iterations, winslow_omega=winslow_omega,
                    n_samples=n_samples, n_traces=n_traces,
                    domain_xmin=domain_xmin, domain_xmax=domain_xmax,
                    domain_zmin=domain_zmin, domain_zmax=domain_zmax,
                    ef_segy2=ef_segy2, parallel_print=parallel_print,
                    z_water_L=z_water_L, z_water_R=z_water_R, pad_x_min=pad_x_min,
                    pad_x_max=pad_x_max, pad_z_min=pad_z_min, a_val=a_val,
                    b_val=b_val, xc=xc, zc=zc, apply_winslow=apply_winslow
                )
            # Rotating mesh for axis (z,x,y)
            if padding_type in ["rectangular", "hyperelliptical"]:
                rotate_xz = [
                    0.0, 1.0, 0.0, 0.0,
                    -1.0, 0.0, 0.0, domain_xmax,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0
                ]
            else:
                rotate_xz = [
                    0.0, 1.0, 0.0, -domain_zmin,
                    -1.0, 0.0, 0.0, domain_xmax,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0
                ]

            gmsh.model.mesh.affineTransform(rotate_xz)

            # Flip the x-axis after the rotation (mirror the second coordinate).
            flip_x = [
                1.0, 0.0, 0.0, 0.0,
                0.0, -1.0, 0.0, domain_xmax,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]
            gmsh.model.mesh.affineTransform(flip_x)
            gmsh.write(output_file)
            parallel_print(f"Gmsh mesh written to {output_file}", comm=self.comm)
            gmsh.finalize()

        # MPI Sync
        if self.comm is not None:
            if hasattr(self.comm, 'ensemble_comm'):
                self.comm.ensemble_comm.barrier()
            self.comm.comm.barrier()
            parallel_print("Loading mesh into Firedrake.", comm=self.comm)
            return FireMeshReader(self.output_file_name, comm=self.comm.comm)
        else:
            return FireMeshReader(self.output_file_name)


    def create_gmsh_3D_mesh(self):
        """Create a 3-D Gmsh mesh with the supplied volumetric logic."""
        if gmsh is None:
            raise ImportError("gmsh is not available. Please install it.")

        if self.dimension != 3:
            raise ValueError(
                "create_gmsh_3D_mesh requires dimension=3, "
                f"not {self.dimension}."
            )

        mesh_parameters = self.mesh_parameters
        depth_z = -abs(self.length_z)
        length_x = self.length_x
        length_y = self.length_y
        padding_z = mesh_parameters.padding_z
        padding_x = mesh_parameters.padding_x
        padding_y = getattr(mesh_parameters, "padding_y", padding_x)
        hyper_n = mesh_parameters.hyper_n
        hmin_segy = mesh_parameters.hmin_segy
        wl = self.cpw
        freq = self.source_frequency
        grade = mesh_parameters.grade
        water_search_value = mesh_parameters.water_search_value
        padding_type = mesh_parameters.padding_type
        output_file = self.output_file_name
        water_interface = mesh_parameters.water_interface
        vp_water = mesh_parameters.vp_water
        structured_mesh = mesh_parameters.structured_mesh
        minElementSize = mesh_parameters.min_element_size
        winslow_implementation = mesh_parameters.winslow_implementation
        apply_winslow = mesh_parameters.apply_winslow
        winslow_iterations = mesh_parameters.winslow_iterations
        winslow_omega = mesh_parameters.winslow_omega
        extend_segy = mesh_parameters.extend_segy
        h_padding = mesh_parameters.h_padding
        parallel = getattr(mesh_parameters, "gmsh_parallel", False)
        gmsh_num_threads = getattr(mesh_parameters, "gmsh_num_threads", 1)

        byte_order = getattr(mesh_parameters, "segy_byte_order", "big")
        axes_order = tuple(
            getattr(mesh_parameters, "segy_axes_order", (0, 1, 2))
        )
        axes_order_sort = getattr(
            mesh_parameters, "segy_axes_order_sort", "F"
        )
        binary_dtype = getattr(mesh_parameters, "segy_dtype", "float32")

        if mesh_parameters.segy_velocity_model is not None:
            self.velocity_model = mesh_parameters.segy_velocity_model
            fname = self.velocity_model
            nz = getattr(mesh_parameters, "segy_nz", None)
            nx = getattr(mesh_parameters, "segy_nx", None)
            ny = getattr(mesh_parameters, "segy_ny", None)
            dz = getattr(mesh_parameters, "segy_dz", None)
            dx = getattr(mesh_parameters, "segy_dx", None)
            dy = getattr(mesh_parameters, "segy_dy", None)
            if any(value is None for value in (nz, nx, ny, dz, dx, dy)):
                raise ValueError(
                    "3-D binary Gmsh meshing requires segy_nz, segy_nx, "
                    "segy_ny, segy_dz, segy_dx and segy_dy."
                )

        elif mesh_parameters.velocity_model is not None:
            velocity_model = mesh_parameters.velocity_model
            if not isinstance(velocity_model, dict):
                raise TypeError(
                    "velocity_model must be a grid dictionary when "
                    "segy_velocity_model is not provided."
                )
            if "vp_values" not in velocity_model:
                raise ValueError(
                    "Grid velocity_model must contain the key 'vp_values'."
                )
            vp = np.asarray(velocity_model["vp_values"])
            if vp.ndim != 3:
                raise ValueError(
                    "Grid velocity_model['vp_values'] must be a 3-D array, "
                    f"but received shape {vp.shape}."
                )
            nz, nx, ny = vp.shape
            dz = abs(depth_z) / max(nz - 1, 1)
            dx = length_x / max(nx - 1, 1)
            dy = length_y / max(ny - 1, 1)
            fname = "tmp_velocity_model3D.bin"
            dtype = np.dtype(binary_dtype).newbyteorder(
                ">" if byte_order == "big" else "<"
            )
            np.asarray(vp, dtype=dtype).ravel(
                order=axes_order_sort
            ).tofile(fname)
            self.velocity_model = fname
        else:
            raise ValueError(
                "Gmsh 3-D meshing requires either 'segy_velocity_model' "
                "or a grid 'velocity_model'."
            )

        if self.comm is None or self.comm.ensemble_comm.rank == 0:
            parallel_print("Generating 3D Gmsh mesh...", comm=self.comm)

            segy_bbox = (
                depth_z, 0.0,
                0.0, length_x,
                0.0, length_y,
            )

            if padding_type is None:
                domain_xmin, domain_xmax = 0.0, length_x
                domain_ymin, domain_ymax = 0.0, length_y
                domain_zmin, domain_zmax = depth_z, 0.0
            elif padding_type in ("rectangular", "hyperelliptical"):
                domain_xmin, domain_xmax = -padding_x, length_x + padding_x
                domain_ymin, domain_ymax = -padding_y, length_y + padding_y
                domain_zmin, domain_zmax = depth_z - padding_z, 0.0
            else:
                raise ValueError(
                    "padding_type must be None, 'rectangular', or "
                    "'hyperelliptical'."
                )

            ef_segy3, f_min, f_max, n_samples, n_traces_x, n_traces_y = (  # noqa: RUF059
                create_sizing_function3D(
                    fname=fname,
                    hmin=hmin_segy,
                    bbox=segy_bbox,
                    wl=wl,
                    freq=freq,
                    pad_type=padding_type,
                    pad_size_x=padding_x,
                    pad_size_y=padding_y,
                    pad_size_z=padding_z,
                    grade=grade,
                    vp_water=vp_water,
                    nz=nz,
                    nx=nx,
                    ny=ny,
                    byte_order=byte_order,
                    axes_order=axes_order,
                    axes_order_sort=axes_order_sort,
                    dtype=binary_dtype,
                )
            )

            gmsh.initialize()
            gmsh.clear()
            gmsh.option.setNumber("Geometry.Tolerance", 1e-16)
            gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
            gmsh.model.add("3d_Velocity_Model_Water_Interface")

            try:
                geom_params = build_gmsh_geometry_and_groups3D(
                    gmsh=gmsh,
                    fname=fname,
                    length_x=length_x,
                    length_y=length_y,
                    depth_z=depth_z,
                    padding_type=padding_type,
                    padding_x=padding_x,
                    padding_y=padding_y,
                    padding_z=padding_z,
                    hyper_n=hyper_n,
                    water_interface=water_interface,
                    water_search_value=water_search_value,
                    structured_mesh=structured_mesh,
                    nz=nz,
                    nx=nx,
                    ny=ny,
                    dz=dz,
                    dx=dx,
                    dy=dy,
                    byte_order=byte_order,
                    axes_order=axes_order,
                    axes_order_sort=axes_order_sort,
                    dtype=binary_dtype,
                    comm=self.comm,
                    parallel_print=parallel_print,
                )

                configure_gmsh_mesh_size3D(
                    gmsh=gmsh,
                    ef_segy3=ef_segy3,
                    bbox=segy_bbox,
                    structured_mesh=structured_mesh,
                    parallel=parallel,
                    extend_segy=extend_segy,
                    padding_type=padding_type,
                    padding_x=padding_x,
                    padding_y=padding_y,
                    padding_z=padding_z,
                    hyper_n=hyper_n,
                    h_padding=h_padding,
                    length_x=length_x,
                    length_y=length_y,
                    depth_z=depth_z,
                    nz=nz,
                    nx=nx,
                    ny=ny,
                    gmsh_num_threads=gmsh_num_threads,
                    comm=self.comm,
                    parallel_print=parallel_print,
                )

                if structured_mesh:
                    gmsh.option.setNumber("Mesh.MeshSizeMin", minElementSize)
                    gmsh.option.setNumber("Mesh.MeshSizeMax", minElementSize)
                    gmsh.model.mesh.setTransfiniteAutomatic(
                        [],
                        np.pi,
                        True,
                    )

                gmsh.option.setNumber("Mesh.SaveWithoutOrphans", 1)
                gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
                gmsh.option.setNumber("Mesh.Algorithm3D", 10)
                gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.5)
                gmsh.option.setNumber("Mesh.Optimize", 1)
                gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
                gmsh.model.mesh.generate(3)

                if structured_mesh:
                    apply_structured_winslow_smoothing3D(
                        gmsh=gmsh,
                        comm=self.comm,
                        geom_params=geom_params,
                        length_x=length_x,
                        length_y=length_y,
                        depth_z=depth_z,
                        padding_type=padding_type,
                        padding_x=padding_x,
                        padding_y=padding_y,
                        padding_z=padding_z,
                        water_interface=water_interface,
                        hyper_n=hyper_n,
                        winslow_implementation=winslow_implementation,
                        apply_winslow=apply_winslow,
                        winslow_iterations=winslow_iterations,
                        winslow_omega=winslow_omega,
                        n_samples=n_samples,
                        n_traces_x=n_traces_x,
                        n_traces_y=n_traces_y,
                        domain_xmin=domain_xmin,
                        domain_xmax=domain_xmax,
                        domain_ymin=domain_ymin,
                        domain_ymax=domain_ymax,
                        domain_zmin=domain_zmin,
                        domain_zmax=domain_zmax,
                        ef_segy3=ef_segy3,
                        parallel_print=parallel_print,
                    )

                if padding_type in ("rectangular", "hyperelliptical"):
                    z_translation = 0.0
                else:
                    z_translation = -domain_zmin

                rotate_xyz_to_zxy = [
                    0.0, 0.0, 1.0, z_translation,
                    -1.0, 0.0, 0.0, domain_xmax,
                    0.0, -1.0, 0.0, domain_ymax,
                    0.0, 0.0, 0.0, 1.0,
                ]
                gmsh.model.mesh.affineTransform(rotate_xyz_to_zxy)
                gmsh.write(output_file)
                parallel_print(
                    f"Gmsh 3D mesh written to {output_file}",
                    comm=self.comm,
                )
            finally:
                gmsh.finalize()

        if self.comm is not None:
            if hasattr(self.comm, "ensemble_comm"):
                self.comm.ensemble_comm.barrier()
            self.comm.comm.barrier()
            parallel_print("Loading 3D mesh into Firedrake.", comm=self.comm)
            return FireMeshReader(self.output_file_name, comm=self.comm.comm)
        return FireMeshReader(self.output_file_name)

