from pathlib import Path


def generate_mesh(
    sizing_function, *, dimension, velocity_file,
    length_z, length_x, length_y=None, output_filename="automatic_mesh.msh",
    **parameters,
):
    """Generate, adapt, export, and return a ``meshio.Mesh``.
    """
    from .generation_utils.automatic_mesh import AutomaticMesh
    from .generation_utils.parameters import MeshingParameters
    from .generation_utils.bindings import _prepared_sizing
    import gmsh

    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3.")
    expected = 5 if dimension == 2 else 6
    if not isinstance(sizing_function, (tuple, list)) or len(sizing_function) != expected:
        raise TypeError(
            f"Pass the complete {expected}-item sizing result for {dimension}D."
        )
    if not callable(sizing_function[0]):
        raise TypeError("The first sizing-result item must be a callable field.")
    if dimension == 3 and length_y is None:
        raise ValueError("length_y is required for 3D.")
    if gmsh.isInitialized():
        raise RuntimeError("Finalize the existing Gmsh session before generating a mesh.")

    defaults = dict(
        mesh_type="gmsh_mesh", dimension=dimension,
        length_z=abs(float(length_z)), length_x=float(length_x),
        length_y=length_y, segy_velocity_model=str(velocity_file),
        output_filename=str(output_filename), abc_pad_length=0.0,
        edge_length=parameters.get("min_element_size", 35.0),
        source_frequency=5.0, padding_type=None,
        padding_x=0.0, padding_y=0.0, padding_z=0.0,
    )
    defaults.update(parameters)
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    config = MeshingParameters(
        input_mesh_dictionary=defaults, dimension=dimension,
        quadrilateral=bool(defaults.get("structured_mesh", False)
                           or defaults.get("unstructured_quad_mesh", False)),
    )
    mesher = AutomaticMesh(mesh_parameters=config)
    token = _prepared_sizing.set(tuple(sizing_function))
    try:
        if dimension == 2:
            return mesher.create_gmsh_2D_mesh()
        return mesher.create_gmsh_3D_mesh()
    finally:
        _prepared_sizing.reset(token)
        if gmsh.isInitialized():
            gmsh.finalize()
