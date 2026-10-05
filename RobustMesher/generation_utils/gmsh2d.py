import numpy as np


def configure_gmsh_2d_meshing_mode(
    gmsh, structured_mesh, unstructured_quad_mesh, padding_type, min_element_size
):
    if structured_mesh and unstructured_quad_mesh:
        raise ValueError(
            "'structured_mesh' and 'unstructured_quad_mesh' are mutually exclusive."
        )

    if structured_mesh and padding_type != "hyperelliptical":
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_element_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", min_element_size)
        gmsh.model.mesh.setTransfiniteAutomatic([], np.pi, True)
    elif unstructured_quad_mesh:
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
        gmsh.option.setNumber("Mesh.Smoothing", 10)
        gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 1)
        gmsh.option.setNumber("Mesh.RecombineNodeRepositioning", 1)
        gmsh.option.setNumber("Mesh.RecombineMinimumQuality", 0.85)
        gmsh.model.mesh.optimize("Relocate2D", niter=1)
        gmsh.option.setNumber("Mesh.QuadqsTopologyOptimizationMethods", 111)
        gmsh.option.setNumber("Mesh.QuadqsSizemapMethod", 0)
