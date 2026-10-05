from pathlib import Path
import numpy as np
import segyio

try:
    import gmsh
except ImportError:
    gmsh = None


def check_gmsh(func):
    """Decorator for gmsh check.

    If gmsh isn't available raises ImportError
    """
    def wrapper(*args, **kwargs):
        if gmsh is None:
            raise ImportError("Please install gmsh to use this function.")
        else:
            return func(*args, **kwargs)

    return wrapper


def generate_water_profile_from_segy(segy_path, z_min, z_max, x_min, x_max, value=0.0, tolerance=1e-3, x_chunk=1024):
    """Generate a water interface profile from a SEGY file.

    Reads a 2D SEGY file, computes grid spacing from physical domain bounds,
    and searches down the z-axis to find the water interface.

    Parameters
    ----------
    segy_path : str or pathlib.Path
        Path to the SEGY velocity model.
    z_min : float
        Minimum z-coordinate of the physical domain.
    z_max : float
        Maximum z-coordinate of the physical domain.
    x_min : float
        Minimum x-coordinate of the physical domain.
    x_max : float
        Maximum x-coordinate of the physical domain.
    value : float, optional
        Velocity value identifying the water layer. Default is 0.0.
    tolerance : float, optional
        Numerical tolerance for matching the water velocity value. Default is 1e-3.
    x_chunk : int, optional
        Number of traces to process in memory at once. Default is 1024.

    Returns
    -------
    tuple
        A tuple containing:
        - Xs (ndarray): 1D array of physical X coordinates.
        - Z_bottom (ndarray): 1D array of the corresponding Z coordinates
          representing the water interface depth.

    Raises
    ------
    FileNotFoundError
        If the specified SEGY file does not exist.
    """
    segy_path = Path(segy_path)
    if not segy_path.exists():
        raise FileNotFoundError(f"SEGY file not found: {segy_path}")

    print(f"Reading SEGY file: {segy_path}")

    # Extract SEGY Data and Compute Spacing
    with segyio.open(segy_path, 'r', ignore_geometry=True) as segy:
        nx_tot = segy.tracecount
        nz_tot = len(segy.samples)

        total_length_z = abs(float(z_max) - float(z_min))
        _dz = total_length_z / (nz_tot - 1) if nz_tot > 1 else total_length_z

        total_length_x = abs(float(x_max) - float(x_min))
        _dx = total_length_x / (nx_tot - 1) if nx_tot > 1 else total_length_x

        print(f"Detected parameters: nx={nx_tot}, nz={nz_tot}")
        print(f"Calculated spacing: dx={_dx:.2f} m, dz={_dz:.2f} m")

        plane_zx = segy.trace.raw[:].T

    # Domain Ranges
    z_top = float(max(z_min, z_max))
    z_bot = float(min(z_min, z_max))
    x_low, x_high = sorted([float(x_min), float(x_max)])

    Lx = (nx_tot - 1) * _dx
    Lz = (nz_tot - 1) * _dz

    x_low = np.clip(x_low, 0.0, Lx)
    x_high = np.clip(x_high, 0.0, Lx)
    z_top = np.clip(z_top, -Lz, 0.0)
    z_bot = np.clip(z_bot, -Lz, 0.0)

    # Map meters → nearest indices
    ix_min = int(np.rint(x_low / _dx))
    ix_max = int(np.rint(x_high / _dx))
    ix_min, ix_max = max(0, min(ix_min, nx_tot - 1)), max(0, min(ix_max, nx_tot - 1))
    if ix_max < ix_min:
        ix_min, ix_max = ix_max, ix_min

    iz_top = int(np.rint(-z_top / _dz))
    iz_bot = int(np.rint(-z_bot / _dz))
    iz_top = max(0, min(iz_top, nz_tot - 1))
    iz_bot = max(0, min(iz_bot, nz_tot - 1))
    if iz_bot < iz_top:
        iz_top, iz_bot = iz_bot, iz_top

    # Physical X coordinates for the points
    Xs = _dx * np.arange(ix_min, ix_max + 1, dtype=np.float64)
    Nx = (ix_max - ix_min)
    x_chunk = int(max(1, x_chunk))

    Z_bottom = np.empty(Nx + 1, dtype=np.float64)
    target = float(value)
    tol = float(tolerance)

    # Water Interface Search
    for xs in range(0, Nx + 1, x_chunk):
        xe = min(xs + x_chunk, Nx + 1)
        ix_tile = ix_min + np.arange(xs, xe, dtype=int)

        block = plane_zx[iz_top:iz_bot + 1, ix_tile]
        block = np.asarray(block, dtype=np.float32)

        in_water = np.abs(block - target) <= tol
        non_water = ~in_water

        any_non_water = np.any(non_water, axis=0)
        first_non_idx = np.argmax(non_water, axis=0).astype(np.int32)

        first_non_idx = np.where(any_non_water, first_non_idx, (iz_bot - iz_top)).astype(np.int32)

        k_global = iz_top + first_non_idx
        z_phys = -k_global.astype(np.float64) * _dz
        Z_bottom[xs:xe] = z_phys

    # Calculate Final Shifted Coordinates
    x_shift = float(min(x_min, x_max))
    Xs = Xs + x_shift

    return Xs, Z_bottom


@check_gmsh
def get_surface_entities_by_physical_name(name):
    """Retrieve Gmsh surface entity tags belonging to a specified physical group.

    Parameters
    ----------
    name : str
        The physical name of the target group in the Gmsh model.

    Returns
    -------
    list
        A list of integer tags representing the geometric surface entities
        associated with the physical group name.
    """
    entities = []
    for dim, pg_tag in gmsh.model.getPhysicalGroups(dim=2):
        if gmsh.model.getPhysicalName(dim, pg_tag) == name:
            entities.extend(gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag))
    return entities


@check_gmsh
def get_nodes_on_surface_entities(tag_to_index, surface_entities):
    """Find node indices belonging to specific geometric surfaces.

    Parameters
    ----------
    tag_to_index : dict
        Mapping from Gmsh node tags to local internal node indices.
    surface_entities : list
        List of Gmsh geometric surface entity tags.

    Returns
    -------
    set
        A set of local node indices located on the provided surfaces.
    """
    nodes = set()
    for surf in surface_entities:
        n_tags, _, _ = gmsh.model.mesh.getNodes(2, surf, includeBoundary=True)
        for t in n_tags:
            if t in tag_to_index:
                nodes.add(tag_to_index[t])
    return nodes


@check_gmsh
def get_water_interface_node_indices(tag_to_index, water_surface_entities, length_x, tol=1e-8):
    """Identify node indices lying on the water interface.

    Scans the boundaries of water region surface entities and extracts nodes
    that do not sit on the top, left, or right edges of the domain, thus
    isolating the bottom interface.

    Parameters
    ----------
    tag_to_index : dict
        Mapping from Gmsh node tags to local internal node indices.
    water_surface_entities : list
        List of surface tags representing the water layer.
    length_x : float
        Total length of the domain in the x-direction.
    tol : float, optional
        Geometric tolerance for boundary checks. Default is 1e-8.

    Returns
    -------
    set
        A set of local node indices defining the water interface.
    """
    interface_nodes = set()
    for surf in water_surface_entities:
        boundary = gmsh.model.getBoundary([(2, surf)], oriented=False, recursive=False)
        for dim, curve_tag in boundary:
            if dim != 1:
                continue
            n_tags, coords, _ = gmsh.model.mesh.getNodes(1, curve_tag, includeBoundary=True)
            if len(n_tags) == 0:
                continue
            xy = np.asarray(coords, dtype=float).reshape(-1, 3)[:, :2]
            xs, zs = xy[:, 0], xy[:, 1]
            is_top = np.all(np.abs(zs - 0.0) < tol)
            is_left = np.all(np.abs(xs - 0.0) < tol)
            is_right = np.all(np.abs(xs - length_x) < tol)
            if not (is_top or is_left or is_right):
                for t in n_tags:
                    if t in tag_to_index:
                        interface_nodes.add(tag_to_index[t])
    return interface_nodes


def align_water_columns_to_interface_x(points_2d, water_surface_nodes, interface_nodes, quads):
    """Align water nodes vertically with the interface nodes.

    Adjusts the x-coordinates of nodes within the water column so that they
    align vertically with the varying water-bottom interface nodes,
    improving mesh orthogonality in the water layer.

    Parameters
    ----------
    points_2d : ndarray
        2D array of shape (N, 2) containing node coordinates [x, z].
    water_surface_nodes : set
        Set of node indices located inside the water layer.
    interface_nodes : set
        Set of node indices defining the water bottom interface.
    quads : ndarray
        2D array containing quadrilateral element connectivity.

    Returns
    -------
    tuple
        A tuple containing:
        - ndarray: Updated node coordinates.
        - int: Number of nodes successfully snapped/aligned.
        - int: Number of vertical columns processed.
    """
    if not interface_nodes:
        return points_2d, 0, 0
    edge_to_quads = {}
    for q in quads:
        if not all(n in water_surface_nodes for n in q):
            continue
        pair_A = [frozenset((q[0], q[1])), frozenset((q[2], q[3]))]
        pair_B = [frozenset((q[1], q[2])), frozenset((q[3], q[0]))]
        q_tuple = tuple(q)
        for edge in pair_A + pair_B:
            if edge not in edge_to_quads:
                edge_to_quads[edge] = []
            edge_to_quads[edge].append(q_tuple)

    horizontal_queue = []
    for e in edge_to_quads.keys():
        u, v = list(e)
        if abs(points_2d[u, 1] - 0.0) < 1e-3 and abs(points_2d[v, 1] - 0.0) < 1e-3:
            horizontal_queue.append(e)

    horizontal_edges = set(horizontal_queue)
    vertical_edges = set()
    visited_quads = set()

    while horizontal_queue:
        curr_e = horizontal_queue.pop(0)
        if curr_e not in edge_to_quads:
            continue
        for q in edge_to_quads[curr_e]:
            if q in visited_quads:
                continue
            visited_quads.add(q)
            pair_A = [frozenset((q[0], q[1])), frozenset((q[2], q[3]))]
            pair_B = [frozenset((q[1], q[2])), frozenset((q[3], q[0]))]
            if curr_e in pair_A:
                opp = pair_A[1] if curr_e == pair_A[0] else pair_A[0]
                vert1, vert2 = pair_B[0], pair_B[1]
            elif curr_e in pair_B:
                opp = pair_B[1] if curr_e == pair_B[0] else pair_B[0]
                vert1, vert2 = pair_A[0], pair_A[1]
            else:
                continue

            if opp not in horizontal_edges:
                horizontal_edges.add(opp)
                horizontal_queue.append(opp)
            vertical_edges.add(vert1)
            vertical_edges.add(vert2)

    vertical_adj = {i: set() for i in water_surface_nodes}
    for e in vertical_edges:
        u, v = list(e)
        if u in vertical_adj and v in vertical_adj:
            vertical_adj[u].add(v)
            vertical_adj[v].add(u)

    visited = set()
    columns = []
    for i in water_surface_nodes:
        if i not in visited:
            col = []
            stack = [i]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    col.append(curr)
                    stack.extend(list(vertical_adj[curr]))
            if len(col) > 1:
                columns.append(col)

    snapped = 0
    n_cols = 0
    for col in columns:
        spline_x = None
        for idx in col:
            if idx in interface_nodes:
                spline_x = points_2d[idx, 0]
                break
        if spline_x is not None:
            n_cols += 1
            for idx in col:
                if points_2d[idx, 0] != spline_x:
                    points_2d[idx, 0] = spline_x
                    snapped += 1
    return points_2d, snapped, n_cols


def intersect(x0, z0, dx, dz, xc, zc, a_val, b_val, hyper_n):
    def f(s):
        x, z = x0 + s * dx, z0 + s * dz
        return (abs(x - xc) / a_val)**hyper_n + (abs(z - zc) / b_val)**hyper_n - 1.0
    s_low, s_high = 0.0, 1.0
    while f(s_high) < 0:
        s_high *= 2.0
    for _ in range(100):
        s_mid = (s_low + s_high) / 2.0
        if f(s_mid) > 0:
            s_high = s_mid
        else:
            s_low = s_mid
    return x0 + s_mid * dx, z0 + s_mid * dz


def get_theta(x, z, xc, zc, a_val, b_val, hyper_n):
    vx, vz = (x - xc) / a_val, (z - zc) / b_val
    vx = vx if abs(vx) > 1e-12 else 0.0
    vz = vz if abs(vz) > 1e-12 else 0.0
    return np.arctan2(np.sign(vz) * np.abs(vz)**(hyper_n / 2.0), np.sign(vx) * np.abs(vx)**(hyper_n / 2.0))


def make_arc(p1_tag, p2_tag, x1, z1, x2, z2, xc, zc, a_val, b_val, hyper_n, num_pts=25):
    t1, t2 = get_theta(x1, z1, xc, zc, a_val, b_val, hyper_n), get_theta(x2, z2, xc, zc, a_val, b_val, hyper_n)
    if t2 - t1 > np.pi:
        t1 += 2 * np.pi
    elif t1 - t2 > np.pi:
        t2 += 2 * np.pi
    pts = [p1_tag]
    for t in np.linspace(t1, t2, num_pts)[1:-1]:
        cos_t, sin_t = np.cos(t), np.sin(t)
        x = xc + a_val * np.sign(cos_t) * np.abs(cos_t)**(2.0 / hyper_n)
        z = zc + b_val * np.sign(sin_t) * np.abs(sin_t)**(2.0 / hyper_n)
        pts.append(gmsh.model.occ.addPoint(x, z, 0.0))
    pts.append(p2_tag)
    return gmsh.model.occ.addSpline(pts)


def build_gmsh_geometry_and_groups(
    gmsh, fname, length_x, depth_z, padding_type, padding_x, padding_z,
    hyper_n, water_interface, water_search_value, structured_mesh, minElementSize
):
    """
    Generates the geometry, handles padding/water interfaces, and creates
    physical groups for the Gmsh model. Returns a dictionary of boundary
    parameters needed for subsequent mesh smoothing operations.
    """
    xc = length_x / 2.0
    zc = depth_z / 2.0
    a_val, b_val = None, None

    pad_x_min, pad_x_max, pad_z_min = None, None, None
    z_water_L, z_water_R = None, None

    if padding_type == "hyperelliptical":
        a_val = (length_x / 2.0) + padding_x
        b_val = abs(depth_z / 2.0) + padding_z
    elif padding_type == "rectangular":
        pad_x_min, pad_x_max = -padding_x, length_x + padding_x
        pad_z_min = depth_z - padding_z

    if water_interface:
        Xs, Z_bottom = generate_water_profile_from_segy(
            fname, z_min=0.0, z_max=depth_z, x_min=0.0, x_max=length_x,
            value=water_search_value, tolerance=0.001
        )
        z_water_L, z_water_R = float(Z_bottom[0]), float(Z_bottom[-1])

        pB = [gmsh.model.occ.addPoint(x, float(z), 0.0) for x, z in zip(Xs, Z_bottom)]
        bottom_curve = gmsh.model.occ.addSpline(pB)
        gmsh.model.occ.synchronize()

        pt_top_left = gmsh.model.occ.addPoint(float(Xs[0]), 0.0, 0.0)
        pt_top_right = gmsh.model.occ.addPoint(float(Xs[-1]), 0.0, 0.0)
        line_left = gmsh.model.occ.addLine(pt_top_left, pB[0])
        line_right = gmsh.model.occ.addLine(pB[-1], pt_top_right)
        line_top = gmsh.model.occ.addLine(pt_top_right, pt_top_left)

        curve_loop = gmsh.model.occ.addCurveLoop([line_left, bottom_curve, line_right, line_top])
        water_surface = gmsh.model.occ.addPlaneSurface([curve_loop])
        gmsh.model.occ.synchronize()

        if padding_type is None:
            rectangle_tag = gmsh.model.occ.addRectangle(0, 0, 0, length_x, depth_z)
            gmsh.model.occ.synchronize()
            fragment_result, fragment_map = gmsh.model.occ.fragment([(2, rectangle_tag), (2, water_surface)], [])
            gmsh.model.occ.synchronize()
            water_tags = [tag for dim, tag in fragment_map[1]]
            clipped_rect_tags = [tag for dim, tag in fragment_map[0] if tag not in water_tags]
            gmsh.model.addPhysicalGroup(2, water_tags, name="WaterSurface")
            gmsh.model.addPhysicalGroup(2, clipped_rect_tags, name="SubSurface")

            water_boundary_curves = {
                tag for dim, tag in gmsh.model.getBoundary(
                    [(2, tag) for tag in water_tags],
                    combined=True,
                    oriented=False,
                    recursive=False,
                )
                if dim == 1
            }
            rock_boundary_curves = {
                tag for dim, tag in gmsh.model.getBoundary(
                    [(2, tag) for tag in clipped_rect_tags],
                    combined=True,
                    oriented=False,
                    recursive=False,
                )
                if dim == 1
            }

            water_interface_curves = sorted(water_boundary_curves & rock_boundary_curves)
            water_outer_curves = sorted(water_boundary_curves - set(water_interface_curves))
            rock_outer_curves = sorted(rock_boundary_curves - set(water_interface_curves))

            tol = 1.0e-6 * max(1.0, abs(length_x), abs(depth_z))

            def _classify_rectangular_curves(curves):
                top, left, right, bottom = [], [], [], []
                for curve in curves:
                    xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, curve)
                    if abs(ymin) <= tol and abs(ymax) <= tol:
                        top.append(curve)
                    elif abs(xmin) <= tol and abs(xmax) <= tol:
                        left.append(curve)
                    elif abs(xmin - length_x) <= tol and abs(xmax - length_x) <= tol:
                        right.append(curve)
                    elif abs(ymin - depth_z) <= tol and abs(ymax - depth_z) <= tol:
                        bottom.append(curve)
                return top, left, right, bottom

            water_top, water_left, water_right, _ = _classify_rectangular_curves(water_outer_curves)
            _, rock_left, rock_right, rock_bottom = _classify_rectangular_curves(rock_outer_curves)

            gmsh.model.addPhysicalGroup(1, water_top, 1, "InternalTop")
            gmsh.model.addPhysicalGroup(1, water_left, 2, "InternalWaterLeft")
            gmsh.model.addPhysicalGroup(1, water_right, 3, "InternalWaterRight")
            gmsh.model.addPhysicalGroup(1, water_interface_curves, 4, "WaterInterface")
            gmsh.model.addPhysicalGroup(1, rock_left, 5, "InternalRockLeft")
            gmsh.model.addPhysicalGroup(1, rock_right, 6, "InternalRockRight")
            gmsh.model.addPhysicalGroup(1, rock_bottom, 7, "InternalBottom")

        if padding_type == "rectangular":
            pt_rock_bl = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)
            pt_rock_br = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
            pt_pad_tl_top = gmsh.model.occ.addPoint(pad_x_min, 0.0, 0.0)
            pt_pad_tl_bot = gmsh.model.occ.addPoint(pad_x_min, z_water_L, 0.0)
            pt_pad_l_bot = gmsh.model.occ.addPoint(pad_x_min, depth_z, 0.0)
            pt_pad_tr_top = gmsh.model.occ.addPoint(pad_x_max, 0.0, 0.0)
            pt_pad_tr_bot = gmsh.model.occ.addPoint(pad_x_max, z_water_R, 0.0)
            pt_pad_r_bot = gmsh.model.occ.addPoint(pad_x_max, depth_z, 0.0)
            pt_pad_b_left = gmsh.model.occ.addPoint(pad_x_min, pad_z_min, 0.0)
            pt_pad_bc_left = gmsh.model.occ.addPoint(0.0, pad_z_min, 0.0)
            pt_pad_bc_right = gmsh.model.occ.addPoint(length_x, pad_z_min, 0.0)
            pt_pad_b_right = gmsh.model.occ.addPoint(pad_x_max, pad_z_min, 0.0)

            rock_right = gmsh.model.occ.addLine(pB[-1], pt_rock_br)
            rock_bottom = gmsh.model.occ.addLine(pt_rock_br, pt_rock_bl)
            rock_left = gmsh.model.occ.addLine(pt_rock_bl, pB[0])
            pad_tr_top = gmsh.model.occ.addLine(pt_top_right, pt_pad_tr_top)
            pad_tr_right = gmsh.model.occ.addLine(pt_pad_tr_top, pt_pad_tr_bot)
            pad_tr_bot = gmsh.model.occ.addLine(pt_pad_tr_bot, pB[-1])
            pad_mr_right = gmsh.model.occ.addLine(pt_pad_tr_bot, pt_pad_r_bot)
            pad_mr_bot = gmsh.model.occ.addLine(pt_pad_r_bot, pt_rock_br)
            pad_tl_top = gmsh.model.occ.addLine(pt_top_left, pt_pad_tl_top)
            pad_tl_left = gmsh.model.occ.addLine(pt_pad_tl_top, pt_pad_tl_bot)
            pad_tl_bot = gmsh.model.occ.addLine(pt_pad_tl_bot, pB[0])
            pad_ml_left = gmsh.model.occ.addLine(pt_pad_tl_bot, pt_pad_l_bot)
            pad_ml_bot = gmsh.model.occ.addLine(pt_pad_l_bot, pt_rock_bl)
            pad_bl_left = gmsh.model.occ.addLine(pt_pad_l_bot, pt_pad_b_left)
            pad_bl_bot = gmsh.model.occ.addLine(pt_pad_b_left, pt_pad_bc_left)
            pad_bl_right = gmsh.model.occ.addLine(pt_pad_bc_left, pt_rock_bl)
            pad_bc_bot = gmsh.model.occ.addLine(pt_pad_bc_left, pt_pad_bc_right)
            pad_bc_right = gmsh.model.occ.addLine(pt_pad_bc_right, pt_rock_br)
            pad_br_bot = gmsh.model.occ.addLine(pt_pad_bc_right, pt_pad_b_right)
            pad_br_right = gmsh.model.occ.addLine(pt_pad_b_right, pt_pad_r_bot)

            surf_rock = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([bottom_curve, rock_right, rock_bottom, rock_left])])
            surf_pad_tr = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_right, pad_tr_top, pad_tr_right, pad_tr_bot])])
            surf_pad_mr = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-pad_tr_bot, pad_mr_right, pad_mr_bot, -rock_right])])
            surf_pad_tl = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([pad_tl_top, pad_tl_left, pad_tl_bot, -line_left])])
            surf_pad_ml = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-pad_tl_bot, pad_ml_left, pad_ml_bot, rock_left])])
            surf_pad_bl = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-pad_ml_bot, pad_bl_left, pad_bl_bot, pad_bl_right])])
            surf_pad_bc = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-pad_bl_right, pad_bc_bot, pad_bc_right, rock_bottom])])
            surf_pad_br = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-pad_bc_right, pad_br_bot, pad_br_right, pad_mr_bot])])

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [water_surface], name="WaterSurface")
            gmsh.model.addPhysicalGroup(2, [surf_rock], name="SubSurface")
            gmsh.model.addPhysicalGroup(2, [surf_pad_tr, surf_pad_mr, surf_pad_tl, surf_pad_ml, surf_pad_bl, surf_pad_bc, surf_pad_br], name="Padding")

            gmsh.model.addPhysicalGroup(1, [pad_tl_top], 1, "PaddingTopLeft")
            gmsh.model.addPhysicalGroup(1, [pad_tl_left, pad_ml_left], 2, "PaddingLeftUpper")
            gmsh.model.addPhysicalGroup(1, [pad_bl_left], 3, "PaddingLeftLower")
            gmsh.model.addPhysicalGroup(1, [pad_bl_bot], 4, "PaddingBottomLeft")
            gmsh.model.addPhysicalGroup(1, [pad_bc_bot], 5, "PaddingBottomCenter")
            gmsh.model.addPhysicalGroup(1, [pad_br_bot], 6, "PaddingBottomRight")
            gmsh.model.addPhysicalGroup(1, [pad_br_right], 7, "PaddingRightLower")
            gmsh.model.addPhysicalGroup(1, [pad_mr_right, pad_tr_right], 8, "PaddingRightUpper")
            gmsh.model.addPhysicalGroup(1, [pad_tr_top], 9, "PaddingTopRight")

            gmsh.model.addPhysicalGroup(1, [line_top], 10, "InternalTop")
            gmsh.model.addPhysicalGroup(1, [line_left], 11, "InternalWaterLeft")
            gmsh.model.addPhysicalGroup(1, [line_right], 12, "InternalWaterRight")
            gmsh.model.addPhysicalGroup(1, [bottom_curve], 13, "WaterInterface")
            gmsh.model.addPhysicalGroup(1, [rock_left], 14, "InternalRockLeft")
            gmsh.model.addPhysicalGroup(1, [rock_right], 15, "InternalRockRight")
            gmsh.model.addPhysicalGroup(1, [rock_bottom], 16, "InternalBottom")

        if padding_type == "hyperelliptical":

            pt_rock_bl = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)
            pt_rock_br = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
            z_mid_L, z_mid_R = (z_water_L + depth_z) / 2.0, (z_water_R + depth_z) / 2.0
            x_mid_L, x_mid_R = length_x * 0.25, length_x * 0.75

            pt_mid_L = gmsh.model.occ.addPoint(0.0, z_mid_L, 0.0)
            pt_mid_R = gmsh.model.occ.addPoint(length_x, z_mid_R, 0.0)
            pt_bot_midL = gmsh.model.occ.addPoint(x_mid_L, depth_z, 0.0)
            pt_bot_midR = gmsh.model.occ.addPoint(x_mid_R, depth_z, 0.0)

            x_O_TL, z_O_TL = intersect(0.0, 0.0, -1, 0, xc, zc, a_val, b_val, hyper_n)
            x_O_WL, z_O_WL = intersect(0.0, z_water_L, -1, 0, xc, zc, a_val, b_val, hyper_n)
            x_O_ML, z_O_ML = intersect(0.0, z_mid_L, -1, 0, xc, zc, a_val, b_val, hyper_n)
            x_O_BL_45, z_O_BL_45 = intersect(0.0, depth_z, -1, -1, xc, zc, a_val, b_val, hyper_n)
            x_O_BML, z_O_BML = intersect(x_mid_L, depth_z, 0, -1, xc, zc, a_val, b_val, hyper_n)
            x_O_BMR, z_O_BMR = intersect(x_mid_R, depth_z, 0, -1, xc, zc, a_val, b_val, hyper_n)
            x_O_BR_45, z_O_BR_45 = intersect(length_x, depth_z, 1, -1, xc, zc, a_val, b_val, hyper_n)
            x_O_MR, z_O_MR = intersect(length_x, z_mid_R, 1, 0, xc, zc, a_val, b_val, hyper_n)
            x_O_WR, z_O_WR = intersect(length_x, z_water_R, 1, 0, xc, zc, a_val, b_val, hyper_n)
            x_O_TR, z_O_TR = intersect(length_x, 0.0, 1, 0, xc, zc, a_val, b_val, hyper_n)

            pt_O_TL = gmsh.model.occ.addPoint(x_O_TL, z_O_TL, 0.0)
            pt_O_WL = gmsh.model.occ.addPoint(x_O_WL, z_O_WL, 0.0)
            pt_O_ML = gmsh.model.occ.addPoint(x_O_ML, z_O_ML, 0.0)
            pt_O_BL_45 = gmsh.model.occ.addPoint(x_O_BL_45, z_O_BL_45, 0.0)
            pt_O_BML = gmsh.model.occ.addPoint(x_O_BML, z_O_BML, 0.0)
            pt_O_BMR = gmsh.model.occ.addPoint(x_O_BMR, z_O_BMR, 0.0)
            pt_O_BR_45 = gmsh.model.occ.addPoint(x_O_BR_45, z_O_BR_45, 0.0)
            pt_O_MR = gmsh.model.occ.addPoint(x_O_MR, z_O_MR, 0.0)
            pt_O_WR = gmsh.model.occ.addPoint(x_O_WR, z_O_WR, 0.0)
            pt_O_TR = gmsh.model.occ.addPoint(x_O_TR, z_O_TR, 0.0)

            rock_R_upper = gmsh.model.occ.addLine(pB[-1], pt_mid_R)
            rock_R_lower = gmsh.model.occ.addLine(pt_mid_R, pt_rock_br)
            rock_B_right = gmsh.model.occ.addLine(pt_rock_br, pt_bot_midR)
            rock_B_mid = gmsh.model.occ.addLine(pt_bot_midR, pt_bot_midL)
            rock_B_left = gmsh.model.occ.addLine(pt_bot_midL, pt_rock_bl)
            rock_L_lower = gmsh.model.occ.addLine(pt_rock_bl, pt_mid_L)
            rock_L_upper = gmsh.model.occ.addLine(pt_mid_L, pB[0])

            ray_TL = gmsh.model.occ.addLine(pt_top_left, pt_O_TL)
            ray_WL = gmsh.model.occ.addLine(pB[0], pt_O_WL)
            ray_ML = gmsh.model.occ.addLine(pt_mid_L, pt_O_ML)
            ray_BL_45 = gmsh.model.occ.addLine(pt_rock_bl, pt_O_BL_45)
            ray_BML = gmsh.model.occ.addLine(pt_bot_midL, pt_O_BML)
            ray_BMR = gmsh.model.occ.addLine(pt_bot_midR, pt_O_BMR)
            ray_BR_45 = gmsh.model.occ.addLine(pt_rock_br, pt_O_BR_45)
            ray_MR = gmsh.model.occ.addLine(pt_mid_R, pt_O_MR)
            ray_WR = gmsh.model.occ.addLine(pB[-1], pt_O_WR)
            ray_TR = gmsh.model.occ.addLine(pt_top_right, pt_O_TR)

            arc_TL_WL = make_arc(pt_O_TL, pt_O_WL, x_O_TL, z_O_TL, x_O_WL, z_O_WL, xc, zc, a_val, b_val, hyper_n)
            arc_WL_ML = make_arc(pt_O_WL, pt_O_ML, x_O_WL, z_O_WL, x_O_ML, z_O_ML, xc, zc, a_val, b_val, hyper_n)
            arc_ML_BL45 = make_arc(pt_O_ML, pt_O_BL_45, x_O_ML, z_O_ML, x_O_BL_45, z_O_BL_45, xc, zc, a_val, b_val, hyper_n)
            arc_BL45_BML = make_arc(pt_O_BL_45, pt_O_BML, x_O_BL_45, z_O_BL_45, x_O_BML, z_O_BML, xc, zc, a_val, b_val, hyper_n)
            arc_BML_BMR = make_arc(pt_O_BML, pt_O_BMR, x_O_BML, z_O_BML, x_O_BMR, z_O_BMR, xc, zc, a_val, b_val, hyper_n)
            arc_BMR_BR45 = make_arc(pt_O_BMR, pt_O_BR_45, x_O_BMR, z_O_BMR, x_O_BR_45, z_O_BR_45, xc, zc, a_val, b_val, hyper_n)
            arc_BR45_MR = make_arc(pt_O_BR_45, pt_O_MR, x_O_BR_45, z_O_BR_45, x_O_MR, z_O_MR, xc, zc, a_val, b_val, hyper_n)
            arc_MR_WR = make_arc(pt_O_MR, pt_O_WR, x_O_MR, z_O_MR, x_O_WR, z_O_WR, xc, zc, a_val, b_val, hyper_n)
            arc_WR_TR = make_arc(pt_O_WR, pt_O_TR, x_O_WR, z_O_WR, x_O_TR, z_O_TR, xc, zc, a_val, b_val, hyper_n)

            surf_rock = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([bottom_curve, rock_R_upper, rock_R_lower, rock_B_right, rock_B_mid, rock_B_left, rock_L_lower, rock_L_upper])])
            surf_pad_TL = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_left, ray_WL, -arc_TL_WL, -ray_TL])])
            surf_pad_ML1 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_L_upper, ray_ML, -arc_WL_ML, -ray_WL])])
            surf_pad_ML2 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_L_lower, ray_BL_45, -arc_ML_BL45, -ray_ML])])
            surf_pad_B_L = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_B_left, ray_BML, -arc_BL45_BML, -ray_BL_45])])
            surf_pad_B_M = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_B_mid, ray_BMR, -arc_BML_BMR, -ray_BML])])
            surf_pad_B_R = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_B_right, ray_BR_45, -arc_BMR_BR45, -ray_BMR])])
            surf_pad_MR2 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_R_lower, ray_MR, -arc_BR45_MR, -ray_BR_45])])
            surf_pad_MR1 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_R_upper, ray_WR, -arc_MR_WR, -ray_MR])])
            surf_pad_TR = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_right, ray_TR, -arc_WR_TR, -ray_WR])])

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [water_surface], name="WaterSurface")
            gmsh.model.addPhysicalGroup(2, [surf_rock], name="SubSurface")
            gmsh.model.addPhysicalGroup(2, [surf_pad_TL, surf_pad_ML1, surf_pad_ML2, surf_pad_B_L, surf_pad_B_M, surf_pad_B_R, surf_pad_MR2, surf_pad_MR1, surf_pad_TR], name="Padding")

            gmsh.model.addPhysicalGroup(1, [ray_TL], 1, "PaddingTopLeft")
            gmsh.model.addPhysicalGroup(1, [ray_TR], 2, "PaddingTopRight")
            gmsh.model.addPhysicalGroup(
                1,
                [arc_TL_WL, arc_WL_ML, arc_ML_BL45],
                3,
                "HyperellipseLeft",
            )
            gmsh.model.addPhysicalGroup(1, [arc_BL45_BML], 4, "HyperellipseBottomLeft")
            gmsh.model.addPhysicalGroup(1, [arc_BML_BMR], 5, "HyperellipseBottomCenter")
            gmsh.model.addPhysicalGroup(1, [arc_BMR_BR45], 6, "HyperellipseBottomRight")
            gmsh.model.addPhysicalGroup(
                1,
                [arc_BR45_MR, arc_MR_WR, arc_WR_TR],
                7,
                "HyperellipseRight",
            )

            gmsh.model.addPhysicalGroup(1, [line_top], 8, "InternalTop")
            gmsh.model.addPhysicalGroup(1, [line_left], 9, "InternalWaterLeft")
            gmsh.model.addPhysicalGroup(1, [line_right], 10, "InternalWaterRight")
            gmsh.model.addPhysicalGroup(1, [bottom_curve], 11, "WaterInterface")
            gmsh.model.addPhysicalGroup(
                1, [rock_L_upper, rock_L_lower], 12, "InternalRockLeft"
            )
            gmsh.model.addPhysicalGroup(
                1, [rock_R_upper, rock_R_lower], 13, "InternalRockRight"
            )
            gmsh.model.addPhysicalGroup(
                1, [rock_B_left, rock_B_mid, rock_B_right], 14, "InternalBottom"
            )

            if structured_mesh:
                len_radial = max(padding_x, padding_z)
                len_x_left, len_x_mid, len_x_right = length_x * 0.25, length_x * 0.50, length_x * 0.25
                len_z_water = abs(z_water_L)
                len_z_rock_upper, len_z_rock_lower = abs(z_mid_L - z_water_L), abs(depth_z - z_mid_L)

                N_radial = max(2, int(np.ceil(len_radial / minElementSize)) + 1)
                N_X_left = max(2, int(np.ceil(len_x_left / minElementSize)) + 1)
                N_X_mid = max(2, int(np.ceil(len_x_mid / minElementSize)) + 1)
                N_X_right = max(2, int(np.ceil(len_x_right / minElementSize)) + 1)
                N_X_total = N_X_left + N_X_mid + N_X_right - 2
                N_Z_water = max(2, int(np.ceil(len_z_water / minElementSize)) + 1)
                N_Z_rock_upper = max(2, int(np.ceil(len_z_rock_upper / minElementSize)) + 1)
                N_Z_rock_lower = max(2, int(np.ceil(len_z_rock_lower / minElementSize)) + 1)

                for curve in [ray_TL, ray_WL, ray_ML, ray_BL_45, ray_BML, ray_BMR, ray_BR_45, ray_MR, ray_WR, ray_TR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_radial)
                for curve in [line_left, line_right, arc_TL_WL, arc_WR_TR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_Z_water)
                for curve in [rock_L_upper, rock_R_upper, arc_WL_ML, arc_MR_WR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_Z_rock_upper)
                for curve in [rock_L_lower, rock_R_lower, arc_ML_BL45, arc_BR45_MR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_Z_rock_lower)
                for curve in [rock_B_left, arc_BL45_BML]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_X_left)
                for curve in [rock_B_mid, arc_BML_BMR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_X_mid)
                for curve in [rock_B_right, arc_BMR_BR45]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_X_right)
                for curve in [line_top, bottom_curve]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_X_total)

                padding_surfs = [surf_pad_TL, surf_pad_ML1, surf_pad_ML2, surf_pad_B_L, surf_pad_B_M, surf_pad_B_R, surf_pad_MR2, surf_pad_MR1, surf_pad_TR]
                for surf in padding_surfs:
                    gmsh.model.mesh.setTransfiniteSurface(surf)
                    gmsh.model.mesh.setRecombine(2, surf)

                gmsh.model.mesh.setTransfiniteSurface(water_surface, cornerTags=[pt_top_left, pt_top_right, pB[-1], pB[0]])
                gmsh.model.mesh.setRecombine(2, water_surface)
                gmsh.model.mesh.setTransfiniteSurface(surf_rock, cornerTags=[pB[0], pB[-1], pt_rock_br, pt_rock_bl])
                gmsh.model.mesh.setRecombine(2, surf_rock)

    if not water_interface:
        if padding_type is None:
            rectangle_tag = gmsh.model.occ.addRectangle(0, 0, 0, length_x, depth_z)
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(2, [rectangle_tag], name="SubSurface")

            boundary_curves = [
                tag for dim, tag in gmsh.model.getBoundary(
                    [(2, rectangle_tag)],
                    oriented=False,
                    recursive=False,
                )
                if dim == 1
            ]
            tol = 1.0e-6 * max(1.0, abs(length_x), abs(depth_z))
            left_curves = []
            right_curves = []
            bottom_curves = []
            top_curves = []

            for curve in boundary_curves:
                xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, curve)
                if abs(xmin) <= tol and abs(xmax) <= tol:
                    left_curves.append(curve)
                elif abs(xmin - length_x) <= tol and abs(xmax - length_x) <= tol:
                    right_curves.append(curve)
                elif abs(ymin - depth_z) <= tol and abs(ymax - depth_z) <= tol:
                    bottom_curves.append(curve)
                elif abs(ymin) <= tol and abs(ymax) <= tol:
                    top_curves.append(curve)

            gmsh.model.addPhysicalGroup(1, left_curves, 1, "Left")
            gmsh.model.addPhysicalGroup(1, right_curves, 2, "Right")
            gmsh.model.addPhysicalGroup(1, bottom_curves, 3, "Bottom")
            gmsh.model.addPhysicalGroup(1, top_curves, 4, "Top")

        if padding_type == "rectangular":
            pt_tl = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
            pt_tr = gmsh.model.occ.addPoint(length_x, 0.0, 0.0)
            pt_br = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
            pt_bl = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)

            pt_pad_tl = gmsh.model.occ.addPoint(pad_x_min, 0.0, 0.0)
            pt_pad_bl = gmsh.model.occ.addPoint(pad_x_min, depth_z, 0.0)
            pt_pad_b_left = gmsh.model.occ.addPoint(pad_x_min, pad_z_min, 0.0)
            pt_pad_bc_left = gmsh.model.occ.addPoint(0.0, pad_z_min, 0.0)
            pt_pad_bc_right = gmsh.model.occ.addPoint(length_x, pad_z_min, 0.0)
            pt_pad_b_right = gmsh.model.occ.addPoint(pad_x_max, pad_z_min, 0.0)
            pt_pad_br = gmsh.model.occ.addPoint(pad_x_max, depth_z, 0.0)
            pt_pad_tr = gmsh.model.occ.addPoint(pad_x_max, 0.0, 0.0)

            line_top = gmsh.model.occ.addLine(pt_tl, pt_tr)
            line_right = gmsh.model.occ.addLine(pt_tr, pt_br)
            line_bottom = gmsh.model.occ.addLine(pt_br, pt_bl)
            line_left = gmsh.model.occ.addLine(pt_bl, pt_tl)

            pad_top_left = gmsh.model.occ.addLine(pt_pad_tl, pt_tl)
            pad_left = gmsh.model.occ.addLine(pt_pad_bl, pt_pad_tl)
            pad_bot_left_horiz = gmsh.model.occ.addLine(pt_bl, pt_pad_bl)

            pad_corner_bl_left = gmsh.model.occ.addLine(pt_pad_b_left, pt_pad_bl)
            pad_corner_bl_bot = gmsh.model.occ.addLine(pt_pad_bc_left, pt_pad_b_left)
            pad_bot_left_vert = gmsh.model.occ.addLine(pt_bl, pt_pad_bc_left)

            pad_bot_mid = gmsh.model.occ.addLine(pt_pad_bc_right, pt_pad_bc_left)
            pad_bot_right_vert = gmsh.model.occ.addLine(pt_br, pt_pad_bc_right)

            pad_corner_br_bot = gmsh.model.occ.addLine(pt_pad_b_right, pt_pad_bc_right)
            pad_corner_br_right = gmsh.model.occ.addLine(pt_pad_br, pt_pad_b_right)

            pad_bot_right_horiz = gmsh.model.occ.addLine(pt_pad_br, pt_br)
            pad_right = gmsh.model.occ.addLine(pt_pad_tr, pt_pad_br)
            pad_top_right = gmsh.model.occ.addLine(pt_tr, pt_pad_tr)

            loop_internal = gmsh.model.occ.addCurveLoop([line_top, line_right, line_bottom, line_left])
            surf_internal = gmsh.model.occ.addPlaneSurface([loop_internal])
            loop_pad_left = gmsh.model.occ.addCurveLoop([pad_top_left, -line_left, -pad_bot_left_horiz, pad_left])
            surf_pad_left = gmsh.model.occ.addPlaneSurface([loop_pad_left])
            loop_pad_bl = gmsh.model.occ.addCurveLoop([pad_bot_left_horiz, -pad_corner_bl_left, -pad_corner_bl_bot, -pad_bot_left_vert])
            surf_pad_bl = gmsh.model.occ.addPlaneSurface([loop_pad_bl])
            loop_pad_bot = gmsh.model.occ.addCurveLoop([pad_bot_left_vert, -pad_bot_mid, -pad_bot_right_vert, line_bottom])
            surf_pad_bot = gmsh.model.occ.addPlaneSurface([loop_pad_bot])
            loop_pad_br = gmsh.model.occ.addCurveLoop([pad_bot_right_vert, -pad_corner_br_bot, -pad_corner_br_right, -pad_bot_right_horiz])
            surf_pad_br = gmsh.model.occ.addPlaneSurface([loop_pad_br])
            loop_pad_right = gmsh.model.occ.addCurveLoop([line_right, -pad_bot_right_horiz, -pad_right, -pad_top_right])
            surf_pad_right = gmsh.model.occ.addPlaneSurface([loop_pad_right])

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [surf_internal], name="SubSurface")
            gmsh.model.addPhysicalGroup(2, [surf_pad_left, surf_pad_bl, surf_pad_bot, surf_pad_br, surf_pad_right], name="Padding")

            gmsh.model.addPhysicalGroup(1, [pad_top_left], 1, "PaddingTopLeft")
            gmsh.model.addPhysicalGroup(1, [pad_left], 2, "PaddingLeftUpper")
            gmsh.model.addPhysicalGroup(1, [pad_corner_bl_left], 3, "PaddingLeftLower")
            gmsh.model.addPhysicalGroup(1, [pad_corner_bl_bot], 4, "PaddingBottomLeft")
            gmsh.model.addPhysicalGroup(1, [pad_bot_mid], 5, "PaddingBottomCenter")
            gmsh.model.addPhysicalGroup(1, [pad_corner_br_bot], 6, "PaddingBottomRight")
            gmsh.model.addPhysicalGroup(1, [pad_corner_br_right], 7, "PaddingRightLower")
            gmsh.model.addPhysicalGroup(1, [pad_right], 8, "PaddingRightUpper")
            gmsh.model.addPhysicalGroup(1, [pad_top_right], 9, "PaddingTopRight")

            gmsh.model.addPhysicalGroup(1, [line_top], 10, "InternalTop")
            gmsh.model.addPhysicalGroup(1, [line_left], 11, "InternalLeft")
            gmsh.model.addPhysicalGroup(1, [line_right], 12, "InternalRight")
            gmsh.model.addPhysicalGroup(1, [line_bottom], 13, "InternalBottom")

        if padding_type == "hyperelliptical":

            pt_tl = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
            pt_tr = gmsh.model.occ.addPoint(length_x, 0.0, 0.0)
            pt_br = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
            pt_bl = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)

            x_mid_L, x_mid_R = length_x * 0.25, length_x * 0.75
            pt_bot_midL = gmsh.model.occ.addPoint(x_mid_L, depth_z, 0.0)
            pt_bot_midR = gmsh.model.occ.addPoint(x_mid_R, depth_z, 0.0)

            x_O_TL, z_O_TL = intersect(0.0, 0.0, -1, 0, xc, zc, a_val, b_val, hyper_n)
            x_O_BL, z_O_BL = intersect(0.0, depth_z, -1, -1, xc, zc, a_val, b_val, hyper_n)
            x_O_BML, z_O_BML = intersect(x_mid_L, depth_z, 0, -1, xc, zc, a_val, b_val, hyper_n)
            x_O_BMR, z_O_BMR = intersect(x_mid_R, depth_z, 0, -1, xc, zc, a_val, b_val, hyper_n)
            x_O_BR, z_O_BR = intersect(length_x, depth_z, 1, -1, xc, zc, a_val, b_val, hyper_n)
            x_O_TR, z_O_TR = intersect(length_x, 0.0, 1, 0, xc, zc, a_val, b_val, hyper_n)

            pt_O_TL = gmsh.model.occ.addPoint(x_O_TL, z_O_TL, 0.0)
            pt_O_BL = gmsh.model.occ.addPoint(x_O_BL, z_O_BL, 0.0)
            pt_O_BML = gmsh.model.occ.addPoint(x_O_BML, z_O_BML, 0.0)
            pt_O_BMR = gmsh.model.occ.addPoint(x_O_BMR, z_O_BMR, 0.0)
            pt_O_BR = gmsh.model.occ.addPoint(x_O_BR, z_O_BR, 0.0)
            pt_O_TR = gmsh.model.occ.addPoint(x_O_TR, z_O_TR, 0.0)

            line_top = gmsh.model.occ.addLine(pt_tl, pt_tr)
            line_right = gmsh.model.occ.addLine(pt_tr, pt_br)
            line_bot_right = gmsh.model.occ.addLine(pt_br, pt_bot_midR)
            line_bot_mid = gmsh.model.occ.addLine(pt_bot_midR, pt_bot_midL)
            line_bot_left = gmsh.model.occ.addLine(pt_bot_midL, pt_bl)
            line_left = gmsh.model.occ.addLine(pt_bl, pt_tl)

            ray_TL = gmsh.model.occ.addLine(pt_tl, pt_O_TL)
            ray_BL = gmsh.model.occ.addLine(pt_bl, pt_O_BL)
            ray_BML = gmsh.model.occ.addLine(pt_bot_midL, pt_O_BML)
            ray_BMR = gmsh.model.occ.addLine(pt_bot_midR, pt_O_BMR)
            ray_BR = gmsh.model.occ.addLine(pt_br, pt_O_BR)
            ray_TR = gmsh.model.occ.addLine(pt_tr, pt_O_TR)

            arc_TL_BL = make_arc(pt_O_TL, pt_O_BL, x_O_TL, z_O_TL, x_O_BL, z_O_BL, xc, zc, a_val, b_val, hyper_n)
            arc_BL_BML = make_arc(pt_O_BL, pt_O_BML, x_O_BL, z_O_BL, x_O_BML, z_O_BML, xc, zc, a_val, b_val, hyper_n)
            arc_BML_BMR = make_arc(pt_O_BML, pt_O_BMR, x_O_BML, z_O_BML, x_O_BMR, z_O_BMR, xc, zc, a_val, b_val, hyper_n)
            arc_BMR_BR = make_arc(pt_O_BMR, pt_O_BR, x_O_BMR, z_O_BMR, x_O_BR, z_O_BR, xc, zc, a_val, b_val, hyper_n)
            arc_BR_TR = make_arc(pt_O_BR, pt_O_TR, x_O_BR, z_O_BR, x_O_TR, z_O_TR, xc, zc, a_val, b_val, hyper_n)

            surf_rock = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_top, line_right, line_bot_right, line_bot_mid, line_bot_left, line_left])])
            surf_pad_left = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([ray_BL, -arc_TL_BL, -ray_TL, -line_left])])
            surf_pad_bot_left = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_bot_left, ray_BL, arc_BL_BML, -ray_BML])])
            surf_pad_bot_mid = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_bot_mid, ray_BML, arc_BML_BMR, -ray_BMR])])
            surf_pad_bot_right = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_bot_right, ray_BMR, arc_BMR_BR, -ray_BR])])
            surf_pad_right = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_right, ray_BR, arc_BR_TR, -ray_TR])])

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [surf_rock], name="SubSurface")
            gmsh.model.addPhysicalGroup(2, [surf_pad_left, surf_pad_bot_left, surf_pad_bot_mid, surf_pad_bot_right, surf_pad_right], name="Padding")

            gmsh.model.addPhysicalGroup(1, [ray_TL], 1, "PaddingTopLeft")
            gmsh.model.addPhysicalGroup(1, [ray_TR], 2, "PaddingTopRight")

            gmsh.model.addPhysicalGroup(1, [arc_TL_BL], 3, "HyperellipseLeft")
            gmsh.model.addPhysicalGroup(1, [arc_BL_BML], 4, "HyperellipseBottomLeft")
            gmsh.model.addPhysicalGroup(1, [arc_BML_BMR], 5, "HyperellipseBottomCenter")
            gmsh.model.addPhysicalGroup(1, [arc_BMR_BR], 6, "HyperellipseBottomRight")
            gmsh.model.addPhysicalGroup(1, [arc_BR_TR], 7, "HyperellipseRight")

            gmsh.model.addPhysicalGroup(1, [line_top], 8, "InternalTop")
            gmsh.model.addPhysicalGroup(1, [line_left], 9, "InternalLeft")
            gmsh.model.addPhysicalGroup(1, [line_right], 10, "InternalRight")
            gmsh.model.addPhysicalGroup(
                1,
                [line_bot_left, line_bot_mid, line_bot_right],
                11,
                "InternalBottom",
            )

            if structured_mesh:
                len_radial = max(padding_x, padding_z)
                len_x_left, len_x_mid, len_x_right = length_x * 0.25, length_x * 0.50, length_x * 0.25
                len_z_total = abs(depth_z)

                N_radial = max(2, int(np.ceil(len_radial / minElementSize)) + 1)
                N_X_left = max(2, int(np.ceil(len_x_left / minElementSize)) + 1)
                N_X_mid = max(2, int(np.ceil(len_x_mid / minElementSize)) + 1)
                N_X_right = max(2, int(np.ceil(len_x_right / minElementSize)) + 1)
                N_X_total = N_X_left + N_X_mid + N_X_right - 2
                N_Z = max(2, int(np.ceil(len_z_total / minElementSize)) + 1)

                for curve in [ray_TL, ray_BL, ray_BML, ray_BMR, ray_BR, ray_TR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_radial)

                for curve in [line_left, line_right, arc_TL_BL, arc_BR_TR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_Z)

                for curve in [line_bot_left, arc_BL_BML]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_X_left)
                for curve in [line_bot_mid, arc_BML_BMR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_X_mid)
                for curve in [line_bot_right, arc_BMR_BR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_X_right)

                gmsh.model.mesh.setTransfiniteCurve(line_top, N_X_total)

                gmsh.model.mesh.setTransfiniteSurface(surf_pad_left, cornerTags=[pt_bl, pt_O_BL, pt_O_TL, pt_tl])
                gmsh.model.mesh.setTransfiniteSurface(surf_pad_bot_left, cornerTags=[pt_bot_midL, pt_bl, pt_O_BL, pt_O_BML])
                gmsh.model.mesh.setTransfiniteSurface(surf_pad_bot_mid, cornerTags=[pt_bot_midR, pt_bot_midL, pt_O_BML, pt_O_BMR])
                gmsh.model.mesh.setTransfiniteSurface(surf_pad_bot_right, cornerTags=[pt_br, pt_bot_midR, pt_O_BMR, pt_O_BR])
                gmsh.model.mesh.setTransfiniteSurface(surf_pad_right, cornerTags=[pt_tr, pt_br, pt_O_BR, pt_O_TR])

                padding_surfs = [surf_pad_left, surf_pad_bot_left, surf_pad_bot_mid, surf_pad_bot_right, surf_pad_right]
                for surf in padding_surfs:
                    gmsh.model.mesh.setRecombine(2, surf)

                gmsh.model.mesh.setTransfiniteSurface(surf_rock, cornerTags=[pt_tl, pt_tr, pt_br, pt_bl])
                gmsh.model.mesh.setRecombine(2, surf_rock)

    return {
        "z_water_L": z_water_L, "z_water_R": z_water_R,
        "a_val": a_val, "b_val": b_val, "xc": xc, "zc": zc,
        "pad_x_min": pad_x_min, "pad_x_max": pad_x_max, "pad_z_min": pad_z_min,

        # Surfaces
        "water_surface": locals().get("water_surface"),
        "surf_pad_TL": locals().get("surf_pad_TL"),
        "surf_pad_TR": locals().get("surf_pad_TR"),
        "surf_pad_left": locals().get("surf_pad_left"),

        # Arcs
        "arc_WL_ML": locals().get("arc_WL_ML"),
        "arc_ML_BL45": locals().get("arc_ML_BL45"),
        "arc_BL45_BML": locals().get("arc_BL45_BML"),
        "arc_BML_BMR": locals().get("arc_BML_BMR"),
        "arc_BMR_BR45": locals().get("arc_BMR_BR45"),
        "arc_BR45_MR": locals().get("arc_BR45_MR"),
        "arc_MR_WR": locals().get("arc_MR_WR"),
        "arc_TL_BL": locals().get("arc_TL_BL"),
        "arc_BL_BML": locals().get("arc_BL_BML"),
        "arc_BMR_BR": locals().get("arc_BMR_BR"),
        "arc_BR_TR": locals().get("arc_BR_TR"),

        # Lines / Rock Boundaries
        "rock_L_lower": locals().get("rock_L_lower"),
        "rock_L_upper": locals().get("rock_L_upper"),
        "rock_R_lower": locals().get("rock_R_lower"),
        "rock_R_upper": locals().get("rock_R_upper"),
        "rock_B_left": locals().get("rock_B_left"),
        "rock_B_mid": locals().get("rock_B_mid"),
        "rock_B_right": locals().get("rock_B_right"),
        "line_left": locals().get("line_left"),
        "line_right": locals().get("line_right"),
        "line_bot_left": locals().get("line_bot_left"),
        "line_bot_mid": locals().get("line_bot_mid"),
        "line_bot_right": locals().get("line_bot_right"),

        # Rays
        "ray_BL_45": locals().get("ray_BL_45"),
        "ray_BR_45": locals().get("ray_BR_45"),
        "ray_TL": locals().get("ray_TL"),
        "ray_BL": locals().get("ray_BL"),
        "ray_BR": locals().get("ray_BR"),
        "ray_TR": locals().get("ray_TR"),
        "x_O_TL": locals().get("x_O_TL"),
        "x_O_TR": locals().get("x_O_TR")
    }
