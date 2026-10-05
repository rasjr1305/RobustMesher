import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator
from ..Io_utils.meshing_readers import read_segy_velocity_model


def calculate_edge_length(cpw, minimum_velocity, frequency):
    """
    Calculate the edge length for mesh generation.

    Parameters
    ----------
    cpw : float
        Cells per wavelength.
    minimum_velocity : float
        Minimum velocity in the domain.
    frequency : float
        Source frequency.

    Returns
    -------
    edge_length : float
        Calculated edge length for mesh elements.
    """
    if cpw == 0.0 or cpw is None:
        raise ValueError("cpw value of {cpw} invalid for edge length calculation.")
    v_min = minimum_velocity

    lbda_min = v_min / frequency

    edge_length = lbda_min / cpw
    return edge_length


def vp_to_sizing(vp, cpw, frequency):
    """
    Convert velocity field to mesh sizing function.

    Parameters
    ----------
    vp : numpy.ndarray
        P-wave velocity field.
    cpw : float
        Cells per wavelength(must be positive).
    frequency : float
        Source frequency in Hz(must be positive).

    Returns
    -------
    sizing : numpy.ndarray
        Mesh element sizes corresponding to the velocity field.

    Raises
    ------
    ValueError
        If cpw or frequency is not positive.

    Notes
    -----
    The mesh size is calculated as: `size = vp / (frequency * cpw)`
    This ensures that the mesh has the specified number of cells per
    wavelength throughout the domain.
    """
    if cpw < 0.0 or cpw == 0.0:
        raise ValueError(f"Cells-per-wavelength value of {cpw} not supported.")
    if frequency < 0.0 or frequency == 0.0:
        raise ValueError(f"Frequency must be positive and non zero, not {frequency}")

    return vp / (frequency * cpw)


def create_sizing_function(
    fname,
    hmin=None,
    bbox=None,
    wl=3,
    freq=5,
    pad_type=None,
    pad_size_x=-1.0,
    pad_size_z=-1.0,
    grade=None,
    vp_water=None,
):
    """Create a mesh sizing function from a SEGY velocity model.

    This function reads a SEGY file, extracts the velocity model, applies
    optional water velocity substitution, and calculates an appropriate
    element size field based on wavelength constraints and padding.

    Parameters
    ----------
    fname : str
        Filename of the SEGY velocity model.
    hmin : float, optional
        Minimum element size allowed. Element sizes below this will be clipped.
    bbox : tuple
        Bounding box tuple in the format (zmin, zmax, xmin, xmax).
    wl : int, optional
        Number of elements per wavelength. Default is 10.
    freq : float, optional
        Maximum source frequency in Hz. Default is 2.
    pad_type : str, optional
        Type of padding to apply ('rectangular', 'elliptical', or None).
    pad_size_x : float, optional
        Size of padding in the x-direction. Default is -1.0.
    pad_size_z : float, optional
        Size of padding in the z-direction. Default is -1.0.
    grade : float, optional
        Grading parameter for applying a 2D Savitzky-Golay filter to smooth
        transitions. Default is None.
    vp_water : float, optional
        Velocity value to substitute where the SEGY model has zeros
        (representing water). If None, defaults to 1500.0.

    Returns
    -------
    tuple
        A tuple containing:
        - sizing_function (callable): A function f(x, y) returning element size.
        - min_val (float): Minimum element size in the domain.
        - max_val (float): Maximum element size in the domain.
        - n_samples (int): Number of depth samples per trace (rows).
        - n_traces (int): Number of lateral traces (columns).
    """

    # Read velocity model with provided bbox
    vp, n_samples, n_traces = read_segy_velocity_model(fname)
    # Set water velocity if value = 0
    if vp_water is not None:
        vp = np.where(vp == 0, vp_water, vp)
    else:
        # If no water velocity provided and model contains zeros, raise error
        if np.any(vp == 0):
            raise ValueError(
                "Velocity model contains zero values (water). Provide 'vp_water' "
                "to substitute water velocities or preprocess the SEGY file."
            )
    # Calculate wavelength-based sizing
    cell_size = calculate_wavelength_sizing(vp, wl, freq)
    # Enforce minimum element size
    if hmin is not None:
        cell_size = np.maximum(cell_size, hmin)

    # Applying padding
    if pad_type == "rectangular" or pad_type == "hyperelliptical":
        if n_samples < 2 or n_traces < 2:
            raise ValueError(
                f"SEGY model must have at least 2 samples and 2 traces; "
                f"got n_samples={n_samples}, n_traces={n_traces}."
            )

        dz = (bbox[1] - bbox[0]) / (n_samples - 1)
        dx = (bbox[3] - bbox[2]) / (n_traces - 1)
        nnz = int(pad_size_z / dz)
        nnx = int(pad_size_x / dx)

        padding = ((0, nnz), (nnx, nnx))
        cell_size = np.pad(cell_size, padding, "edge")
        bbox = (
            bbox[0] - pad_size_z,
            bbox[1],
            bbox[2] - pad_size_x,
            bbox[3] + pad_size_x,
        )

    print(cell_size.shape[0])
    print(cell_size.shape[1])
    if grade is not None and grade > 0.0:
        polyorder = 3

        def calculate_window(dim_size, grade_factor, polyorder=3):
            """
            Return a valid odd Savitzky-Golay window length.

            grade_factor:
                0.1 -> small/local smoothing
                0.9 -> high smoothing

            Safety rule:
                The window may not exceed 90% of the corresponding axis size.
            """
            if not 0.0 < grade_factor <= 1.0:
                raise ValueError(
                    f"grade must be in the interval (0, 1], got {grade_factor}."
                )

            # Savitzky-Golay requires window_length > polyorder.
            min_valid = polyorder + 2 if polyorder % 2 != 0 else polyorder + 1

            if dim_size < min_valid:
                return None

            requested_window = int(round(grade_factor * dim_size))

            # Maximum window size
            max_window = int(np.floor(0.90 * dim_size))

            # Savitzky-Golay requires an odd window.
            if max_window % 2 == 0:
                max_window -= 1

            # For very small dimensions, do not violate the 90% safety rule.
            if max_window < min_valid:
                return None

            # Keep the window inside the valid range.
            window = max(min_valid, min(requested_window, max_window))

            # Force odd.
            if window % 2 == 0:
                window -= 1

            return window

        window_length_z = calculate_window(cell_size.shape[0], grade)
        window_length_x = calculate_window(cell_size.shape[1], grade)

        # Apply the filter
        cell_size = apply_savitzky_golay_filter_2d(
            cell_size,
            window_length_x=window_length_x,
            window_length_z=window_length_z,
            polyorder=polyorder
        )

    print("Function Minimum and Maximum values:")
    print(cell_size.min(), cell_size.max())

    # Create interpolation function
    def sizing_function(x, y):
        """Return element size at position (x, y)

        Parameters:
        -----------
        x, y : float or array-like
            Coordinates where to evaluate element size

        Returns:
        --------
        float or array
            Element size at the given position(s)
        """
        return interpolate_size(x, y, cell_size, bbox)

    return sizing_function, cell_size.min(), cell_size.max(), n_samples, n_traces


def calculate_wavelength_sizing(vp, wl, freq):
    """Calculate the target element size based on a wavelength criterion.

    Parameters
    ----------
    vp : ndarray
        2D array representing the velocity model.
    wl : int
        Desired number of elements per wavelength.
    freq : float
        Maximum frequency in Hz for the simulation.

    Returns
    -------
    ndarray
        2D array of the calculated element sizes based on the velocity field.
    """
    # Wavelength = velocity / frequency
    wavelength = vp / freq

    # Element size = wavelength / number of elements per wavelength
    cell_size = wavelength / wl

    return cell_size


def interpolate_size(x, y, cell_size, bbox):
    """Interpolate element size at specific spatial coordinates.

    Uses RegularGridInterpolator to return the interpolated
    element size at a specific value.

    Parameters
    ----------
    x : float or array-like
        x-coordinates (lateral) where to evaluate the element size.
    y : float or array-like
        y-coordinates (depth/z-axis) where to evaluate the element size.
    cell_size : ndarray
        2D array of pre-calculated element sizes.
    bbox : tuple
        Bounding box tuple defining the domain boundaries (zmin, zmax, xmin, xmax).

    Returns
    -------
    float or ndarray
        Interpolated element size(s) at the given spatial positions.
    """
    # Create coordinate arrays
    z_coords = np.linspace(bbox[0], bbox[1], cell_size.shape[0])
    x_coords = np.linspace(bbox[2], bbox[3], cell_size.shape[1])
    cell_size_flipped = np.flipud(cell_size)

    # Create interpolator
    interpolator = RegularGridInterpolator(
        (z_coords, x_coords),
        cell_size_flipped,
        method='linear',
        bounds_error=False,
        fill_value=None
    )

    # Handle scalar or array inputs
    if np.isscalar(x) and np.isscalar(y):
        points = np.array([[y, x]])  # Note: y corresponds to z (depth), x to x
    else:
        x = np.asarray(x)
        y = np.asarray(y)
        points = np.column_stack([y.ravel(), x.ravel()])

    result = interpolator(points)

    # Return scalar if input was scalar
    if np.isscalar(x) and np.isscalar(y):
        return float(result[0])
    else:
        return result.reshape(x.shape)


def apply_savitzky_golay_filter_2d(
    grid_values,
    window_length_x=501,
    window_length_z=501,
    polyorder=3,
):
    """Apply a two-dimensional Savitzky-Golay grading filter.

    Parameters
    ----------
    grid_values : array-like
        Two-dimensional array containing mesh element sizes.
    window_length_x : int, optional
        Odd Savitzky-Golay window length along the lateral x direction
    window_length_z : int, optional
        Odd Savitzky-Golay window length along the depth z direction
    polyorder : int, optional
        Polynomial order used by the Savitzky-Golay filter. Default is 3.

    Returns
    -------
    numpy.ndarray
        Filtered sizing field. If any negative value is created by the
        filter, all values below the original minimum positive element size
        are replaced by that minimum size.

    Raises
    ------
    ValueError
        If the input is invalid, contains NaN/Inf, has no positive values,
        or uses incompatible Savitzky-Golay window parameters.
    """
    grid_values = np.asarray(grid_values, dtype=np.float64)

    if grid_values.ndim != 2:
        raise ValueError(
            f"grid_values must be a 2D array, received shape "
            f"{grid_values.shape}."
        )

    if not np.all(np.isfinite(grid_values)):
        raise ValueError(
            "Sizing field contains NaN or Inf values before filtering."
        )

    if window_length_z is None or window_length_x is None:
        return grid_values.copy()

    if window_length_z <= 0 or window_length_x <= 0:
        raise ValueError(
            "Savitzky-Golay window lengths must be positive. "
            f"Got z={window_length_z}, x={window_length_x}."
        )

    if window_length_z % 2 == 0 or window_length_x % 2 == 0:
        raise ValueError(
            "Savitzky-Golay window lengths must be odd. "
            f"Got z={window_length_z}, x={window_length_x}."
        )

    if window_length_z > grid_values.shape[0]:
        raise ValueError(
            f"window_length_z={window_length_z} exceeds z-axis size "
            f"{grid_values.shape[0]}."
        )

    if window_length_x > grid_values.shape[1]:
        raise ValueError(
            f"window_length_x={window_length_x} exceeds x-axis size "
            f"{grid_values.shape[1]}."
        )

    if window_length_z <= polyorder or window_length_x <= polyorder:
        raise ValueError(
            "Savitzky-Golay window lengths must be greater than polyorder. "
            f"Got z={window_length_z}, x={window_length_x}, "
            f"polyorder={polyorder}."
        )

    positive_values = grid_values[grid_values > 0.0]

    if positive_values.size == 0:
        raise ValueError(
            "Sizing field contains no positive values, so the minimum "
            "element size cannot be determined."
        )

    min_element_size = float(np.min(positive_values))

    filtered_values = savgol_filter(
        grid_values,
        window_length_z,
        polyorder,
        axis=0,
        mode="nearest",
    )

    filtered_values = savgol_filter(
        filtered_values,
        window_length_x,
        polyorder,
        axis=1,
        mode="nearest",
    )

    if not np.all(np.isfinite(filtered_values)):
        raise ValueError(
            "Sizing field contains NaN or Inf values after filtering."
        )

    negative_mask = filtered_values < 0.0
    n_negative = int(np.count_nonzero(negative_mask))

    if n_negative > 0:
        below_min_mask = filtered_values < min_element_size
        n_replaced = int(np.count_nonzero(below_min_mask))

        print(
            f"Warning: Savitzky-Golay grading produced {n_negative} negative "
            f"sizing values. Replacing all {n_replaced} values below "
            f"min_element_size={min_element_size:.6f} with the minimum "
            "element size."
        )

        filtered_values[below_min_mask] = min_element_size

    return filtered_values
