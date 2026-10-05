import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter
from ..Io_utils.binary_io import _read_velocity_binary3D


def apply_savitzky_golay_filter_3d(
    grid_values,
    window_length_x=501,
    window_length_z=501,
    window_length_y=501,
    polyorder=3,
):
    """Extend the existing 2-D grading filter to a ``(z, x, y)`` cube.

    Filtering uses ``savgol_filter`` with ``mode="nearest"`` in z, then x,
    then y order. Window validation and negative-value replacement follow
    :func:`apply_savitzky_golay_filter_2d`. A ``None`` window skips only its
    own axis, allowing z/x grading on thin cubes. The sizing constructor
    chooses ``None`` independently for dimensions too small to support an
    odd window greater than the polynomial order within its 90% size cap.

    Parameters
    ----------
    grid_values : array-like
        Three-dimensional mesh element sizes in ``(z, x, y)`` order.
    window_length_x, window_length_z, window_length_y : int or None
        Odd filter windows greater than ``polyorder``. An explicit window
        must fit its axis; ``None`` disables that axis's filtering.
    polyorder : int, optional
        Savitzky-Golay polynomial order. Default is 3, as in the 2-D filter.

    Returns
    -------
    numpy.ndarray
        Filtered sizing cube. If grading produces a negative size, all
        values below the original minimum positive size are replaced by
        that minimum, matching the existing 2-D filter.
    """
    grid_values = np.asarray(grid_values, dtype=np.float64)

    if grid_values.ndim != 3:
        raise ValueError(
            f"grid_values must be a 3D array, received shape "
            f"{grid_values.shape}."
        )
    if not np.all(np.isfinite(grid_values)):
        raise ValueError(
            "Sizing field contains NaN or Inf values before filtering."
        )

    windows = (window_length_z, window_length_x, window_length_y)
    if all(window is None for window in windows):
        return grid_values.copy()

    for axis, (name, window) in enumerate(zip(("z", "x", "y"), windows)):
        if window is None:
            continue
        if window <= 0:
            raise ValueError(
                "Savitzky-Golay window lengths must be positive. "
                f"Got {name}={window}."
            )
        if window % 2 == 0:
            raise ValueError(
                "Savitzky-Golay window lengths must be odd. "
                f"Got {name}={window}."
            )
        if window > grid_values.shape[axis]:
            raise ValueError(
                f"window_length_{name}={window} exceeds {name}-axis size "
                f"{grid_values.shape[axis]}."
            )
        if window <= polyorder:
            raise ValueError(
                "Savitzky-Golay window lengths must be greater than polyorder. "
                f"Got {name}={window}, polyorder={polyorder}."
            )

    positive_values = grid_values[grid_values > 0.0]
    if positive_values.size == 0:
        raise ValueError(
            "Sizing field contains no positive values, so the minimum "
            "element size cannot be determined."
        )
    min_element_size = float(np.min(positive_values))

    filtered_values = grid_values.copy()
    for axis, window in enumerate(windows):
        if window is not None:
            filtered_values = savgol_filter(
                filtered_values,
                window,
                polyorder,
                axis=axis,
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


def _calculate_savgol_window(dim_size, grade_factor, polyorder=3):
    """Use the existing 2-D grade/window rule independently for one axis."""
    if not 0.0 < grade_factor <= 1.0:
        raise ValueError(
            f"grade must be in the interval (0, 1], got {grade_factor}."
        )

    min_valid = polyorder + 2 if polyorder % 2 != 0 else polyorder + 1
    if dim_size < min_valid:
        return None

    requested_window = int(round(grade_factor * dim_size))
    max_window = int(np.floor(0.90 * dim_size))
    if max_window % 2 == 0:
        max_window -= 1
    if max_window < min_valid:
        return None

    window = max(min_valid, min(requested_window, max_window))
    if window % 2 == 0:
        window -= 1
    return window


def sizing_function_xyz(
    X,
    Y,
    Z,
    ef_segy,
    length_x,
    length_y,
    depth_z,
):
    """Evaluate structured Winslow sizing using nearest-edge extension.

    Parameters
    ----------
    X : numpy.ndarray
        X coordinates of the structured mesh nodes.
    Y : numpy.ndarray
        Y coordinates of the structured mesh nodes.
    Z : numpy.ndarray
        Z coordinates of the structured mesh nodes.
    ef_segy : callable
        Mesh-sizing function evaluated in ``(z, x, y)`` coordinates.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth.

    Returns
    -------
    numpy.ndarray
        Positive sizing values with the same shape as the input coordinate arrays.

    Notes
    -----
    Queries outside the physical model are projected to the nearest model edge before
    evaluating the sizing field.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError(
            "Winslow sizing coordinates X, Y and Z must have matching shapes."
        )

    finite_coordinates = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    if not np.all(finite_coordinates):
        bad_count = int(
            finite_coordinates.size - np.count_nonzero(finite_coordinates)
        )
        raise FloatingPointError(
            "Winslow generated NaN or infinite coordinates before "
            f"sizing evaluation ({bad_count} invalid nodes). Reduce "
            "winslow_omega or inspect the structured topology."
        )

    X_edge = np.clip(X.reshape(-1), 0.0, length_x)
    Y_edge = np.clip(Y.reshape(-1), 0.0, length_y)
    Z_edge = np.clip(Z.reshape(-1), -abs(depth_z), 0.0)

    queries_zxy = np.column_stack((Z_edge, X_edge, Y_edge))

    sizes = np.asarray(
        ef_segy(queries_zxy),
        dtype=float,
    ).reshape(X.shape)

    if not np.all(np.isfinite(sizes)):
        invalid = np.flatnonzero(~np.isfinite(sizes))
        first = int(invalid[0])
        raise ValueError(
            "The structured edge-extended sizing function returned "
            "NaN or infinity. First projected query "
            f"(z, x, y)={queries_zxy[first].tolist()}. Check the "
            "velocity binary metadata and sizing-function construction."
        )

    if np.any(sizes <= 0.0):
        minimum = float(np.min(sizes))
        raise ValueError(
            "The structured Winslow sizing function must be positive; "
            f"minimum value is {minimum}."
        )

    return sizes


def create_sizing_function3D(
    fname,
    hmin,
    bbox,
    wl,
    freq,
    pad_type=None,
    pad_size_x=0.0,
    pad_size_y=0.0,
    pad_size_z=0.0,
    grade=0.15,
    vp_water=None,
    nz=None,
    nx=None,
    ny=None,
    byte_order="big",
    axes_order=(0, 1, 2),
    axes_order_sort="F",
    dtype="float32",
):
    """Create a graded wavelength-based sizing function from a binary model.

    The sizing baseline is ``velocity / (freq * wl)``. Grading extends the
    existing 2-D Savitzky-Golay filter with a third y pass, independently
    skipping any axis too small for the cubic filter's 90% window cap.
    In particular, an unpadded y axis of five or fewer samples does not
    disable grading in z or x. Input z rows run from the top downward.
    Numeric construction parameters and bounds must be finite; ``wl`` and
    ``freq`` must also be positive. Non-positive ``hmin`` values retain
    their existing meaning of disabling the requested minimum.

    Parameters
    ----------
    fname : str or pathlib.Path
        Path to the binary velocity model.
    hmin : float or None
        Requested minimum element size.
    bbox : sequence of float
        Bounding box ordered as ``(zmin, zmax, xmin, xmax, ymin, ymax)``.
    wl : float
        Number of mesh points per wavelength.
    freq : float
        Reference frequency used to convert velocity to element size.
    pad_type : str or None
        ``None``, ``"rectangular"``, or ``"hyperelliptical"``. Both padding
        geometries edge-extend the sizing cube before grading: bottom z
        padding and symmetric x/y padding, with the top left unpadded.
    pad_size_x : float
        Padding extent in x used by the sizing function.
    pad_size_y : float
        Padding extent in y used by the sizing function.
    pad_size_z : float
        Padding extent in z used by the sizing function.
    grade : float or None
        Fraction of each padded axis size used to choose its odd grading
        window, capped at 90% of that size. Must be finite in ``[0, 1]``;
        ``None`` or zero disables grading. This is the same window rule
        used by the existing 2-D sizing function.
    vp_water : float or None
        Velocity assigned to water or zero-valued cells.
    nz : int
        Number of velocity samples in the z direction.
    nx : int
        Number of velocity samples in the x direction.
    ny : int
        Number of velocity samples in the y direction.
    byte_order : {"big", "little"}
        Byte order of the velocity-model binary file.
    axes_order : tuple of int
        Permutation mapping binary axes to Spyro ``(z, x, y)`` order.
    axes_order_sort : {"C", "F"}
        Memory order used to reshape the binary velocity data.
    dtype : str or numpy.dtype
        Numeric data type stored in the velocity-model file.

    Returns
    -------
    tuple
        ``(sizing_callable, actual_minimum, actual_maximum, nz, nx, ny)``.
        Dimensions describe the original binary cube, before padding.
        The callable takes a single ``(z, x, y)`` point or an ``(N, 3)``
        array. Queries outside the padded bounds use nearest-edge values.
    """
    if any(value is None for value in (nz, nx, ny)):
        raise ValueError("3-D sizing requires nz, nx and ny.")
    try:
        wl = float(wl)
    except (TypeError, ValueError) as exc:
        raise ValueError("wl must be finite and positive.") from exc
    if not np.isfinite(wl) or wl <= 0.0:
        raise ValueError("wl must be finite and positive.")
    try:
        freq = float(freq)
    except (TypeError, ValueError) as exc:
        raise ValueError("freq must be finite and positive.") from exc
    if not np.isfinite(freq) or freq <= 0.0:
        raise ValueError("freq must be finite and positive.")

    if grade is not None:
        grade = float(grade)
        if not np.isfinite(grade) or not 0.0 <= grade <= 1.0:
            raise ValueError(
                f"grade must be finite and in the interval [0, 1], got {grade}."
            )

    try:
        zmin, zmax, xmin, xmax, ymin, ymax = map(float, bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox must contain six finite coordinates.") from exc
    if not np.all(np.isfinite((zmin, zmax, xmin, xmax, ymin, ymax))):
        raise ValueError("bbox must contain six finite coordinates.")
    if not (zmin < zmax and xmin < xmax and ymin < ymax):
        raise ValueError(
            "bbox must be ordered as "
            "(zmin, zmax, xmin, xmax, ymin, ymax)."
        )

    requested_hmin = None
    if hmin is not None:
        try:
            hmin = float(hmin)
        except (TypeError, ValueError) as exc:
            raise ValueError("hmin must be finite when provided.") from exc
        if not np.isfinite(hmin):
            raise ValueError("hmin must be finite when provided.")
        if hmin > 0.0:
            requested_hmin = hmin

    if vp_water is not None:
        try:
            vp_water = float(vp_water)
        except (TypeError, ValueError) as exc:
            raise ValueError("vp_water must be finite when provided.") from exc
        if not np.isfinite(vp_water):
            raise ValueError("vp_water must be finite when provided.")

    if vp_water is not None and float(vp_water) > 0.0:
        physical_positive_floor = (
            float(vp_water) / (float(freq) * float(wl))
        )
    elif requested_hmin is not None:
        physical_positive_floor = requested_hmin
    else:
        physical_positive_floor = np.finfo(np.float64).eps

    if requested_hmin is not None:
        physical_positive_floor = max(
            physical_positive_floor,
            requested_hmin,
        )

    def edge_extended_callable(base_callable):
        """Wrap a core interpolator with nearest-edge extension outside the model bounds.

        Parameters
        ----------
        base_callable : callable
            Core sizing interpolator defined inside the velocity-model bounds.

        Returns
        -------
        callable
            Sizing function that clamps queries to the velocity-model bounds.
        """
        def evaluate(coordinates):
            """Evaluate the edge-extended sizing function at one or more coordinates.

            Parameters
            ----------
            coordinates : numpy.ndarray
                Coordinates with shape ``(N, 3)`` in ``(z, x, y)`` order.

            Returns
            -------
            float or numpy.ndarray
                Positive sizing values corresponding to the supplied coordinates.
            """
            points = np.asarray(coordinates, dtype=np.float64)
            scalar_input = points.ndim == 1

            if scalar_input:
                points = points.reshape(1, 3)

            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(
                    "3-D sizing coordinates must have shape (N, 3) "
                    "in (z, x, y) order."
                )
            if not np.all(np.isfinite(points)):
                raise ValueError(
                    "3-D sizing coordinates contain NaN or infinity."
                )

            projected = points.copy()
            projected[:, 0] = np.clip(
                projected[:, 0], zmin, zmax
            )
            projected[:, 1] = np.clip(
                projected[:, 1], xmin, xmax
            )
            projected[:, 2] = np.clip(
                projected[:, 2], ymin, ymax
            )

            values = np.asarray(
                base_callable(projected),
                dtype=np.float64,
            ).reshape(-1)

            if values.size != projected.shape[0]:
                raise ValueError(
                    "The 3-D sizing callable returned an unexpected "
                    f"number of values: {values.size} for "
                    f"{projected.shape[0]} coordinates."
                )
            if not np.all(np.isfinite(values)):
                first = int(
                    np.flatnonzero(~np.isfinite(values))[0]
                )
                raise ValueError(
                    "The 3-D sizing function returned NaN or infinity "
                    "inside the velocity-model box. First projected "
                    f"coordinate (z, x, y)="
                    f"{projected[first].tolist()}."
                )
            if np.any(values < 0.0):
                first = int(np.flatnonzero(values < 0.0)[0])
                raise ValueError(
                    "The 3-D sizing function returned a negative size "
                    f"{values[first]} at projected coordinate "
                    f"(z, x, y)={projected[first].tolist()}."
                )

            values = np.maximum(
                values,
                physical_positive_floor,
            )

            if scalar_input:
                return values[0]
            return values

        return evaluate

    velocity = _read_velocity_binary3D(
        fname=fname,
        nz=nz,
        nx=nx,
        ny=ny,
        byte_order=byte_order,
        axes_order=axes_order,
        axes_order_sort=axes_order_sort,
        dtype=dtype,
    )

    if vp_water is not None:
        velocity = np.where(
            velocity == 0.0,
            float(vp_water),
            velocity,
        )

    if np.any(~np.isfinite(velocity)):
        raise ValueError(
            "The velocity model contains NaN or infinity."
        )
    if np.any(velocity <= 0.0):
        raise ValueError(
            "The velocity model contains non-positive values. Set vp_water "
            "or preprocess the model before constructing the sizing field."
        )

    sizes = velocity / (float(freq) * float(wl))
    if requested_hmin is not None:
        sizes = np.maximum(sizes, requested_hmin)

    sizes = np.maximum(sizes, physical_positive_floor)

    if pad_type in ("rectangular", "hyperelliptical"):
        pad_size_x, pad_size_y, pad_size_z = map(
            float, (pad_size_x, pad_size_y, pad_size_z)
        )
        if any(
            not np.isfinite(value) or value < 0.0
            for value in (pad_size_x, pad_size_y, pad_size_z)
        ):
            raise ValueError("Padding extents must be finite and non-negative.")

        def padding_samples(extent, minimum, maximum, count, axis_name):
            if extent == 0.0:
                return 0
            if int(count) < 2:
                raise ValueError(
                    f"Padding in {axis_name} requires at least 2 model samples "
                    f"on that axis; got {count}."
                )
            spacing = (maximum - minimum) / (int(count) - 1)
            return int(extent / spacing)

        nnz = padding_samples(pad_size_z, zmin, zmax, nz, "z")
        nnx = padding_samples(pad_size_x, xmin, xmax, nx, "x")
        nny = padding_samples(pad_size_y, ymin, ymax, ny, "y")
        sizes = np.pad(sizes, ((0, nnz), (nnx, nnx), (nny, nny)), "edge")
        zmin -= pad_size_z
        xmin -= pad_size_x
        xmax += pad_size_x
        ymin -= pad_size_y
        ymax += pad_size_y

    if grade is not None and grade > 0.0:
        polyorder = 3
        sizes = apply_savitzky_golay_filter_3d(
            sizes,
            window_length_z=_calculate_savgol_window(sizes.shape[0], grade),
            window_length_x=_calculate_savgol_window(sizes.shape[1], grade),
            window_length_y=_calculate_savgol_window(sizes.shape[2], grade),
            polyorder=polyorder,
        )

    # Cubic grading can undershoot a positive field without becoming
    # negative. Retain the requested/physical minimum after grading.
    sizes = np.maximum(sizes, physical_positive_floor)
    if not np.all(np.isfinite(sizes)) or np.any(sizes <= 0.0):
        raise ValueError("The constructed sizing field must be finite and positive.")

    # Match the 2-D interpolation convention: the requested padded bounds
    # set the endpoints even when padding is not an integer sample spacing.
    z_axis = np.linspace(zmax, zmin, sizes.shape[0])[::-1]
    x_axis = np.linspace(xmin, xmax, sizes.shape[1])
    y_axis = np.linspace(ymin, ymax, sizes.shape[2])
    sizes_for_interpolation = sizes[::-1, :, :]

    interpolator = RegularGridInterpolator(
        (z_axis, x_axis, y_axis),
        sizes_for_interpolation,
        method="linear",
        # An axis with one sample represents a constant extrusion along
        # that direction. Explicit outer-bound clamping remains below.
        bounds_error=False,
        fill_value=None,
    )

    ef = edge_extended_callable(interpolator)

    return (
        ef,
        float(np.min(sizes)),
        float(np.max(sizes)),
        int(nz),
        int(nx),
        int(ny),
    )
