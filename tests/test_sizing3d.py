"""Contract and numerical checks for the NumPy/SciPy-only 3-D sizing field."""
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest
import RobustMesher
from scipy.signal import savgol_filter

from RobustMesher.sizing_utils.sizing2d import apply_savitzky_golay_filter_2d
from RobustMesher.sizing_utils.sizing3d import (
    apply_savitzky_golay_filter_3d,
    create_sizing_function3D,
)


def _write_binary(path, values, byte_order='big', axes_order=(0, 1, 2), order='F', dtype='float32'):
    canonical = np.asarray(values)
    inverse = np.argsort(np.asarray(axes_order))
    on_disk = canonical.transpose(tuple(inverse))
    disk_dtype = np.dtype(dtype).newbyteorder('>' if byte_order == 'big' else '<')
    np.asarray(on_disk, dtype=disk_dtype).ravel(order=order).tofile(path)


def _field(tmp_path, values, bbox=None, grade=0.0, **kwargs):
    values = np.asarray(values)
    nz, nx, ny = values.shape
    if bbox is None:
        bbox = (-float(nz - 1), 0.0, 0.0, float(nx - 1), 0.0, float(ny - 1))
    path = tmp_path / 'model.bin'
    disk = {
        'byte_order': kwargs.pop('byte_order', 'big'),
        'axes_order': kwargs.pop('axes_order', (0, 1, 2)),
        'order': kwargs.pop('axes_order_sort', 'F'),
        'dtype': kwargs.pop('dtype', 'float32'),
    }
    _write_binary(path, values, **disk)
    arguments = dict(fname=path, hmin=None, bbox=bbox, wl=2.0, freq=5.0,
                     grade=grade, nz=nz, nx=nx, ny=ny,
                     byte_order=disk['byte_order'], axes_order=disk['axes_order'],
                     axes_order_sort=disk['order'], dtype=disk['dtype'])
    arguments.update(kwargs)
    return create_sizing_function3D(**arguments)


def _node_queries(shape, bbox):
    nz, nx, ny = shape
    zmin, zmax, xmin, xmax, ymin, ymax = bbox
    z, x, y = np.meshgrid(np.linspace(zmax, zmin, nz),
                          np.linspace(xmin, xmax, nx),
                          np.linspace(ymin, ymax, ny), indexing='ij')
    return np.column_stack((z.ravel(), x.ravel(), y.ravel()))


@pytest.mark.parametrize('grade', [None, 0.0])
@pytest.mark.parametrize('byte_order,axes_order,order,dtype', [
    ('big', (0, 1, 2), 'F', 'float32'),
    ('little', (1, 0, 2), 'C', 'float32'),
    ('big', (2, 0, 1), 'F', 'float64'),
    ('little', (2, 1, 0), 'C', 'float64'),
])
def test_ungraded_wavelength_field_binary_layout_and_orientation(tmp_path, grade, byte_order, axes_order, order, dtype):
    iz, ix, iy = np.indices((3, 4, 5))
    velocity = 1800.0 + 100.0 * iz + 20.0 * ix + 3.0 * iy
    bbox = (-2.0, 0.0, 0.0, 3.0, 0.0, 4.0)
    result = _field(tmp_path, velocity, bbox=bbox, grade=grade,
                    byte_order=byte_order, axes_order=axes_order,
                    axes_order_sort=order, dtype=dtype)
    assert len(result) == 6
    ef, hmin, hmax, nz, nx, ny = result
    assert callable(ef) and (nz, nx, ny) == velocity.shape
    assert hmin == pytest.approx(180.0)
    assert hmax == pytest.approx(float(velocity.max()) / 10.0)
    queries = _node_queries(velocity.shape, bbox)
    np.testing.assert_allclose(ef(queries).reshape(velocity.shape), velocity / 10.0)
    # Distinct coefficients identify all three axes, including top-to-bottom z.
    point = np.array([-0.37, 1.4, 2.6])
    expected = (1800.0 + 100.0 * 0.37 + 20.0 * 1.4 + 3.0 * 2.6) / 10.0
    assert np.isscalar(ef(point))
    assert ef(point) == pytest.approx(expected)
    np.testing.assert_allclose(ef(np.array([[-20.0, -5.0, 50.0], [10.0, 20.0, -2.0]])),
                               [(1800.0 + 200.0 + 12.0) / 10.0,
                                (1800.0 + 60.0) / 10.0])


def test_extruded_grid_matches_original_spyro_2d_filter():
    iz, ix = np.indices((21, 25))
    sizes2d = 150.0 + 10.0 * np.sin(1.3 * iz) + 8.0 * np.cos(0.9 * ix)
    sizes3d = np.repeat(sizes2d[:, :, None], 9, axis=2)
    original = apply_savitzky_golay_filter_2d(
        sizes2d, window_length_z=5, window_length_x=7, polyorder=3)
    actual = apply_savitzky_golay_filter_3d(
        sizes3d, window_length_z=5, window_length_x=7,
        window_length_y=5, polyorder=3)
    assert np.min(original) > 0.0
    np.testing.assert_allclose(actual, np.repeat(original[:, :, None], 9, axis=2),
                               rtol=3e-14, atol=5e-12)


def test_y_only_variation_is_smoothed_by_third_pass(tmp_path):
    profile = np.full(21, 100.0)
    profile[10:] = 200.0
    grid = np.broadcast_to(profile, (3, 3, 21)).copy()
    expected = savgol_filter(grid, 5, 3, axis=2, mode='nearest')
    direct = apply_savitzky_golay_filter_3d(
        grid, window_length_z=None, window_length_x=None, window_length_y=5)
    np.testing.assert_allclose(direct, expected)
    assert np.max(np.abs(direct - grid)) > 1.0
    result = _field(tmp_path, grid * 10.0, grade=0.25)
    queries = _node_queries(grid.shape, (-2.0, 0.0, 0.0, 2.0, 0.0, 20.0))
    np.testing.assert_allclose(result[0](queries).reshape(grid.shape), expected)
    assert result[1] == pytest.approx(float(expected.min()))
    assert result[2] == pytest.approx(float(expected.max()))


@pytest.mark.parametrize("ny", [1, 3])
def test_short_y_axis_does_not_disable_z_and_x_grading(tmp_path, ny):
    iz, ix = np.indices((21, 25))
    base = 150.0 + 10.0 * np.sin(1.3 * iz) + 8.0 * np.cos(0.9 * ix)
    grid = np.repeat(base[:, :, None], ny, axis=2)
    expected2d = apply_savitzky_golay_filter_2d(base, window_length_z=5, window_length_x=5)
    bbox = (-20.0, 0.0, 0.0, 24.0, 0.0, 2.0)
    result = _field(tmp_path, grid * 10.0, bbox=bbox, grade=0.25, dtype='float64')
    actual = result[0](_node_queries(grid.shape, bbox)).reshape(grid.shape)
    np.testing.assert_allclose(actual, np.repeat(expected2d[:, :, None], ny, axis=2),
                               rtol=2e-14, atol=5e-12)
    assert np.max(np.abs(actual - grid)) > 1.0
    assert result[3:] == grid.shape


@pytest.mark.parametrize("pad_type", ["rectangular", "hyperelliptical"])
def test_padding_is_added_before_filter_and_preserves_original_counts(tmp_path, pad_type):
    iz, ix, iy = np.indices((7, 7, 7))
    grid = 150.0 + 5.0 * iz * iz + 3.0 * ix * ix + 2.0 * iy * iy
    padded = np.pad(grid, ((0, 2), (2, 2), (1, 1)), mode='edge')
    expected = apply_savitzky_golay_filter_3d(
        padded, window_length_z=5, window_length_x=5, window_length_y=5)
    result = _field(tmp_path, grid * 10.0, grade=0.5, dtype='float64',
                    pad_type=pad_type, pad_size_z=2.0,
                    pad_size_x=2.0, pad_size_y=1.0)
    extended_bbox = (-8.0, 0.0, -2.0, 8.0, -1.0, 7.0)
    queried = result[0](_node_queries(padded.shape, extended_bbox)).reshape(padded.shape)
    np.testing.assert_allclose(queried, expected, rtol=2e-14, atol=5e-12)
    assert result[3:] == grid.shape
    assert result[1] == pytest.approx(float(expected.min()))
    assert result[2] == pytest.approx(float(expected.max()))
    smoothed_first = apply_savitzky_golay_filter_3d(
        grid, window_length_z=5, window_length_x=5, window_length_y=5)
    assert np.max(np.abs(expected - np.pad(smoothed_first, ((0, 2), (2, 2), (1, 1)), mode='edge'))) > 1.0
    outside = np.array([[-100.0, -100.0, 100.0], [100.0, 100.0, -100.0]])
    corners = np.array([[-8.0, -2.0, 7.0], [0.0, 8.0, -1.0]])
    np.testing.assert_allclose(result[0](outside), result[0](corners))


def test_positive_filter_undershoot_still_respects_requested_hmin(tmp_path):
    grid = np.full((9, 9, 21), 100.0)
    grid[:, :, 10:] = 200.0
    unbounded = savgol_filter(grid, 5, 3, axis=2, mode='nearest')
    assert 0.0 < unbounded.min() < 95.0
    result = _field(tmp_path, grid * 10.0, grade=0.25, hmin=95.0)
    bbox = (-8.0, 0.0, 0.0, 8.0, 0.0, 20.0)
    actual = result[0](_node_queries(grid.shape, bbox))
    assert actual.min() == pytest.approx(95.0)
    assert result[1] == pytest.approx(95.0)
    assert result[2] == pytest.approx(float(actual.max()))


def test_negative_filter_overshoot_is_finite_and_positive(tmp_path):
    grid = np.full((11, 11, 11), 220.0)
    grid[5, 5, 5] = 20000.0
    unbounded = grid.copy()
    for axis in range(3):
        unbounded = savgol_filter(unbounded, 5, 3, axis=axis, mode='nearest')
    assert unbounded.min() < 0.0
    actual = apply_savitzky_golay_filter_3d(
        grid, window_length_z=5, window_length_x=5, window_length_y=5)
    assert np.all(np.isfinite(actual))
    assert actual.min() >= grid.min()
    result = _field(tmp_path, grid * 10.0, grade=0.5, hmin=220.0)
    bbox = (-10.0, 0.0, 0.0, 10.0, 0.0, 10.0)
    values = result[0](_node_queries(grid.shape, bbox))
    assert np.all(np.isfinite(values)) and np.min(values) >= 220.0
    assert result[1] == pytest.approx(float(values.min()))
    assert result[2] == pytest.approx(float(values.max()))


def test_zero_water_substitution_preserves_original_floor(tmp_path):
    velocity = np.full((3, 4, 5), 1800.0)
    velocity[0, 0, 0] = 0.0
    result = _field(tmp_path, velocity, grade=0.0, vp_water=1500.0)
    assert result[0](np.array([0.0, 0.0, 0.0])) == pytest.approx(150.0)
    assert result[1:3] == pytest.approx((150.0, 180.0))


@pytest.mark.parametrize('grade', [-0.1, 1.1, np.nan, np.inf])
def test_rejects_invalid_grading(tmp_path, grade):
    with pytest.raises(ValueError):
        _field(tmp_path, np.full((7, 7, 7), 1800.0), grade=grade)


@pytest.mark.parametrize('name,value', [('wl', 0.0), ('freq', -1.0), ('freq', np.nan), ('wl', np.inf)])
def test_rejects_nonpositive_or_nonfinite_wavelength_inputs(tmp_path, name, value):
    with pytest.raises(ValueError):
        _field(tmp_path, np.full((3, 4, 5), 1800.0), **{name: value})


@pytest.mark.parametrize('bad', [0.0, -1.0, np.nan, np.inf])
def test_rejects_invalid_velocity_samples(tmp_path, bad):
    velocity = np.full((3, 4, 5), 1800.0)
    velocity[1, 2, 3] = bad
    with pytest.raises(ValueError):
        _field(tmp_path, velocity)


@pytest.mark.parametrize('bbox', [
    (0.0, -2.0, 0.0, 3.0, 0.0, 4.0),
    (-2.0, 0.0, 0.0, np.inf, 0.0, 4.0),
    (-2.0, 0.0, 0.0, 3.0, np.nan, 4.0),
])
def test_rejects_invalid_bbox(tmp_path, bbox):
    with pytest.raises(ValueError):
        _field(tmp_path, np.full((3, 4, 5), 1800.0), bbox=bbox)


@pytest.mark.parametrize('query', [np.zeros((2, 2)), np.array([np.nan, 0.0, 0.0]), np.array([[0.0, np.inf, 0.0]])])
def test_callable_rejects_malformed_queries(tmp_path, query):
    result = _field(tmp_path, np.full((3, 4, 5), 1800.0))
    with pytest.raises(ValueError):
        result[0](query)


@pytest.mark.parametrize('windows', [
    dict(window_length_z=4, window_length_x=None, window_length_y=None),
    dict(window_length_z=21, window_length_x=None, window_length_y=None),
    dict(window_length_z=3, window_length_x=None, window_length_y=None),
])
def test_filter_rejects_incompatible_window_parameters(windows):
    with pytest.raises(ValueError):
        apply_savitzky_golay_filter_3d(np.ones((9, 9, 9)), **windows)


def test_import_and_creation_never_attempt_seismicmesh():
    repo = Path(RobustMesher.__file__).resolve().parent.parent
    code = r'''
import builtins,tempfile
from pathlib import Path
import numpy as np
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.lower().split('.')[0] == 'seismicmesh':
        raise AssertionError('Unexpected optional dependency import attempt')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from RobustMesher.sizing_utils.sizing3d import create_sizing_function3D
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / 'model.bin'
    values = np.full((2,3,4),1800.0)
    np.asarray(values,dtype='>f4').ravel(order='F').tofile(path)
    result = create_sizing_function3D(path,None,(-1,0,0,2,0,3),2,5,
                                     grade=0.0,nz=2,nx=3,ny=4)
    assert len(result) == 6 and result[3:] == (2,3,4)
    assert float(result[0](np.array([0.0,0.0,0.0]))) == 180.0
print('No SeismicMesh import attempted')
'''
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo)
    done = subprocess.run([sys.executable, '-c', code], cwd='/tmp', env=env,
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert done.returncode == 0, done.stdout
    assert 'No SeismicMesh import attempted' in done.stdout


@pytest.mark.parametrize('name,value', [('hmin', np.nan), ('hmin', np.inf), ('vp_water', np.nan), ('vp_water', np.inf)])
def test_rejects_nonfinite_size_floor_inputs(tmp_path, name, value):
    with pytest.raises(ValueError):
        _field(tmp_path, np.full((3, 4, 5), 1800.0), **{name: value})


@pytest.mark.parametrize('size', [5, 6])
def test_90_percent_window_guard_on_small_axes(tmp_path, size):
    iz, ix, iy = np.indices((size, size, size))
    grid = 150.0 + 10.0 * np.sin(iz) + 8.0 * np.cos(ix) + 4.0 * np.sin(iy)
    result = _field(tmp_path, grid * 10.0, grade=1.0, dtype='float64')
    bbox = (-float(size - 1), 0.0, 0.0, float(size - 1), 0.0, float(size - 1))
    actual = result[0](_node_queries(grid.shape, bbox)).reshape(grid.shape)
    if size == 5:
        # A five-sample cubic window would exceed 90% of this axis, so skip it.
        np.testing.assert_allclose(actual, grid, atol=5e-12)
    else:
        expected = apply_savitzky_golay_filter_3d(
            grid, window_length_z=5, window_length_x=5, window_length_y=5)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=5e-12)
        assert np.max(np.abs(actual - grid)) > 0.1
