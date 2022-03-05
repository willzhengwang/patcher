"""
Tests for `patcher` package
"""

import numpy as np
from patcher import make_patches, merge_patches


def test_1d():
    """ Test on a Vector """
    # case 1: win_size > step
    arr = np.arange(9)
    patches = make_patches(arr, (5,), step=3, do_pad=True)
    assert np.allclose(patches, np.array([[0, 1, 2, 3, 4], [3, 4, 5, 6, 7], [6, 7, 8, 0, 0]]))
    mosaic = merge_patches(patches, out_shape=arr.shape, step=3)
    assert np.allclose(arr, mosaic)

    # case 2: win_size < step
    arr = np.arange(9)
    patches = make_patches(arr, (2,), step=3, do_pad=True)
    assert np.allclose(patches, np.array([[0, 1], [3, 4], [6, 7]]))
    mosaic = merge_patches(patches, out_shape=arr.shape, step=3)
    # There are some zeros in the gaps between patches
    assert np.allclose(mosaic, np.array([0, 1, 0, 3, 4, 0, 6, 7, 0]))

    # case 3: no padding in during patching
    arr = np.arange(9)
    patches = make_patches(arr, (5,), step=3, do_pad=False)
    assert np.allclose(patches, np.array([[0, 1, 2, 3, 4],
                                          [3, 4, 5, 6, 7]]))
    mosaic = merge_patches(patches, out_shape=arr.shape, step=3)
    assert not np.allclose(arr, mosaic)
    assert np.allclose(arr[:-1], mosaic[:-1])

    # case 4: no padding in during patching
    arr = np.arange(9)
    patches = make_patches(arr, (5,), step=3, do_pad=False)
    assert np.allclose(patches, np.array([[0, 1, 2, 3, 4],
                                          [3, 4, 5, 6, 7]]))
    mosaic = merge_patches(patches, step=3)
    assert np.allclose(arr[:-1], mosaic)


def test_2d():
    """ Test on a Matrix """

    # case 1: no padding
    arr = np.arange(8 * 9).reshape((8, 9))
    patch_shape, step = (5, 5), (4, 3)
    patches = make_patches(arr, patch_shape, step=step, do_pad=False)

    # case 1 - 1: out_shape is None
    mosaic = merge_patches(patches, step=step)
    assert np.allclose(arr[:5, :8], mosaic)

    # case 1 - 2: out_shape is larger than the originally merged array shape
    mosaic = merge_patches(patches, out_shape=arr.shape, step=step)
    assert np.allclose(arr[:5, :8], mosaic[:5, :8])
    assert mosaic.shape == arr.shape

    # case 2: padding
    arr = np.arange(9 * 9).reshape((9, 9))
    patch_shape, step = (5, 5), (3, 2)
    patches = make_patches(arr, patch_shape, step=step, do_pad=True)

    # case 2 - 1: out_shape is None
    mosaic = merge_patches(patches, step=step)
    assert mosaic.shape != arr.shape
    assert np.allclose(arr, mosaic[:9, :9])

    # case 2 - 2: out_shape is less than the originally merged array shape
    mosaic = merge_patches(patches, out_shape=arr.shape, step=step)
    assert np.allclose(arr, mosaic)


def test_3d():
    """ Test on a 3D array """

    arr = np.arange(8*9*6).reshape((8, 9, 6))
    patch_shape, step = (5, 5, 5), 3
    patches = make_patches(arr, patch_shape, step=step, do_pad=True)
    assert patches.shape == (2, 3, 2, 5, 5, 5)
    mosaic = merge_patches(patches, out_shape=arr.shape, step=3)
    assert np.allclose(arr, mosaic)


def test_4d():
    """ Test on a 4D array """

    arr = np.arange(8*9*6*2).reshape((8, 9, 6, 2))
    patches = make_patches(arr, (5, 5, 3, 2), step=(3, 3, 2, 2), do_pad=True)
    assert patches.shape == (2, 3, 3, 1, 5, 5, 3, 2)
    mosaic = merge_patches(patches, out_shape=arr.shape, step=(3, 3, 2, 2))
    assert np.allclose(arr, mosaic)
