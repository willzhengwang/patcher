#!/usr/bin/env python

"""
Tests for `patcher` package
"""

import numpy as np
from patcher import patch, unpatch


def test_1d():
    
    # case 1: win_size > step
    arr = np.arange(9)
    patches = patch(arr, (5, ), step=3, do_pad=True)
    mosaic = unpatch(patches, arr.shape, step=3)
    assert np.allclose(arr, mosaic)
    
    # case 2: win_size < step 
    arr = np.arange(9)
    patches = patch(arr, (2, ), step=3, do_pad=True)
    mosaic = unpatch(patches, arr.shape, step=3)


def test_2d():
    # Test on 2D data
    arr = np.arange(8*9).reshape((8, 9))
    patch_shape, step = (5, 5), (3, 2)
    patches = patch(arr, patch_shape, step=step, do_pad=True)
    mosaic = unpatch(patches, arr.shape, step=step)
    assert np.allclose(arr, mosaic)


def test_3d():
    # Test on 3D data
    arr = np.arange(8*9*6).reshape((8, 9, 6))
    patch_shape, step = (5, 5, 5), 3
    patches = patch(arr, patch_shape, step=step, do_pad=True)
    mosaic = unpatch(patches, arr.shape, step=3)
    # There are some zeros in the gaps between patches
    assert not np.allclose(arr, mosaic)
    assert arr.shape == mosaic.shape


def test_4d():
    # Test on 4D data
    arr = np.arange(8*9*6*2).reshape((8, 9, 6, 2))
    patches = patch(arr, (5, 5, 3, 2), step=3, do_pad=True)
    mosaic = unpatch(patches, arr.shape, step=3)
    assert not np.allclose(arr, mosaic)
    print('Done!')
