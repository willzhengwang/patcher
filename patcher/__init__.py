#!/usr/bin/env python

from typing import Tuple
import numpy as np
from numpy.lib.stride_tricks import as_strided
import numbers


def view_as_windows(arr_in: np.ndarray, win_shape: Tuple, step=1, do_pad=False, mode='constant', **kwargs):
    """
    # The code is developed on top of https://github.com/scikit-image/scikit-image/blob/main/skimage/util/shape.py
    
    # Copyright (C) 2011, the scikit-image team All rights reserved.
    #
    # Redistribution and use in source and binary forms, with or without modification, 
    # are permitted provided that the following conditions are met:
    # 
    # Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    # Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
    # in the documentation and/or other materials provided with the distribution.
    # Neither the name of skimage nor the names of its contributors may be used to endorse or promote products derived from this software 
    # without specific prior written permission.
    # THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
    # THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
    # IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    # (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
    # HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Args:
        arr_in (np.ndarray): input array
        win_shape (Tuple): window shape
        step (int/Tuple, optional): the step size between patches. A single integer, or a tuple that has the same as patch_size. Defaults to 1.
        do_pad (bool, optional): padding option. Defaults to False.
        mode (str, optional): the mode used in numpy.pad(). Defaults to 'constant'.
        **kwargs: dict. Any keyword arguments the numpy.pad() requires.

    Returns:
        _type_: _description_
    """

    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(win_shape, numbers.Number):
        win_shape = (win_shape,) * ndim
    if not (len(win_shape) == ndim):
        raise ValueError("`win_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
        
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    win_shape = np.array(win_shape, dtype=arr_shape.dtype)

    if ((arr_shape - win_shape) < 0).any():
        raise ValueError("`win_shape` is too large")

    if ((win_shape - 1) < 0).any():
        raise ValueError("`win_shape` is too small")

    win_indices_shape = (np.array(arr_in.shape) - np.max(np.array([np.array(win_shape), np.array(step)]), axis=0)) // np.array(step) + 1
    
    if do_pad: 
        win_remainders = np.remainder((np.array(arr_in.shape) - np.max(np.array([np.array(win_shape), np.array(step)]), axis=0)), np.array(step))
        if np.any(win_remainders): 
            npad = [(0, 0)] * ndim
            for i, remainder in enumerate(win_remainders):
                if remainder != 0:
                    npad[i] = (0, win_indices_shape[i] * step[i] + win_shape[i] - arr_in.shape[i])
                    win_indices_shape[i] += 1
            arr_in = np.pad(arr_in, pad_width=npad, mode=mode, **kwargs)
    
    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides
    
    new_shape = tuple(list(win_indices_shape) + list(win_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out
