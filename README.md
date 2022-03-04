# Patcher

Pacther is a python3-based library, aiming to split a large N-dimensional array (e.g. a 2D grey or 3D color image) into small patches for processing, and then merge the small patches into a large array. 

# Features
## 1. Padding
is supported in the patching function. 
## 2. Overlap manipulation
 Operation functions (such as mean, max, and min) can be applied to the overlaps between neighbouring patches, which may be useful for semantic segmentation. 

# Installation

Install `patcher` using `pip`:

``` {.sourceCode .bash}
$ pip install patcher
```

# Usages
