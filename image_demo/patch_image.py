import cv2
import numpy as np
import matplotlib.pyplot as plt
from patcher import make_patches, merge_patches

if __name__ == '__main__':

    # read an image
    img = cv2.imread('sample.jpg')
    fig = plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original Image')
    fig.tight_layout()
    fig.savefig('original.jpg')
    plt.show()

    # split the image into small patches with padding
    patches = make_patches(img, (256, 256, 3), step=(228, 228, 3), do_pad=True)
    patch_indices = list(patches.shape[:3])
    fig, axarr = plt.subplots(patch_indices[0], patch_indices[1])

    for r in np.arange(patch_indices[0]):
        for c in np.arange(patch_indices[1]):
            patch = patches[r, c, 0]
            axarr[r, c].imshow(patch)
            axarr[r, c].axis('off')
    plt.suptitle('Patches with Overlapping and Padding')
    fig.tight_layout()
    plt.savefig('patches.jpg')
    plt.show()

    mosaic = merge_patches(patches, out_shape=img.shape, step=(228, 228, 3))
    fig = plt.figure()
    plt.imshow(mosaic)
    plt.axis('off')
    plt.title('Merged Image')
    fig.tight_layout()
    fig.savefig('merged.jpg')
    plt.show()
