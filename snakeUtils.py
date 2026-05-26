import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from gimpformats.gimpXcfDocument import GimpDocument

#from datasets import SkullUSDataset
from utils import load_xcf, makeDict
from metrics import hull, concave_hull, convex_hull

#from descartes import PolygonPatch

import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.segmentation import morphological_chan_vese


def anatStructMask(set1, height, width, alpha=0.01):
    """
        Function that returns a binary mask representing the area
        delimited by a set of point
        Does not use the image!!!!
    """
    if len(set1) > 2:
        poly1 = hull(set1, alpha)

        minx1, miny1, maxx1, maxy1 = poly1.bounds

        if np.any(np.isnan(poly1.bounds)):
            raise Exception("Problems computing Anatomical Structure Hull, hull crashed ")
        else:
            #maxx = int(max(maxx1, maxx2) * 1.2)
            #maxy = int(max(maxy1, maxy2) * 1.2)

            mask = np.zeros([height, width, 1])

            points1 = list(zip(*poly1.exterior.coords.xy))
            points1 = [[int(x), int(y)] for x, y in points1]
            points1 = np.array(points1)

            #print(points1)

            cv2.fillPoly(mask, pts=[points1], color=255)
            cv2.polylines(mask, pts=[points1], color=255, isClosed=True, thickness=1)

            return mask
    else:
        raise Exception("Problems computing Anatomical Structure Hull, not enough points ")


def layerMask(lay,aDict,height, width):
    """
        Input: Dimensions of the output image, name of a layer "lay"
               aDict, dictionary with the names of layers and points in every layer for an image
        Output: A binary mask creating the alphashape of the layer entered as a parameter
    """
    aPointSet = aDict[lay]
    #if we don't have at least 3 points we cannot compute anything
    if len(aPointSet) > 2 :
        return anatStructMask(aPointSet, height, width)
    else:
        raise Exception("layerMask, too few points")

def active_contours_from_rough_contour(image,mask):
    """
    Funcion that receives an image and
    the contour of an anatomical structure
    and refines the contour using
    morphological active contours
    """
    # prepare image and mask
    imageProcess = img_as_float(image)

    # Initial level set
    init_ls = np.squeeze(mask)

    # Call Active contours function, parameters could be adjusted
    return morphological_chan_vese(imageProcess, iterations=10, init_level_set=init_ls,
                                 smoothing=3, lambda1=0.5,lambda2=0.5)

def pointsToContouredMasks(imagePath, layerNames):
    """
        Receives the path to a xcf image with
        anatomical structures annotated as masks
        Also receives a list of layer names
        Converts every layer named into an area mask using
        alphashapes and active contours (functions layer mask
        and active_contours_from_rough_contour )
        For each mask created, store it in the same path as
        the original image with FilledMask and the name of the layer
        added to the original image name
    """
    def processLayer(lay):
        # create layer mask
        mask = layerMask(lay,dict1,height, width)
        ls = active_contours_from_rough_contour(image,mask)
        cv2.imwrite(outPath+"Layer"+lay+"FilledArea.png",255*ls)

    # variables to be used for layer processing
    image, data, labels, points = load_xcf(imagePath)
    dict1 = makeDict(labels,points)
    outPath = imagePath[:-4]
    height, width = image.shape
    # process each layer individually
    list(map(processLayer,layerNames))

def pointsToContouredMasksFolder(folderPath,layers):
    """
        receive a path with xcf images and a
        list of layers to process and call function
        pointsToContouredMasks in every xcf image
    """
    for root, _, files in os.walk(folderPath):
        for filename in files:
            if filename[-4:]==".xcf":
                pointsToContouredMasks(os.path.join(root, filename), layers)


def main():
    layers = ["cerebel","silvio","cavum"]

    pointsToContouredMasksFolder(sys.argv[1],layers)
    sys.exit()
    pointsToContouredMasks("data/97.1.xcf",layers)
    # Load image
    image, data, labels, points = load_xcf("data/97.1.xcf")
    dict1 = makeDict(labels,points)
    height, width = image.shape

    #choose a layer
    lay = "cerebel"

    # create layer mask
    mask = layerMask(lay,dict1,height, width)
    ls = active_contours_from_rough_contour(image,mask)

    cv2.imwrite("im.jpg",image)
    cv2.imwrite("initMask.jpg",mask)
    cv2.imwrite("out.jpg",255*ls)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(ls, [0.5], colors='r')
    ax[0].set_title("Morphological ACWE segmentation", fontsize=12)


if __name__ == '__main__':
    main()
