# POINT METRICS
from scipy.spatial.distance import directed_hausdorff
import numpy as np
import persim
import cv2
import alphashape
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


def hausdorff(set1, set2):
    """ Given two sets of points, compute their hausdorff distance

    """
    return directed_hausdorff(set1, set2)[0]


"""
The following function can be made much more efficient with Kd-trees (but 
 probably no need given our data sizes)
"""


def avEUC(set1, set2):
    """
    Average euclidean distance from a point in one set to the closest point
    in the other set
    """

    def closestDist(p, set2):
        """
            Smaller euclidean Distance from point to set
        """
        minDist = np.linalg.norm(p - set2[0])
        for q in set2[1:]:
            newDist = np.linalg.norm(p - q)
            if newDist < minDist:
                minDist = newDist
        return minDist

    # iterate over all the set
    sum = 0
    for p in set1:
        sum += closestDist(p, set2)
    return sum / len(set1)


def bottleneck(set1, set2):
    """
        Bottleneck Distance between point sets
    """
    return persim.bottleneck(set1, set2, matching=True)[0]


# Function that returns a concave hull from a set of points
def concave_hull(points, alpha=0.2):
    """
    Compute the concave hull of a set of points, at least 5 points are
    recommended 3 are necessary
    """
    if len(points) <= 2:
        raise Exception(
            "Concave_hull Hull, cannot compute with less than 3 points"
        )
    return alphashape.alphashape(points, alpha)


# Function that returns a convex hull from a set of points
def convex_hull(points):
    """
    Compute the convex hull of a set of points, at least 2 points are necessary
    """
    if len(points) <= 2:
        raise Exception(
            "Convex Hull, cannot compute with less than 3 points"
        )
    return alphashape.alphashape(points, 0.)


def hull(points, alpha=0):
    if len(points) < 5:
        return convex_hull(points)
    else:
        return concave_hull(points, alpha)


# Function that receives two polygons and computes two binary
# images with coordinates that contain them both
def polygonsDice(set1, set2):
    if len(set1) > 2 and len(set2) > 2:
        poly1 = hull(set1)
        poly2 = hull(set2)

        minx1, miny1, maxx1, maxy1 = poly1.bounds
        minx2, miny2, maxx2, maxy2 = poly2.bounds

        maxx = int(max(maxx1, maxx2) * 1.2)
        maxy = int(max(maxy1, maxy2) * 1.2)

        im1 = np.zeros([maxx, maxy, 1])
        im2 = np.zeros([maxx, maxy, 1])

        points1 = list(zip(*poly1.exterior.coords.xy))
        points1 = [[int(x[0]), int(x[1])] for x in points1]
        points1 = np.array(points1)
        points2 = list(zip(*poly2.exterior.coords.xy))
        points2 = [[int(x[0]), int(x[1])] for x in points2]
        points2 = np.array(points2)

        # print(points1)
        # print(points2)

        cv2.fillPoly(im1, pts=[points1], color=255)
        cv2.fillPoly(im2, pts=[points2], color=255)

        # Now compute the dice between the two
        X = np.sum(im1 == 255)
        Y = np.sum(im2 == 255)
        aux = im1.copy()
        aux[im2 == 0] = 0
        XandY = np.sum(aux == 255)

        if (X + Y) > 0:
            dsc = 2 * XandY / (X + Y)
        else:
            dsc = 0
    else:
        dsc = None

    return dsc


def MSE(im1,im2):
    """
    Mean Squared Error image metric
    """
    return  mean_squared_error(im1, im2)

def SSIM(im1,im2):
    """
    Structural Similarity Index image metric
    """
    data_range = max(im1.max(),im2.max()) - min(im1.min(),im2.min())
    return ssim(im1, im2,data_range=data_range)

