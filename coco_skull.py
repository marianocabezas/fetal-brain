"""
COCOSkull: one-class (skull) version of EllipseNet's COCOFHD dataset.
Installed into ellipsenet/src/lib/datasets/dataset/ by patch_ellipsenet.py.

Expects, like COCOFHD:
  <data_dir>/images/*.png
  <data_dir>/annotations/ellipse_annotations_{train,val}.json
(written by prepare_skull_coco.py)
"""
from .coco_fhd import COCOFHD
import pycocotools.coco as coco
import numpy as np
import os


class COCOSkull(COCOFHD):
  num_classes = 1
  default_resolution = [544, 800]          # HC18 images are 540 x 800
  # Grayscale mean/std of the training images: replace with the values
  # printed by prepare_skull_coco.py (these are COCOFHD's).
  mean = np.array([0.151, 0.151, 0.151], dtype=np.float32).reshape(1, 1, 3)
  std = np.array([0.171, 0.171, 0.171], dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    # Same as COCOFHD.__init__ but one class, one object per image and the
    # ellipse annotation file for every split.
    self.data_dir = opt.data_dir
    self.img_dir = os.path.join(self.data_dir, 'images')
    self.annot_path = os.path.join(
      self.data_dir, 'annotations', 'ellipse_annotations_{}.json'.format(split))
    self.max_objs = 1
    self.class_name = ['__background__', 'skull']
    self._valid_ids = [1]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(255, 0, 0)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)
    self.split = split
    self.opt = opt
    print('==> initializing skull {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)
    print('Loaded {} {} samples'.format(split, self.num_samples))
