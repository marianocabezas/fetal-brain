"""
Build an EllipseNet training set for the skull from images + skull masks.

HC18 layout (default):  <src>/<name>.png  and  <src>/<name>_Annotation.png
The annotation may be an ellipse outline or a filled ellipse; an ellipse is
fitted to its outer contour with OpenCV.

Writes, in EllipseNet's format (same as scripts/prepare_label.ipynb):
  <out>/images/<name>.png
  <out>/annotations/ellipse_annotations_{train,val}.json
with ann['ellipse'] = [cx, cy, rx, ry, theta]: pixels, semi-axes, theta in
radians (direction of rx from the x axis, image coordinates, as cv2.ellipse).

    python prepare_skull_coco.py HC18/training_set skull_coco --val 0.2
"""
import os
import json
import glob
import argparse
import shutil

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


def fit_skull_ellipse(mask):
    """(cx, cy, rx, ry, theta_rad) of the outer contour of a skull mask."""
    m = binary_fill_holes(mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return None
    (cx, cy), (w, h), ang = cv2.fitEllipse(c)
    return float(cx), float(cy), w / 2.0, h / 2.0, float(np.deg2rad(ang))


def envelope(cx, cy, rx, ry, theta):
    """Axis-aligned bounding box [x, y, w, h] of a rotated ellipse."""
    hw = np.sqrt((rx * np.cos(theta)) ** 2 + (ry * np.sin(theta)) ** 2)
    hh = np.sqrt((rx * np.sin(theta)) ** 2 + (ry * np.cos(theta)) ** 2)
    return [cx - hw, cy - hh, 2 * hw, 2 * hh]


def new_coco():
    return {'info': 'Fetal skull ellipses', 'license': ['none'], 'type': 'instances',
            'images': [], 'annotations': [],
            'categories': [{'supercategory': 'none', 'id': 1, 'name': 'Skull'}]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src')
    p.add_argument('out')
    p.add_argument('--suffix', default='_Annotation.png')
    p.add_argument('--val', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    names = sorted(os.path.basename(f)[:-len(args.suffix)]
                   for f in glob.glob(os.path.join(args.src, '*' + args.suffix)))
    # Split by subject, not by image: HC18 has several images of the same
    # fetus (010_HC, 010_2HC, ...), which must not end up on both sides.
    subject = lambda n: n.split('_')[0]
    subjects = sorted(set(subject(n) for n in names))
    rng = np.random.RandomState(args.seed)
    val_subj = set(rng.choice(subjects, int(round(args.val * len(subjects))), replace=False))
    val = set(n for n in names if subject(n) in val_subj)

    os.makedirs(os.path.join(args.out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'annotations'), exist_ok=True)
    sets = {'train': new_coco(), 'val': new_coco()}
    pix = []
    for i, name in enumerate(names):
        img = cv2.imread(os.path.join(args.src, name + '.png'), cv2.IMREAD_GRAYSCALE)
        ann = cv2.imread(os.path.join(args.src, name + args.suffix), cv2.IMREAD_GRAYSCALE)
        e = fit_skull_ellipse(ann) if ann is not None else None
        if img is None or e is None:
            print('skipped', name)
            continue
        cx, cy, rx, ry, th = e
        coco = sets['val' if name in val else 'train']
        img_id, ann_id = i + 1, i + 1
        coco['images'].append({'id': img_id, 'file_name': name + '.png',
                               'height': img.shape[0], 'width': img.shape[1]})
        poly = cv2.ellipse2Poly((int(cx), int(cy)), (int(rx), int(ry)),
                                int(np.rad2deg(th)), 0, 360, 15)
        bbox = envelope(cx, cy, rx, ry, th)
        coco['annotations'].append({
            'id': ann_id, 'image_id': img_id, 'category_id': 1,
            'bbox': bbox, 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'ignore': 0,
            'segmentation': [poly.reshape(-1).tolist()],
            'ellipse': [cx, cy, rx, ry, th]})
        cv2.imwrite(os.path.join(args.out, 'images', name + '.png'), img)
        if name not in val:
            pix.append(img.reshape(-1)[::50] / 255.0)

    for split, coco in sets.items():
        with open(os.path.join(args.out, 'annotations',
                               'ellipse_annotations_{}.json'.format(split)), 'w') as f:
            json.dump(coco, f)
        print(split, len(coco['images']), 'images')
    if pix:
        pix = np.concatenate(pix)
        print('Training mean/std for coco_skull.py: {:.3f} / {:.3f}'.format(pix.mean(), pix.std()))


if __name__ == '__main__':
    main()
