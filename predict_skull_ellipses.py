"""
Run a trained EllipseNet skull model on our images and write the ellipse CSV
read by fetal_biometry.py (--ellipse-csv).

    python predict_skull_ellipses.py --ellipsenet /path/to/ellipsenet \
        --model exp/eldet/skull/model_best.pth --processed ProcessedData/ \
        --out skull_ellipses.csv --overlay-dir skull_overlays/

Inputs: --processed / --dicom (folder of <case>.dcm) or --images (folder of
<case>.png). Runs on the GPU by default; --gpus -1 runs on the CPU (slow).

Output columns: case, x0, y0, a, b, theta, score. Pixels in the original
image; a is the major semi-axis; theta (radians) is its direction from the
x axis in image coordinates (x right, y down), as cv2.ellipse draws it.
"""
import os
import sys
import csv
import glob
import argparse

import cv2
import numpy as np


def decode(rows):
    """Best detection of EllipseNet's eldet output for one class.

    Each row (EldetDetector.post_process) is
    [x1, y1, x2, y2, cx, cy, l, ratio_al, ratio_bl, theta, sin, cos, score],
    already mapped back to image pixels. As in scripts/Inference_ellipse.ipynb:
    a = ratio_al * l / 2, b = ratio_bl * l / 2, angle = theta * 180 degrees.
    """
    rows = np.asarray(rows, dtype=float).reshape(-1, 13)
    if len(rows) == 0:
        return None
    r = rows[np.argmax(rows[:, 12])]
    cx, cy, l = r[4], r[5], r[6]
    return dict(x0=cx, y0=cy, a=r[7] * l / 2, b=r[8] * l / 2,
                theta=r[9] * np.pi, score=r[12])


def to_bgr(img):
    """uint8 3-channel image, as cv2.imread gives EllipseNet in training."""
    img = np.asarray(img, dtype=float)
    if img.ndim == 3:
        img = img.mean(axis=-1)                      # RGB ultrasound -> gray
    lo, hi = img.min(), img.max()
    g = (255 * (img - lo) / max(hi - lo, 1e-6)).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def list_cases(args):
    if args.images:
        return [(os.path.basename(f)[:-4], f) for f in sorted(glob.glob(os.path.join(args.images, '*.png')))]
    folder = args.processed or args.dicom
    return [(os.path.basename(f)[:-4], f) for f in sorted(glob.glob(os.path.join(folder, '*.dcm')))]


def load_image(path):
    if path.endswith('.dcm'):
        import pydicom
        return to_bgr(pydicom.dcmread(path).pixel_array)
    return to_bgr(cv2.imread(path, cv2.IMREAD_GRAYSCALE))


def build_detector(args):
    import torch
    if args.gpus == '-1' or not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **k: None   # EllipseNet calls it unconditionally
    lib = os.path.join(args.ellipsenet, 'src', 'lib')
    # EllipseNet's packages (datasets, models, utils, ...) have no __init__.py,
    # so same-named modules next to this script (fetal-brain's datasets.py,
    # models.py, utils.py) would be imported instead. Hide this folder first.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or os.getcwd()) not in (here, os.getcwd())]
    for name in ('datasets', 'models', 'utils', 'detectors', 'trains', 'external',
                 'opts', 'fhd_opts', 'logger'):
        sys.modules.pop(name, None)
    sys.path.insert(0, lib)
    from fhd_opts import opts
    from datasets.dataset_factory import dataset_factory
    from detectors.detector_factory import detector_factory
    opt = opts().parse(['eldet', '--dataset', 'coco_skull', '--load_model', args.model,
                        '--arch', args.arch, '--K', '1', '--gpus', args.gpus,
                        '--input_h', str(args.input_h), '--input_w', str(args.input_w)])
    opt = opts().update_dataset_info_and_set_heads(opt, dataset_factory['coco_skull'])
    return detector_factory['eldet'](opt)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--processed')
    src.add_argument('--dicom')
    src.add_argument('--images')
    p.add_argument('--ellipsenet', required=True, help='EllipseNet checkout (patched)')
    p.add_argument('--model', required=True)
    p.add_argument('--arch', default='dla_34', help='same as used for training')
    p.add_argument('--input_h', type=int, default=544)
    p.add_argument('--input_w', type=int, default=800)
    p.add_argument('--gpus', default='0')
    p.add_argument('--out', default='skull_ellipses.csv')
    p.add_argument('--overlay-dir')
    args = p.parse_args()

    detector = build_detector(args)
    if args.overlay_dir:
        os.makedirs(args.overlay_dir, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['case', 'x0', 'y0', 'a', 'b', 'theta', 'score'])
        w.writeheader()
        for case, path in list_cases(args):
            img = load_image(path)
            e = decode(detector.run(img)['results'][1])
            if e is None:
                print('no skull found:', case)
                continue
            w.writerow(dict(case=case, **e))
            print('{:<8} centre ({:.0f}, {:.0f})  a {:.0f}  b {:.0f}  angle {:.1f} deg  score {:.2f}'.format(
                case, e['x0'], e['y0'], e['a'], e['b'], np.rad2deg(e['theta']), e['score']))
            if args.overlay_dir:
                cv2.ellipse(img, (int(e['x0']), int(e['y0'])), (int(e['a']), int(e['b'])),
                            np.rad2deg(e['theta']), 0, 360, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(args.overlay_dir, case + '_skull.png'), img)
    print('Saved', args.out)


if __name__ == '__main__':
    main()
