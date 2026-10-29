"""
fetal_biometry.py
Fetal brain US biometry measured in the frame of the skull ellipse.

The skull ellipse defines the frame: "horizontal" is the direction of its
major axis and "vertical" is perpendicular to it. All values are in mm.

    DTC   longest vertical segment contained in the cerebellum
    CM    longest horizontal segment contained in the cisterna magna
          (along the brain's front-to-back axis)
    PN    longest segment of the nuchal fold cut by a line through the
          ellipse centre (all line directions are tried)
    CSPV  longest vertical segment contained in the cavum
    CSPH  longest horizontal segment contained in the cavum

Segments are contiguous runs inside the structure: a line that leaves the
structure and comes back gives two segments, and only the longer counts.

Geometry is done in mm (pixel x * sx, pixel y * sy), so "perpendicular"
is correct even if the row and column spacings differ.

Inputs (pick one):
  --processed DIR     Mariano's processed folder: <case>.png label map
                      (1 cavum, 2 cerebellum, 3 cisterna magna, 4 nuchal fold)
                      + <case>.dcm. Works for model predictions saved as label
                      maps in the same format.
  --xcf DIR --dicom DIR
                      Raw GIMP annotations, read as foetal_yago.py does
                      (needs utils.py / foetal_yago.py from fetal-brain).
  --masks-csv FILE    One binary or outline PNG per structure, as in our
                      biometry.py. Columns: case, dicom, and any of cavum,
                      cerebellum, cisterna_magna, nuchal_fold, skull.

Skull ellipse (one of):
  --ellipse-csv FILE  Predicted ellipses (e.g. EllipseNet output exported to
                      this format). Columns: case, x0, y0, a, b, theta.
                      Pixels, image coordinates (x right, y down); theta in
                      radians is the angle of semi-axis a from the x axis.
  skull column in --masks-csv: an ellipse is fitted to that mask.

Check the overlays (--overlay-dir) before trusting the numbers: an angle
convention that differs from the one above swaps horizontal and vertical.
"""
import os
import csv
import argparse
from dataclasses import dataclass

import cv2
import numpy as np
import pydicom
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize

STRUCTURES = ['cavum', 'cerebellum', 'cisterna_magna', 'nuchal_fold']
# Label values in the processed PNGs (class order in foetal_yago.py)
LABELS = {'cavum': 1, 'cerebellum': 2, 'cisterna_magna': 3, 'nuchal_fold': 4}
# XCF layer names (lowercased, after normalize_layer_dict)
XCF_LAYERS = {'cavum': 'cavum', 'cerebellum': 'cerebel',
              'cisterna_magna': 'cisterna magna', 'nuchal_fold': 'plec nucal'}

CLOSE_KERNEL = 7      # px, closes small gaps in outline masks
STEP_PX = 0.5         # sampling step along and between lines, in pixels


# ----------------------------------------------------------------- spacing
def mm_per_pixel(ds):
    """(sx, sy) in mm from a pydicom dataset, as segmentation.py does:
    PixelSpacing first (row, col) = (y, x), ultrasound regions as fallback."""
    if "PixelSpacing" in ds:
        return float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0])
    if "SequenceOfUltrasoundRegions" in ds:
        for r in ds.SequenceOfUltrasoundRegions:
            if r.PhysicalUnitsXDirection == 3 and r.PhysicalUnitsYDirection == 3:  # cm
                return float(r.PhysicalDeltaX) * 10, float(r.PhysicalDeltaY) * 10
    raise ValueError("No pixel spacing in DICOM")


def read_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array
    if img.ndim == 3:                      # RGB ultrasound, as in foetal_yago.py
        img = img.mean(axis=-1)
    return img, mm_per_pixel(ds)


# ----------------------------------------------------------------- ellipse
@dataclass
class Ellipse:
    """Pixels, image coordinates. Semi-axis a points along (cos theta, sin theta)."""
    x0: float
    y0: float
    a: float
    b: float
    theta: float


def fit_ellipse_to_mask(mask):
    """HC18-style fit: ellipse through the skeleton of a skull mask (OpenCV).
    Swap in Mariano's robust_fit_ellipse here if preferred (see notes)."""
    ys, xs = np.nonzero(skeletonize(mask > 0))
    if len(xs) < 5:
        raise ValueError("Skull mask too small to fit an ellipse")
    (cx, cy), (w, h), ang = cv2.fitEllipse(np.column_stack([xs, ys]).astype(np.float32))
    # OpenCV: axis of length w is rotated by ang degrees from x
    return Ellipse(cx, cy, w / 2, h / 2, np.deg2rad(ang))


def read_ellipse_csv(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r['case']] = Ellipse(*(float(r[k]) for k in ('x0', 'y0', 'a', 'b', 'theta')))
    return out


def ellipse_frame_mm(e, spacing):
    """Centre (mm) and unit vectors u (major axis, 'horizontal') and v
    ('vertical') in mm space. The pixel ellipse is mapped to mm and its axes
    recomputed, so this stays correct for anisotropic pixels."""
    sx, sy = spacing
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    ca, sa = np.cos(e.theta), np.sin(e.theta)
    px = e.x0 + e.a * np.cos(t) * ca - e.b * np.sin(t) * sa
    py = e.y0 + e.a * np.cos(t) * sa + e.b * np.sin(t) * ca
    pts = np.column_stack([px * sx, py * sy])
    c = pts.mean(axis=0)
    w, vecs = np.linalg.eigh(np.cov((pts - c).T))
    u = vecs[:, np.argmax(w)]
    v = np.array([-u[1], u[0]])
    return np.array([e.x0 * sx, e.y0 * sy]), u, v


# ----------------------------------------------------------------- masks
def fill_mask(mask):
    """Binary mask from a filled or outline annotation: close small gaps,
    then fill holes (as in biometry.py / foetal_yago.py)."""
    m = (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return binary_fill_holes(m)


# ----------------------------------------------------------------- line scans
@dataclass
class Segment:
    length_mm: float
    p0: tuple = None      # endpoints in pixels (x, y), for overlays
    p1: tuple = None


def _longest_runs(mask, spacing, origins, dirs, t_min, t_max, h, chunk=256):
    """For each line origin + t*dir (mm), the longest contiguous run of samples
    inside the mask. Returns the best Segment over all lines."""
    sx, sy = spacing
    H, W = mask.shape
    ts = np.arange(t_min, t_max + h, h)
    best = Segment(0.0)
    for i in range(0, len(origins), chunk):
        o = origins[i:i + chunk]
        d = dirs[i:i + chunk] if dirs.ndim == 2 else np.broadcast_to(dirs, o.shape)
        X = (o[:, None, 0] + ts[None] * d[:, None, 0]) / sx
        Y = (o[:, None, 1] + ts[None] * d[:, None, 1]) / sy
        xi, yi = np.rint(X).astype(int), np.rint(Y).astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        inside = np.zeros(X.shape, bool)
        inside[ok] = mask[yi[ok], xi[ok]]
        # run length ending at each sample, vectorised over lines
        run = np.zeros(len(o), int)
        best_len = np.zeros(len(o), int)
        best_end = np.zeros(len(o), int)
        for j in range(inside.shape[1]):
            run = (run + 1) * inside[:, j]
            better = run > best_len
            best_len[better] = run[better]
            best_end[better] = j
        k = int(np.argmax(best_len))
        n = best_len[k]
        # boundaries taken half a step outside the first/last inside sample
        length = n * h
        if length > best.length_mm:
            j1, j0 = best_end[k], best_end[k] - n + 1
            t0, t1 = ts[j0] - h / 2, ts[j1] + h / 2
            p0 = (o[k] + t0 * d[k]) / (sx, sy)
            p1 = (o[k] + t1 * d[k]) / (sx, sy)
            best = Segment(float(length), tuple(p0), tuple(p1))
    return best


def longest_parallel(mask, spacing, direction):
    """Longest segment inside the mask along lines parallel to direction (mm)."""
    if not mask.any():
        return Segment(np.nan)
    sx, sy = spacing
    h = STEP_PX * min(sx, sy)
    ys, xs = np.nonzero(mask)
    P = np.column_stack([xs * sx, ys * sy])
    n = np.array([-direction[1], direction[0]])
    s, t = P @ n, P @ direction
    pad = max(sx, sy)
    offsets = np.arange(s.min() - pad, s.max() + pad + h, h)
    return _longest_runs(mask, spacing, offsets[:, None] * n, direction,
                         t.min() - pad, t.max() + pad, h)


def longest_through_point(mask, spacing, centre):
    """Longest segment inside the mask along any line through centre (mm)."""
    if not mask.any():
        return Segment(np.nan)
    sx, sy = spacing
    h = STEP_PX * min(sx, sy)
    ys, xs = np.nonzero(mask)
    r = np.hypot(xs * sx - centre[0], ys * sy - centre[1])
    r_max = r.max() + max(sx, sy)
    dphi = h / r_max                      # < half a pixel apart at the far end
    phi = np.arange(0, np.pi, dphi)
    dirs = np.column_stack([np.cos(phi), np.sin(phi)])
    origins = np.broadcast_to(centre, dirs.shape)
    return _longest_runs(mask, spacing, origins, dirs, -r_max, r_max, h)


def measure(masks, spacing, ellipse):
    """All biometries for one case. masks: dict name -> bool array."""
    c, u, v = ellipse_frame_mm(ellipse, spacing)
    empty = np.zeros((1, 1), bool)
    get = lambda k: masks.get(k, empty)
    return {
        'DTC':  longest_parallel(get('cerebellum'), spacing, v),
        'CM':   longest_parallel(get('cisterna_magna'), spacing, u),
        'PN':   longest_through_point(get('nuchal_fold'), spacing, c),
        'CSPV': longest_parallel(get('cavum'), spacing, v),
        'CSPH': longest_parallel(get('cavum'), spacing, u),
    }


# ----------------------------------------------------------------- loaders
def load_processed(folder, case):
    label = cv2.imread(os.path.join(folder, case + '.png'), cv2.IMREAD_UNCHANGED)
    img, spacing = read_dicom(os.path.join(folder, case + '.dcm'))
    masks = {k: label == v for k, v in LABELS.items()}
    return img, spacing, masks, None


def load_xcf_case(xcf_folder, dicom_folder, case):
    from utils import load_xcf                       # fetal-brain repo
    from foetal_yago import normalize_layer_dict
    names, data = load_xcf(os.path.join(xcf_folder, case + '.xcf'))
    layers = normalize_layer_dict(
        {n.lower(): d for n, d in zip(names, data) if '.dcm' not in n})
    img, spacing = read_dicom(os.path.join(dicom_folder, case + '.dcm'))
    masks = {k: binary_fill_holes(np.array(layers[l])[..., -1] > 0)
             for k, l in XCF_LAYERS.items() if l in layers}
    return img, spacing, masks, None


def load_masks_row(row):
    img, spacing = read_dicom(row['dicom'])
    masks, skull = {}, None
    for k in STRUCTURES + ['skull']:
        if row.get(k):
            m = cv2.imread(row[k], cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(row[k])
            if k == 'skull':
                skull = m > 0
            else:
                masks[k] = fill_mask(m)
    return img, spacing, masks, skull


# ----------------------------------------------------------------- overlay
COLOURS = {'DTC': (0, 200, 0), 'CM': (255, 160, 0), 'PN': (255, 0, 255),
           'CSPV': (0, 220, 255), 'CSPH': (0, 120, 255)}           # BGR
STRUCT_COLOURS = {'cerebellum': (0, 200, 0), 'cisterna_magna': (255, 160, 0),
                  'nuchal_fold': (255, 0, 255), 'cavum': (0, 200, 255)}
ELLIPSE_COLOUR = (0, 0, 255)


def _to_bgr(img):
    g = np.asarray(img, float)
    g = (255 * (g - g.min()) / max(np.ptp(g), 1)).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _label(img, text, org, colour, scale, thick):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def save_overlay(path, img, masks, ellipse, results, case=''):
    """Two panels side by side: image with the biometry lines only | image
    with the skull ellipse, its major axis and centre, the structure outlines
    and every biometry line with names and values."""
    left = _to_bgr(img)
    right = left.copy()
    plain = left.copy()
    H, W = left.shape[:2]
    t = max(1, int(round(min(H, W) / 400)))          # line thickness scales with image size
    scale = 0.5 * t + 0.2

    # structure outlines (thin), in the colour of the measurement they define
    for k, m in masks.items():
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(right, cnts, -1, STRUCT_COLOURS.get(k, (160, 160, 160)), t, cv2.LINE_AA)

    # skull ellipse, major axis ("horizontal") and centre
    e = ellipse
    c = np.array([e.x0, e.y0])
    cv2.ellipse(right, (int(round(e.x0)), int(round(e.y0))), (int(round(e.a)), int(round(e.b))),
                np.rad2deg(e.theta), 0, 360, ELLIPSE_COLOUR, 2 * t, cv2.LINE_AA)
    if abs(e.a) >= abs(e.b):
        ax = np.array([np.cos(e.theta), np.sin(e.theta)]) * abs(e.a)
    else:
        ax = np.array([-np.sin(e.theta), np.cos(e.theta)]) * abs(e.b)
    cv2.line(right, tuple(np.rint(c - ax).astype(int)), tuple(np.rint(c + ax).astype(int)),
             ELLIPSE_COLOUR, t, cv2.LINE_AA)
    cv2.circle(right, tuple(np.rint(c).astype(int)), 3 * t, ELLIPSE_COLOUR, -1, cv2.LINE_AA)

    # biometry lines with end dots and a short label next to each line
    used = []                                         # label boxes already placed
    for k, sgm in results.items():
        if sgm.p0 is None:
            continue
        p0, p1 = np.rint(sgm.p0).astype(int), np.rint(sgm.p1).astype(int)
        cv2.line(right, tuple(p0), tuple(p1), COLOURS[k], 2 * t, cv2.LINE_AA)
        cv2.line(plain, tuple(p0), tuple(p1), COLOURS[k], 2 * t, cv2.LINE_AA)
        for p in (p0, p1):
            cv2.circle(right, tuple(p), 2 * t + 1, COLOURS[k], -1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(k, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
        x, y = int(max(p0[0], p1[0]) + 6 * t), int(min(p0[1], p1[1]) + th)
        while any(abs(x - ux) < max(tw, uw) and abs(y - uy) < th + 4 for ux, uy, uw in used):
            y += th + 6                               # move down until it does not overlap
        x, y = min(x, W - tw - 2), min(max(y, th + 2), H - 2)
        used.append((x, y, tw))
        _label(right, k, (x, y), COLOURS[k], scale, t)

    # legend
    y = int(30 * scale + 10)
    _label(right, 'skull ellipse (major axis = horizontal)', (10, y), ELLIPSE_COLOUR, scale, t)
    for k, sgm in results.items():
        y += int(32 * scale + 8)
        val = 'n/a' if not np.isfinite(sgm.length_mm) else '{:.2f} mm'.format(sgm.length_mm)
        _label(right, '{:<5} {}'.format(k, val), (10, y), COLOURS[k], scale, t)
    if case:
        _label(plain, str(case), (10, int(30 * scale + 10)), (255, 255, 255), scale, t)

    gap = np.full((H, 8 * t, 3), 255, np.uint8)
    cv2.imwrite(path, np.hstack([plain, gap, right]))


# ----------------------------------------------------------------- main
def run_case(case, img, spacing, masks, skull, ellipses, overlay_dir):
    for k, m in masks.items():
        if m.shape != img.shape[:2]:
            raise ValueError(f"{case}: {k} mask {m.shape} != DICOM {img.shape[:2]}")
    if case in ellipses:
        e = ellipses[case]
    elif skull is not None:
        e = fit_ellipse_to_mask(skull)
    else:
        raise ValueError(f"{case}: no skull ellipse (give --ellipse-csv or a skull mask)")
    res = measure(masks, spacing, e)
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
        save_overlay(os.path.join(overlay_dir, case + '_biometry.png'), img, masks, e, res, case)
    row = {'case': case, 'sx_mm': spacing[0], 'sy_mm': spacing[1],
           'ellipse_x0': e.x0, 'ellipse_y0': e.y0, 'ellipse_a': e.a,
           'ellipse_b': e.b, 'ellipse_theta': e.theta}
    row.update({k: s.length_mm for k, s in res.items()})
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--processed')
    src.add_argument('--xcf')
    src.add_argument('--masks-csv')
    p.add_argument('--dicom', help='DICOM folder (with --xcf)')
    p.add_argument('--ellipse-csv')
    p.add_argument('--out', default='biometry_results.csv')
    p.add_argument('--overlay-dir')
    args = p.parse_args()

    ellipses = read_ellipse_csv(args.ellipse_csv) if args.ellipse_csv else {}

    if args.processed:
        cases = sorted(f[:-4] for f in os.listdir(args.processed) if f.endswith('.png'))
        jobs = [(c, lambda c=c: load_processed(args.processed, c)) for c in cases]
    elif args.xcf:
        if not args.dicom:
            p.error('--xcf needs --dicom')
        cases = sorted(f[:-4] for f in os.listdir(args.xcf) if f.endswith('.xcf'))
        jobs = [(c, lambda c=c: load_xcf_case(args.xcf, args.dicom, c)) for c in cases]
    else:
        with open(args.masks_csv) as f:
            rows = list(csv.DictReader(f))
        jobs = [(r['case'], lambda r=r: load_masks_row(r)) for r in rows]

    results = []
    for case, load in jobs:
        try:
            results.append(run_case(case, *load(), ellipses, args.overlay_dir))
            print(case, {k: round(results[-1][k], 2) for k in ('DTC', 'CM', 'PN', 'CSPV', 'CSPH')})
        except Exception as ex:                     # report and keep going
            print('ERROR', case, ':', ex)
            results.append({'case': case, 'error': str(ex)})

    keys = []
    for r in results:
        keys += [k for k in r if k not in keys]
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print('Saved', args.out)


if __name__ == '__main__':
    main()
