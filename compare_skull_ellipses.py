"""
Compare the two skull ellipses (Mariano-style UNet + robust_fit_ellipse vs
EllipseNet) and the biometries computed from each.

Step 1 - convert the UNet ellipses (skull_pipeline.py output) to the format
fetal_biometry.py reads:

    python compare_skull_ellipses.py convert OOO_ellipse_params.csv unet_ellipses.csv

Step 2 - run fetal_biometry.py once per ellipse source, so that only the
skull frame differs between the two runs (same masks, same measurement):

    python fetal_biometry.py --processed ProcessedData --ellipse-csv unet_ellipses.csv  --out biometry_unet.csv
    python fetal_biometry.py --processed ProcessedData --ellipse-csv skull_ellipses.csv --out biometry_ellipsenet.csv

Step 3 - compare:

    python compare_skull_ellipses.py compare \
        --unet unet_ellipses.csv --ellipsenet skull_ellipses.csv \
        --bio-unet biometry_unet.csv --bio-ellipsenet biometry_ellipsenet.csv \
        --tcd biometries_result/tcd_measurement.csv --out ellipse_comparison.csv

--tcd (biometry.py's TCD) is optional. TCD vs DTC differs in more than the
skull frame (different cerebellum masks, centreline vs outer edge), so it is
reported separately and is not a pure test of the ellipses.

Ellipse format: case, x0, y0, a, b, theta. Pixels, image coordinates
(x right, y down); theta in radians is the direction of semi-axis a.
"""
import os
import re
import sys
import csv
import argparse

import numpy as np
import pandas as pd

MEASURES = ['DTC', 'CM', 'PN', 'CSPV', 'CSPH']


def case_id(name):
    """Common case key: drop folders, 'result_' prefix and file extension."""
    name = os.path.basename(str(name).strip())
    name = re.sub(r'^result_', '', name)
    name = re.sub(r'\.(xcf|dcm|png)$', '', name, flags=re.IGNORECASE)
    return name


def convert(src, dst):
    df = pd.read_csv(src, encoding='utf-8-sig')
    out = pd.DataFrame({'case': df['name'].map(case_id),
                        'x0': df['x0'], 'y0': df['y0'],
                        'a': df['a'], 'b': df['b'], 'theta': df['theta']})
    out.to_csv(dst, index=False)
    print('Wrote {} ellipses to {}'.format(len(out), dst))


def major_axis_deg(a, b, theta):
    """Direction of the major axis in degrees, in [-90, 90)."""
    deg = np.degrees(theta) + np.where(np.abs(a) >= np.abs(b), 0.0, 90.0)
    return (deg + 90) % 180 - 90


def angle_diff(d1, d2):
    """Smallest difference between two axis directions (mod 180), in [-90, 90)."""
    return (d1 - d2 + 90) % 180 - 90


def load_ellipses(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['case'] = df['case'].map(case_id)
    return df.set_index('case')


def summary(diff):
    d = diff.dropna()
    if len(d) == 0:
        return 'no cases'
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return ('n={:d}  bias {:+.2f}  SD {:.2f}  LoA [{:.2f}, {:.2f}]  '
            'median |d| {:.2f}  max |d| {:.2f}').format(
        len(d), d.mean(), sd, d.mean() - 1.96 * sd, d.mean() + 1.96 * sd,
        d.abs().median(), d.abs().max())


def worst(series, n=5):
    s = series.dropna()
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:n]
    return ', '.join('{} ({:+.2f})'.format(k, v) for k, v in s.items())


def compare(args):
    u = load_ellipses(args.unet)
    e = load_ellipses(args.ellipsenet)
    cases = sorted(set(u.index) & set(e.index))
    print('Ellipses: {} UNet, {} EllipseNet, {} in both'.format(len(u), len(e), len(cases)))
    only = sorted(set(u.index) ^ set(e.index))
    if only:
        print('  in only one source (first 10):', only[:10])

    u, e = u.loc[cases], e.loc[cases]
    res = pd.DataFrame(index=pd.Index(cases, name='case'))
    res['axis_unet_deg'] = major_axis_deg(u['a'], u['b'], u['theta'])
    res['axis_ellipsenet_deg'] = major_axis_deg(e['a'], e['b'], e['theta'])
    res['axis_diff_deg'] = angle_diff(res['axis_unet_deg'], res['axis_ellipsenet_deg'])
    res['centre_dist_px'] = np.hypot(u['x0'] - e['x0'], u['y0'] - e['y0'])
    res['major_unet_px'] = np.maximum(u['a'].abs(), u['b'].abs())
    res['major_ellipsenet_px'] = np.maximum(e['a'].abs(), e['b'].abs())
    res['minor_unet_px'] = np.minimum(u['a'].abs(), u['b'].abs())
    res['minor_ellipsenet_px'] = np.minimum(e['a'].abs(), e['b'].abs())

    print('\nSkull ellipse, UNet - EllipseNet')
    print('  axis direction (deg):  ', summary(res['axis_diff_deg']))
    print('    largest:', worst(res['axis_diff_deg']))
    print('  centre distance (px):   median {:.1f}  max {:.1f}  (largest: {})'.format(
        res['centre_dist_px'].median(), res['centre_dist_px'].max(),
        worst(res['centre_dist_px'])))
    print('  major semi-axis (px):  ', summary(res['major_unet_px'] - res['major_ellipsenet_px']))

    if args.bio_unet and args.bio_ellipsenet:
        bu = pd.read_csv(args.bio_unet).assign(case=lambda d: d['case'].map(case_id)).set_index('case')
        be = pd.read_csv(args.bio_ellipsenet).assign(case=lambda d: d['case'].map(case_id)).set_index('case')
        print('\nBiometry from each ellipse, UNet - EllipseNet (mm; same masks, only the frame differs)')
        for m in MEASURES:
            if m in bu and m in be:
                res[m + '_unet'] = bu[m].reindex(res.index)
                res[m + '_ellipsenet'] = be[m].reindex(res.index)
                res[m + '_diff'] = res[m + '_unet'] - res[m + '_ellipsenet']
                print('  {:<5} {}'.format(m, summary(res[m + '_diff'])))
                print('        largest:', worst(res[m + '_diff']))

    if args.tcd:
        t = pd.read_csv(args.tcd, encoding='utf-8-sig')
        t['case'] = t['xcf_name'].map(case_id)
        t = t.set_index('case')['tcd(mm)']
        res['TCD_biometry_py'] = t.reindex(res.index)
        print('\nbiometry.py TCD - fetal_biometry.py DTC (mm; different masks and method, not only the frame)')
        for src in ('unet', 'ellipsenet'):
            col = 'DTC_' + src
            if col in res:
                d = res['TCD_biometry_py'] - res[col]
                res['TCD_minus_' + col] = d
                print('  vs DTC from {:<10} {}'.format(src, summary(d)))
                print('        largest:', worst(d))

    res.round(3).to_csv(args.out)
    print('\nPer-case table:', args.out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('convert', help='UNet ellipse CSV -> fetal_biometry ellipse CSV')
    c.add_argument('src')
    c.add_argument('dst')
    k = sub.add_parser('compare')
    k.add_argument('--unet', required=True)
    k.add_argument('--ellipsenet', required=True)
    k.add_argument('--bio-unet')
    k.add_argument('--bio-ellipsenet')
    k.add_argument('--tcd')
    k.add_argument('--out', default='ellipse_comparison.csv')
    args = p.parse_args()
    if args.cmd == 'convert':
        convert(args.src, args.dst)
    else:
        compare(args)


if __name__ == '__main__':
    main()
