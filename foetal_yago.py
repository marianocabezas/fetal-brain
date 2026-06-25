import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import skimage.io as skio
from shutil import copyfile
from functools import partial
from scipy.ndimage import binary_fill_holes
from utils import load_xcf
from midline import fit_polynomial_to_mask, MIDLINE_SPEC, CURVE_SPEC, PiecewiseSpec
from midline_models import MidlineSegmenter

# ── curve prediction scope ────────────────────────────────────────────────────
# Switch here to control which structures are predicted. Both CURVE_STRUCTURES
# and the network wrapper use ACTIVE_SPEC — this is the only line to change.
#   MIDLINE_SPEC  — midline only        (4-value target)
#   CURVE_SPEC    — midline + sylvian   (10-value target)
ACTIVE_SPEC = MIDLINE_SPEC

from experiments_yago import SegmentationExperiment
from time import time, strftime
import pandas as pd
import torch
from modelsYago import (
    FCN_ResNet50, FCN_ResNet101,
    DeeplabV3_ResNet50, DeeplabV3_ResNet101,
    DeeplabV3_MobileNet, LRASPP_MobileNet,
    Unet2D, SegFormer,
)


# ── data processing ────────────────────────────────────────────────────────────

def get_xcf_files(xcf_path):
    """Return sorted list of XCF filenames from xcf_path."""
    return sorted(
        [f for f in os.listdir(xcf_path) if f.endswith('.xcf')],
        key=lambda x: int(x.split('.')[0])
    )


def build_segmentation(layer_dict, layer_list):
    """Build a multi-class segmentation mask from a layer dictionary."""
    return np.max([
        (i + 1) * binary_fill_holes(np.array(layer_dict[name])[..., -1] > 0)
        for i, name in enumerate(layer_list[:-2])
        if name in layer_dict
    ], axis=0)


# All known curve structures; filtered by ACTIVE_SPEC to build CURVE_STRUCTURES.
# Each entry must correspond to one section of its spec, in the same order.
ALL_CURVE_STRUCTURES = [
    {'name': 'midline', 'layer': 'midline', 'degree': 1},
    {'name': 'sylvian', 'layer': 'silvio',  'degree': 3},
]
CURVE_STRUCTURES = [
    s for s in ALL_CURVE_STRUCTURES if s['name'] in ACTIVE_SPEC.names
]


def build_curves(layer_dict):
    """
    Extract every curve in CURVE_STRUCTURES and return the concatenated target
    vector laid out per CURVE_SPEC (midline 4 values + sylvian 6 values = 10).

    Each curve is expected in every image; raise if a layer is missing or its
    mask is empty/degenerate so a wrongly-named or empty layer is reported
    rather than silently producing a bad target.
    """
    vectors = []
    for s in CURVE_STRUCTURES:
        key = s['layer']
        if key not in layer_dict:
            raise ValueError(
                "No '{:}' layer (for '{:}'). Present layers: {:}".format(
                    key, s['name'], sorted(layer_dict.keys())
                )
            )
        mask = np.array(layer_dict[key])[..., -1] > 0
        poly = fit_polynomial_to_mask(mask, degree=s['degree'])
        if poly is None:
            raise ValueError(
                "'{:}' layer ('{:}') is empty or degenerate for a degree-{:d} "
                "fit.".format(key, s['name'], s['degree'])
            )
        vectors.append(poly.to_vector())
    return np.concatenate(vectors).astype(np.float32)


def save_debug_plot(dcm_image, final_seg, subj_code, debug_path):
    """Save a side-by-side debug plot of the DICOM image and segmentation overlay."""
    os.makedirs(debug_path, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(20, 40))
    plt.subplot(1, 2, 1)
    plt.imshow(dcm_image, cmap='gray')
    plt.imshow(final_seg, cmap='tab10', vmin=0, alpha=0.50)
    axes[0].set_title('Segmentation (Subject {:>2})'.format(subj_code))
    axes[0].axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(dcm_image, cmap='gray')
    axes[1].set_title('DICOM (Subject {:>2})'.format(subj_code))
    axes[1].axis('off')
    plt.savefig(os.path.join(debug_path, subj_code + '_plot.png'), bbox_inches='tight')
    plt.close(fig)


def process_subject(xcf_f, xcf_path, dicom_path, out_path, layer_list, verbose=False):
    """
    Process a single XCF/DICOM pair:
      - Load XCF layers and DICOM image
      - Build segmentation mask
      - Save segmentation PNG, midline sidecar (.npy) and copy DICOM to out_path
    Returns (subj_code, dcm_image, pixel_array, final_seg, curves, layer_keys)
    or None on failure.
    """
    f_split   = xcf_f.split('.')
    subj_code = '.'.join(f_split[:-1])
    dicom_f   = subj_code + '.dcm'

    try:
        names, data = load_xcf(os.path.join(xcf_path, xcf_f))
        layer_dict  = {
            name.lower(): d_i
            for name, d_i in zip(names, data)
            if '.dcm' not in name
        }
        ds        = dicom.dcmread(os.path.join(dicom_path, dicom_f))
        dcm_image = np.mean(ds.pixel_array, axis=-1)
        final_seg = build_segmentation(layer_dict, layer_list)
        curves    = build_curves(layer_dict)

        os.makedirs(out_path, exist_ok=True)
        skio.imsave(os.path.join(out_path, subj_code + '.png'), final_seg.astype(np.uint8))
        np.save(os.path.join(out_path, subj_code + '_curves.npy'), curves)
        copyfile(os.path.join(dicom_path, dicom_f), os.path.join(out_path, dicom_f))

        if verbose:
            print('{:<6}'.format(subj_code), sorted(layer_dict.keys()), final_seg.shape, dcm_image.shape)
        return subj_code, dcm_image, ds.pixel_array, final_seg, curves, sorted(layer_dict.keys())

    except (IndexError, FileNotFoundError) as e:
        print('ERROR loading', subj_code, ':', e)
        return None


def load_processed_subject(subj_code, out_path, active_spec=None, verbose=False):
    """
    Load an already-processed subject directly from out_path (skip XCF processing).
    Returns (subj_code, dcm_image, pixel_array, final_seg[, curves]) or None on failure.
    Curves are only loaded when active_spec is not None.
    """
    try:
        final_seg = skio.imread(os.path.join(out_path, subj_code + '.png'))
        ds        = dicom.dcmread(os.path.join(out_path, subj_code + '.dcm'))
        dcm_image = np.mean(ds.pixel_array, axis=-1)
        if verbose:
            print('{:<6} loaded from processed'.format(subj_code))
        if active_spec is not None:
            curves = np.load(os.path.join(out_path, subj_code + '_curves.npy'))
            return subj_code, dcm_image, ds.pixel_array, final_seg, curves
        return subj_code, dcm_image, ds.pixel_array, final_seg

    except FileNotFoundError as e:
        print('ERROR loading processed', subj_code, ':', e)
        return None


def run_pipeline(mode, xcf_path, dicom_path, out_path, debug_path, layer_list, active_spec=None, verbose=False):
    """
    Run the fetal data sorting pipeline.

    Parameters
    ----------
    mode : str
        'full'      — process XCF/DICOM files from scratch and save outputs.
        'load_only' — read segmentation PNGs and DICOMs from out_path.
    verbose : bool
        If True, save debug plots and print loading messages for every subject.
    """
    global_dict = {}
    labels      = []

    for xcf_f in get_xcf_files(xcf_path):
        f_split   = xcf_f.split('.')
        sub       = f_split[0]
        subj_code = '.'.join(f_split[:-1])

        if mode == 'full':
            result = process_subject(xcf_f, xcf_path, dicom_path, out_path, layer_list, verbose)
            if result is not None:
                subj_code, dcm_image, pixel_array, final_seg, curves, layer_keys = result
                labels += layer_keys
                global_dict.setdefault(sub, []).append((pixel_array, final_seg, curves))
        elif mode == 'load_only':
            result = load_processed_subject(subj_code, out_path, active_spec=active_spec, verbose=verbose)
            if result is not None:
                if active_spec is not None:
                    subj_code, dcm_image, pixel_array, final_seg, curves = result
                    global_dict.setdefault(sub, []).append((pixel_array, final_seg, curves))
                else:
                    subj_code, dcm_image, pixel_array, final_seg = result
                    global_dict.setdefault(sub, []).append((pixel_array, final_seg))
        else:
            raise ValueError("mode must be 'full' or 'load_only', got '{:}'".format(mode))

        if result is not None and verbose:
            save_debug_plot(dcm_image, final_seg, subj_code, debug_path)

    if labels:
        print('\nAll labels found:', np.unique(labels))

    return global_dict


if __name__ == '__main__':

    xcf_path, dicom_path, debug_path = 'Anotacions 2026/', 'Cerebel_dicom_anonim/', 'outDebug/'
    processed_path = 'ProcessedData/'   # local — images, DICOMs, weights
    results_path   = '/home/yago/Yago Lab Dropbox/medicalImaging/experiments/'  # Dropbox — CSVs only
    mode = 'load_only'  # 'full'

    # ── structure definitions ──────────────────────────────────────────────────
    # Each structure has:
    #   name  : display name used in results
    #   layer : XCF layer key (lowercased)
    #   type  : 'area' -> segmentation class / 'line' -> curve prediction
    #   degree: polynomial degree (only used for 'line' structures)
    structures = [
        {'name': 'cavum',         'layer': 'cavum',         'type': 'area'},
        {'name': 'cerebellum',    'layer': 'cerebel',       'type': 'area'},
        {'name': 'cisterna magna','layer': 'cisterna magna', 'type': 'area'},
        {'name': 'nuchal fold',   'layer': 'plec nucal',    'type': 'area'},
        #{'name': 'midline',       'layer': 'midline',       'type': 'line', 'degree': 1},
        #{'name': 'sylvian',       'layer': 'silvio',        'type': 'line', 'degree': 3},
    ]

    area_structures = [s for s in structures if s['type'] == 'area']
    line_structures = [s for s in structures if s['type'] == 'line']

    layer_list = [s['layer'] for s in area_structures] + [s['layer'] for s in line_structures]
    classes    = ['background'] + [s['name'] for s in area_structures]

    # Build ACTIVE_SPEC from the active line structures (empty -> area-only mode)
    if line_structures:
        ACTIVE_SPEC = PiecewiseSpec(
            [s['degree'] for s in line_structures],
            names=[s['name'] for s in line_structures]
        )
        CURVE_STRUCTURES = [
            {'name': s['name'], 'layer': s['layer'], 'degree': s['degree']}
            for s in line_structures
        ]
    else:
        ACTIVE_SPEC = None

    global_dict = run_pipeline(mode, xcf_path, dicom_path, processed_path, debug_path, layer_list, active_spec=ACTIVE_SPEC, verbose=False)

    networks = [
        #('fcn-resnet101',     'FCN ResNet101',     partial(FCN_ResNet101,        lr=1e-4, pretrained=True)),
        #('deeplab-resnet101', 'DeepLab ResNet101', partial(DeeplabV3_ResNet101,  lr=1e-4, pretrained=True)),
        ('fcn-resnet50',      'FCN ResNet50',      partial(FCN_ResNet50,         lr=1e-4, pretrained=True)),
        #('deeplab-resnet50',  'DeepLab ResNet50',  partial(DeeplabV3_ResNet50,   lr=1e-4, pretrained=True)),
        #('deeplab-mobilenet', 'DeepLab MobileNet', partial(DeeplabV3_MobileNet,  lr=1e-4, pretrained=True)),
        #('lraspp-mobilenet',  'LRASPP MobileNet',  partial(LRASPP_MobileNet,     lr=1e-4, pretrained=True)),
        #('unet2d',            'UNet 2D',           partial(Unet2D,               lr=1e-4)),
        #('segformer-b2',      'SegFormer B2',      partial(SegFormer,            lr=1e-4)),
        #('segformer-b4',      'SegFormer B4',      partial(SegFormer, backbone='nvidia/mit-b4', lr=1e-4)),
        ('segformer-b5',      'SegFormer B5',      partial(SegFormer, backbone='nvidia/mit-b5', lr=1e-4)),
    ]

    all_results = []
    t_total_start = time()
    for net_name, display_name, base_network_f in networks:
        if ACTIVE_SPEC is not None:
            # wrap with curve prediction mouth
            def network_f(_base_f=base_network_f, _spec=ACTIVE_SPEC, **kwargs):
                return MidlineSegmenter(
                    _base_f(**kwargs), spec=_spec, lambda_int=0.01, lambda_midline=1.0
                )
        else:
            # area-only: use the base segmenter directly
            def network_f(_base_f=base_network_f, **kwargs):
                return _base_f(**kwargs)

        print('[{:}] Starting {:}'.format(strftime("%d/%m/%Y - %H:%M:%S"), display_name))
        t_start = time()
        exp = SegmentationExperiment(
            net_name, display_name, 'fetal us segmentation', network_f, global_dict,
            os.path.join(processed_path, 'Weights'),
            os.path.join(processed_path, 'Predictions', net_name),
            classes, n_inputs=3, n_classes=len(classes), epochs = 30, patience=5,
            train_batch=1, test_batch=1, n_seeds=5, verbose=1, save_plots=True,
        )
        _, _, results_df = exp.run(master_seed=42)
        results_df['network'] = display_name
        exp.save_results(results_df, os.path.join(results_path, '{:}_results.csv'.format(net_name)))
        all_results.append(results_df)
        torch.cuda.empty_cache()
        elapsed = time() - t_start
        print('[{:}] Finished {:} in {:.0f}m {:.0f}s'.format(
            strftime("%d/%m/%Y - %H:%M:%S"), display_name, elapsed // 60, elapsed % 60
        ))

    total = time() - t_total_start
    print('\nTotal experiment time: {:.0f}h {:.0f}m {:.0f}s'.format(
        total // 3600, (total % 3600) // 60, total % 60
    ))

    comparison_df = pd.concat(all_results, ignore_index=True)
    comparison_df.to_csv(os.path.join(results_path, 'comparison_results.csv'), index=False)
    print('\n── Comparison summary (global mean row per network) ──')
    summary = comparison_df[(comparison_df['fold'] == 'mean') & (comparison_df['seed'] == 'all')]
    print(summary.set_index('network').drop(columns=['seed', 'fold']).to_string())