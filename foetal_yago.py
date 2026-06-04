import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import skimage.io as skio
from shutil import copyfile
from functools import partial
from scipy.ndimage import binary_fill_holes

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
      - Save segmentation PNG and copy DICOM to out_path
    Returns (subj_code, dcm_image, pixel_array, final_seg, layer_keys) or None on failure.
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

        os.makedirs(out_path, exist_ok=True)
        skio.imsave(os.path.join(out_path, subj_code + '.png'), final_seg.astype(np.uint8))
        copyfile(os.path.join(dicom_path, dicom_f), os.path.join(out_path, dicom_f))

        if verbose:
            print('{:<6}'.format(subj_code), sorted(layer_dict.keys()), final_seg.shape, dcm_image.shape)
        return subj_code, dcm_image, ds.pixel_array, final_seg, sorted(layer_dict.keys())

    except (IndexError, FileNotFoundError) as e:
        print('ERROR loading', subj_code, ':', e)
        return None


def load_processed_subject(subj_code, out_path, verbose=False):
    """
    Load an already-processed subject directly from out_path (skip XCF processing).
    Returns (subj_code, dcm_image, pixel_array, final_seg) or None on failure.
    """
    try:
        final_seg = skio.imread(os.path.join(out_path, subj_code + '.png'))
        ds        = dicom.dcmread(os.path.join(out_path, subj_code + '.dcm'))
        dcm_image = np.mean(ds.pixel_array, axis=-1)
        if verbose:
            print('{:<6} loaded from processed'.format(subj_code))
        return subj_code, dcm_image, ds.pixel_array, final_seg

    except FileNotFoundError as e:
        print('ERROR loading processed', subj_code, ':', e)
        return None


def run_pipeline(mode, xcf_path, dicom_path, out_path, debug_path, layer_list, verbose=False):
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
                subj_code, dcm_image, pixel_array, final_seg, layer_keys = result
                labels += layer_keys
                global_dict.setdefault(sub, []).append((pixel_array, final_seg))
        elif mode == 'load_only':
            result = load_processed_subject(subj_code, out_path, verbose)
            if result is not None:
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

    xcf_path, dicom_path, out_path, debug_path = 'Anotacions 2026/', 'Cerebel_dicom_anonim/', 'ProcessedData/', 'outDebug/'
    layer_list  = ['cavum', 'cerebel', 'cisterna magna', 'plec nucal', 'midline', 'silvio']
    layer_names = ['cavum', 'cerebellum', 'cisterna magna', 'nuchal fold', 'midline', 'sylvian']
    mode    = 'load_only'  # 'full'
    classes = ['background'] + layer_names[:-2]

    global_dict = run_pipeline(mode, xcf_path, dicom_path, out_path, debug_path, layer_list, verbose=False)

    # Each entry: (network_name, display_name, network_f)
    networks = [
        ('fcn-resnet101',     'FCN ResNet101',     partial(FCN_ResNet101,        lr=1e-4, pretrained=True)),
        ('deeplab-resnet101', 'DeepLab ResNet101', partial(DeeplabV3_ResNet101,  lr=1e-4, pretrained=True)),
        ('fcn-resnet50',      'FCN ResNet50',      partial(FCN_ResNet50,         lr=1e-4, pretrained=True)),
        ('deeplab-resnet50',  'DeepLab ResNet50',  partial(DeeplabV3_ResNet50,   lr=1e-4, pretrained=True)),
        ('deeplab-mobilenet', 'DeepLab MobileNet', partial(DeeplabV3_MobileNet,  lr=1e-4, pretrained=True)),
        ('lraspp-mobilenet',  'LRASPP MobileNet',  partial(LRASPP_MobileNet,     lr=1e-4, pretrained=True)),
        ('unet2d',            'UNet 2D',           partial(Unet2D,               lr=1e-4)),
        ('segformer-b2',      'SegFormer B2',      partial(SegFormer,           lr=1e-4)),
    ]

    all_results = []
    for net_name, display_name, network_f in networks:
        exp = SegmentationExperiment(
            net_name, display_name, 'fetal us segmentation', network_f, global_dict,
            os.path.join(out_path, 'Weights'), os.path.join(out_path, 'Predictions', net_name),
            classes, n_inputs=3, n_classes=len(classes), epochs=10, patience=5,
            train_batch=1, test_batch=1, n_seeds=5, verbose=1,
        )
        _, _, results_df = exp.run(master_seed=42)
        results_df['network'] = display_name
        exp.save_results(results_df, os.path.join(out_path, '{:}_results.csv'.format(net_name)))
        all_results.append(results_df)
        torch.cuda.empty_cache()

    comparison_df = pd.concat(all_results, ignore_index=True)
    comparison_df.to_csv(os.path.join(out_path, 'comparison_results.csv'), index=False)
    print('\n── Comparison summary (global mean row per network) ──')
    summary = comparison_df[(comparison_df['fold'] == 'mean') & (comparison_df['seed'] == 'all')]
    print(summary.set_index('network').drop(columns=['seed', 'fold']).to_string())