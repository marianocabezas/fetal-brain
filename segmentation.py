import os
import csv
import argparse
import numpy as np
import pydicom as dicom
from pathlib import Path
from time import strftime
from tqdm.auto import tqdm
from functools import partial
from shutil import copyfile
from skimage import io as skio
from scipy.ndimage import binary_fill_holes

import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap

import torch
from torch.utils.data import DataLoader

from utils import color_codes, load_xcf
from datasets import FetalDataset
from models import FCN_ResNet50, FCN_ResNet101
from biometrics import compute_tcd


def parse_inputs():
    parser = argparse.ArgumentParser(
        description='Segment ultrasound data'
    )

    # Mode selector
    parser.add_argument(
        '-x', '--xcf-path',
        dest='xcf_path', default='/home/mariano/Datasets/GIMPXCF-Anna/',
        help='Path to the GIMP file annotations.'
    )
    parser.add_argument(
        '-t', '--xcf-test-path',
        dest='test_path', default='/home/mariano/Datasets/GIMPXCF-Test/',
        help='Path to the GIMP file annotations.'
    )
    parser.add_argument(
        '-d', '--dicom-path',
        dest='dicom_path', default='/home/mariano/Datasets/DCM/',
        help='Path to the DICOM image files.'
    )
    parser.add_argument(
        '-o', '--output-path',
        dest='out_path', default='/home/mariano/Datasets/USFoetalSorted-Anna/',
        help='Path to the DICOM image files.'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int, default=50,
        help='Number of epochs for model training.'
    )
    parser.add_argument(
        '-i', '--axes-iterations',
        dest='axes_iterations',
        type=int, default=1000,
        help='Number of epochs for axes estimation.'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=10,
        help='Maximum number of epochs without improvement for model training.'
    )
    parser.add_argument(
        '-l', '--learning-rate',
        dest='learning_rate',
        type=float, default=1e-4,
        help='Learning rate for the axis optimisation.'
    )
    parser.add_argument(
        '-a', '--axes-learning-rate',
        dest='axes_lr',
        type=float, default=5e-2,
        help='Learning rate for the axis optimisation.'
    )
    parser.add_argument(
        '-n', '--init_angles',
        dest='n_init',
        default=32, action='store_true',
        help='Number of axis optimisation tries.'
    )
    parser.add_argument(
        '-s', '--master-seed',
        dest='master_seed',
        type=int, default=42,
        help='Initial seed that controls all pseudo-random number generators.'
    )
    options = vars(parser.parse_args())

    return options


def train_val_split(train_val_ids, data_dict, val_split=0.8):
    train_ids = train_val_ids[:int(len(train_val_ids) * val_split)]
    val_ids = train_val_ids[int(len(train_val_ids) * val_split):]
    training_images = [
        im for sub_id in train_ids for _, im, _ in data_dict[sub_id]
    ]
    training_masks = [
        mask for sub_id in train_ids for _, _, mask in data_dict[sub_id]
    ]
    training_set = FetalDataset(training_images, training_masks)
    validation_images = [
        im for sub_id in val_ids for _, im, _ in data_dict[sub_id]
    ]
    validation_masks = [
        mask for sub_id in val_ids for _, _, mask in data_dict[sub_id]
    ]
    validation_set = FetalDataset(validation_images, validation_masks)
    return training_set, validation_set


def run_segmentation_experiments(
        master_seed, network_name, display_name, experiment_name, network_f,
        data_set, weight_path, maps_path, classes=None, n_folds=5, val_split=0.8,
        epochs=10, patience=5, n_seeds=1, n_inputs=3, n_classes=5,
        train_batch=2, test_batch=2, verbose=1
):
    # Make paths if they did not exist
    for d in [weight_path, maps_path]: Path(d).mkdir(parents=True, exist_ok=True)
    c = color_codes()

    subj_ids = list(data_set.keys())
    subs_x_fold = len(subj_ids) / n_folds

    # Choosing random runs.
    np.random.seed(master_seed)
    seeds = np.random.randint(0, 100000, n_seeds)

    # Main loop to run each independent random experiment.
    dsc_list = []
    class_dsc_list = []
    prediction_dict = {
        im_id: [] for sub_id in subj_ids for im_id, _, _ in data_set[sub_id]
    }
    for test_n, seed in enumerate(seeds):
        if verbose > 1:
            print(
                '{:}[{:}] {:}Starting experiment {:}(seed {:05d})'
                '{:} [{:02d}/{:02d}] {:}for {:} segmentation{:}'.format(
                    c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"), c['g'],
                    c['nc'] + c['y'], seed,
                    c['nc'] + c['c'], test_n + 1, len(seeds),
                    c['nc'] + c['g'],
                    c['b'] + experiment_name + c['nc'] + c['g'], c['nc']
                )
            )

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Dataset is a dictionary where the keys are subject, and
        # for each key a list of tuples is given.
        rand_perm = np.random.permutation(subj_ids).tolist()

        seed_dsc_list = []
        seed_class_dsc_list = []

        for f_i in range(n_folds):
            test_ini = int(round(f_i * subs_x_fold))
            test_end = int(round((f_i + 1) * subs_x_fold))
            test_ids = rand_perm[test_ini:test_end]
            train_val_ids = rand_perm[:test_ini] + rand_perm[test_end:]
            training_set, validation_set = train_val_split(train_val_ids, data_set)

            testing_set = [
                im for sub_id in test_ids for _, im, _ in data_set[sub_id]
            ]

            # The network will only be instantiated with the number of output classes
            # (2 in this notebok). Therefore, networks that need extra parameters
            # (like ViT) will need to be passed as a partial function.
            net = network_f(n_inputs=n_inputs, n_outputs=n_classes)

            # This is a leftover from legacy code. If init is set to True (the default
            # option), a first validation epoch will be run to determine the loss
            # before training.
            net.init = False

            # The number of parameters is only captured for debugging and printing.
            n_param = sum(
                p.numel() for p in net.parameters() if p.requires_grad
            )

            if verbose > 1:
                print(
                    '   {:}Fold {:}{:02d}/{:02d} {:}({:} - {:}[{:,} parameters]{:}){:}'.format(
                        c['g'], c['nc'] + c['c'], f_i + 1, n_folds,
                        c['y'], c['b'] + display_name,
                        c['nc'], n_param, c['y'], c['nc']
                    )
                )
            print(
                '   |_ Training', len(training_set)
            )
            print(
                '   |_ Validation', len(validation_set)
            )
            print(
                '   |_ Testing', len(testing_set), '[{:d} subjects]'.format(len(test_ids)),
            )

            training_loader = DataLoader(
                training_set, train_batch, True
            )
            validation_loader = DataLoader(
                validation_set, test_batch
            )
            model_path = os.path.join(
                weight_path, '{:}-balanced_s{:05d}_f{:01d}.pt'.format(
                    network_name, seed, f_i
                )
            )

            # For efficiency, we only run the code once. If the weights are
            # stored on disk, we do not need to train again.
            try:
                net.load_model(model_path)
            except IOError:
                net.train()
                net.fit(training_loader, validation_loader, epochs=epochs, patience=patience)
                net.save_model(model_path)

            if verbose > 2:
                print(
                    '   {:}Testing <{:02d} samples>{:}'.format(
                        c['g'], len(testing_set), c['nc']
                    )
                )

            # Metric evaluation.
            net.eval()
            with torch.no_grad():
                mosaic_dsc = []
                mosaic_class_dsc = []
                # Intermediate buffers for class metrics.
                # for input_mosaic, mask_i in zip(testing_mosaics, testing_masks):
                im_i = 0
                for sub_id in test_ids:
                    for tst_id, im, mask in data_set[sub_id]:
                        pred_map = net.inference(
                            np.expand_dims(
                                np.moveaxis(im, -1, 0).astype(np.float32),
                                axis=0
                            )
                        )[0]

                        pred_y = np.argmax(pred_map, axis=0).astype(np.uint8)
                        y = mask.astype(np.uint8)
                        intersection = np.stack([
                            2 * np.sum(np.logical_and(pred_y == lab, y == lab))
                            for lab in range(n_classes)
                        ])
                        card_pred_y = np.stack([
                            np.sum(pred_y == lab) for lab in range(n_classes)
                        ])
                        card_y = np.stack([
                            np.sum(y == lab) for lab in range(n_classes)
                        ])
                        dsc_k = intersection / (card_pred_y + card_y)
                        dsc = np.nanmean(dsc_k)
                        dsc_std = np.nanstd(dsc_k)
                        mosaic_dsc.append(dsc)
                        mosaic_class_dsc.append(dsc_k.tolist())

                        prediction_dict[tst_id].append(pred_y)

                        dsc = np.nanmean(mosaic_dsc, axis=0)
                        class_dsc = np.nanmean(mosaic_class_dsc, axis=0)
                        class_std = np.nanstd(mosaic_class_dsc, axis=0)

            if verbose > 2:
                print(
                    '   {:}Fold {:}{:02d}/{:02d} {:}DSC{:} (seed {:05d}){:} [{:02d}/{:02d}] {:}'
                    '{:5.3f} ± {:5.3f}{:}'.format(
                        c['g'], c['nc'] + c['c'], f_i + 1, n_folds, c['g'],
                                c['nc'] + c['y'], seed, c['nc'] + c['c'], test_n + 1, len(seeds),
                                c['nc'] + c['b'], dsc, dsc_std, c['nc']
                    )
                )

                class_dsc_s = ', '.join([
                    '{:} {:5.3f} ± {:5.3f}'.format(k, dsc_k, std_k)
                    for k, dsc_k, std_k in zip(classes, class_dsc, class_std)
                ])
                print(
                    '   {:}Fold {:}{:02d}/{:02d} {:}Class DSC{:} (seed {:05d}){:} [{:02d}/{:02d}] {:}'.format(
                        c['g'], c['nc'] + c['c'], f_i + 1, n_folds, c['g'],
                                c['nc'] + c['y'], seed, c['nc'] + c['c'], test_n + 1, len(seeds),
                                c['nc'] + c['b'] + class_dsc_s + c['nc']
                    )
                )
            elif verbose > 1:
                print(
                    '{:}Seed {:05d} {:} [{:,} parameters] '
                    '{:}[{:02d}/{:02d}] {:} {:5.3f}{:}'.format(
                        c['y'], seed, c['b'] + display_name + c['nc'], n_param,
                        c['c'], test_n + 1, len(seeds),
                                      c['nc'] + c['g'] + c['b'] + experiment_name + c['nc'] + c['b'],
                        dsc, c['nc']
                    )
                )
            elif verbose > 0:
                print(
                    '{:}Seed {:05d} {:} [{:,} parameters] '
                    '{:}[{:02d}/{:02d}] {:} {:5.3f}{:}'.format(
                        c['y'], seed, c['b'] + display_name + c['nc'], n_param,
                        c['c'], test_n + 1, len(seeds),
                                      c['nc'] + c['g'] + c['b'] + experiment_name + c['nc'] + c['b'],
                        dsc, c['nc']
                    ), end='\r'
                )
            net = None
            seed_dsc_list.append(mosaic_dsc)
            seed_class_dsc_list.append(mosaic_class_dsc)

        if verbose > 0:
            seed_mean_dsc = np.nanmean(np.concatenate(seed_dsc_list))
            seed_std_dsc = np.nanstd(np.concatenate(seed_dsc_list))
            seed_class_mean_dsc = np.nanmean(
                np.concatenate(seed_class_dsc_list, axis=0), axis=0
            )
            seed_class_std_dsc = np.nanstd(
                np.concatenate(seed_class_dsc_list, axis=0), axis=0
            )

            print(
                '   {:}Seed {:05d} - mean DSC{:} {:5.3f} ± {:5.3f}{:}'.format(
                    c['y'] + c['b'] + display_name + c['nc'] + c['g'],
                    seed, c['nc'] + c['b'], seed_mean_dsc, seed_std_dsc, c['nc']
                )
            )
            class_dsc_s = ', '.join([
                '{:} {:5.3f} ± {:5.3f}'.format(k, dsc_k, std_k)
                for k, dsc_k, std_k in zip(
                    classes, seed_class_mean_dsc, seed_class_std_dsc
                )
            ])
            print(
                '   {:}Seed {:05d} - mean class DSC {:}'.format(
                    c['y'] + c['b'] + display_name + c['nc'] + c['g'],
                    seed, c['nc'] + c['b'] + class_dsc_s + c['nc']
                )
            )
        dsc_list.append(seed_mean_dsc)
        class_dsc_list.append(seed_class_mean_dsc)

    # Metrics for all the runs.
    if verbose > 0:
        print(
            '{:}[{:}] {:} Overall mean DSC{:} {:5.3f}{:}'.format(
                c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"),
                c['nc'] + c['y'] + c['b'] + display_name + c['nc'] + c['g'],
                c['nc'] + c['b'], np.nanmean(dsc_list), c['nc']
            )
        )
        class_dsc_s = ', '.join([
            '{:} {:5.3f}'.format(k, dsc_k)
            for k, dsc_k in zip(
                classes, np.nanmean(class_dsc_list, axis=0)
            )
        ])
        print(
            '{:}[{:}] {:} Mean class DSC {:}'.format(
                c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"),
                c['nc'] + c['y'] + c['b'] + display_name + c['nc'] + c['g'],
                c['nc'] + c['b'] + class_dsc_s + c['nc']
            )
        )

    return dsc_list, class_dsc_list, prediction_dict


def main():
    # Init
    options = parse_inputs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xcf_path = options['xcf_path']
    dicom_path = options['dicom_path']
    out_path = options['out_path']
    test_path = options['test_path']
    weight_path = os.path.join(out_path, 'Weights')
    Path(weight_path).mkdir(parents=True, exist_ok=True)
    metrics_path = os.path.join(out_path, 'Biometrics')
    Path(metrics_path).mkdir(parents=True, exist_ok=True)
    lr = options['learning_rate']
    axes_lr = options['axes_lr']
    axes_iterations = options['axes_iterations']
    n_epochs = options['epochs']
    patience = options['patience']
    master_seed = options['master_seed']
    network_name = 'fcn-resnet50'
    display_name = 'FCN ResNet50'
    c = color_codes()

    train_batch = 1
    test_batch = 1
    n_seeds = 5
    name = 'fetal us'

    layer_list = [
        'cavum', 'cerebel', 'cisterna magna', 'plec nucal', 'midline', 'silvio'
    ]
    layer_names = [
        'cavum', 'cerebellum', 'cisterna magna', 'nuchal fold', 'midline', 'sylvian'
    ]
    n_classes = len(layer_list) - 1

    xcf_files = sorted(
        [f for f in os.listdir(xcf_path) if f.endswith('.xcf')],
        key=lambda x: int(x.split('.')[0])
    )
    xcf_loop = tqdm(
        enumerate(xcf_files), total=len(list(xcf_files)), leave=False,
        desc='XCF and DCM file reading',
    )

    global_dict = {}
    subjects = []

    for i, xcf_f in xcf_loop:
        f_split = xcf_f.split('.')
        sub = f_split[0]
        if sub not in subjects:
            subjects.append(sub)
            global_dict[sub] = []
        im_id = '.'.join(f_split[:-1])
        dicom_f = im_id + '.dcm'
        dicom_filepath = os.path.join(dicom_path, dicom_f)
        try:
            xcf_filepath = os.path.join(xcf_path, xcf_f)
            names, data = load_xcf(xcf_filepath)
            layer_dict = {
                name.lower(): d_i
                for name, d_i in zip(names, data)
                if '.dcm' not in name
            }
            dcm_data = dicom.dcmread(dicom_filepath)

            #### Data preparation
            # > Area segmentations
            mask = np.max([
                (i + 1) * binary_fill_holes(np.array(layer_dict[name])[..., -1] > 0)
                for i, name in enumerate(layer_list[:-2])
            ], axis=0)

            #seg_filepath = os.path.join(out_path, im_id + '.png')
            #skio.imsave(seg_filepath, mask.astype(np.uint8))
            #dicom_outpath = os.path.join(out_path, dicom_f)
            #copyfile(dicom_filepath, dicom_outpath)

            global_dict[sub].append(
                (im_id, dcm_data.pixel_array, mask)
            )
        except (IndexError, FileNotFoundError) as e:
            print('ERROR loading', im_id, e)

    np.random.seed(master_seed)
    torch.manual_seed(master_seed)

    classes = ['background'] + layer_names[:-2]

    '''# The experiments are run next. We capture some warnings related to
    # image loading to clean the debugging console.
    # FCN ResNet50
    fcn50_dsc, fcn50_k_dsc, fcn50_pred = run_segmentation_experiments(
        master_seed, network_name, display_name,
        name, partial(FCN_ResNet50, lr=1e-4, pretrained=True),
        global_dict, weight_path, pred_path,
        classes, n_inputs=3, n_classes=n_classes,
        epochs=n_epochs, patience=patience,
        train_batch=train_batch, test_batch=test_batch,
        n_seeds=n_seeds, verbose=3
    )

    np.random.seed(master_seed)
    seeds = np.random.randint(0, 100000, n_seeds)
    print(
        f'Image ID    | Expert  |',
        f'S{seeds[0]:05d}  |'
        f'S{seeds[1]:05d}  |'
        f'S{seeds[2]:05d}  |'
        f'S{seeds[3]:05d}  |'
        f'S{seeds[4]:05d}  |'
    )

    tcd_csv = os.path.join(out_path, 'tcd_seed_results.csv')
    with open(tcd_csv, 'w') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow([
            'image_id', 'anna',
            f's{seeds[0]:05d}',
            f's{seeds[1]:05d}',
            f's{seeds[2]:05d}',
            f's{seeds[3]:05d}',
            f's{seeds[4]:05d}',
        ])
        for sub, sub_data in global_dict.items():
            for im_id, _, gt_mask in sub_data:
                dicom_f = im_id + '.dcm'
                dicom_filepath = os.path.join(dicom_path, dicom_f)
                dcm_data = dicom.dcmread(dicom_filepath)
                spacing = dcm_data[0x00280030].value

                pred_list = fcn50_pred[im_id]
                gt_tcd, _, _ = compute_tcd(
                    gt_mask == 2, spacing, lr=axes_lr,
                    n_iters=axes_iterations
                )
                pred_tcd, _, _ = [
                    compute_tcd(
                        pred_i == 2, spacing, lr=axes_lr,
                    n_iters=axes_iterations
                    ) for pred_i in pred_list
                ]

                print(
                    f'Image {im_id:>5} | {gt_tcd:5.2f}mm |',
                    ' | '.join(f'{pred_i:5.2f}mm' for pred_i in pred_tcd),
                    '|'
                )
                writer.writerow([im_id, gt_tcd] + [pred_i for pred_i in pred_tcd])'''

    experts = sorted(os.listdir(test_path))

    e_files = np.array(sorted(
        os.listdir(os.path.join(test_path, experts[0])),
        key=lambda x: float('.'.join(x.split('.')[:-1]))
    ))

    for expert in experts[1:]:
        e_path = os.path.join(test_path, expert)
        e_mask = np.isin(e_files, os.listdir(e_path))
        e_files = e_files[e_mask]

    e_files = e_files.tolist()
    test_subs = [
        f.split('.')[0] for f in e_files
    ]

    train_val_ids = [
        sub_id for sub_id in global_dict.keys() if sub_id not in test_subs
    ]
    training_set, validation_set = train_val_split(train_val_ids, global_dict)

    np.random.seed(master_seed)
    torch.manual_seed(master_seed)

    partial(FCN_ResNet50, )
    test_net = FCN_ResNet50(
        lr=lr, pretrained=True, n_inputs=3, n_outputs=n_classes
    )

    # This is a leftover from legacy code. If init is set to True (the default option),
    # a first validation epoch will be run to determine the loss before training.
    test_net.init = False

    # The number of parameters is only captured for debugging and printing.
    n_param = sum(
        p.numel() for p in test_net.parameters() if p.requires_grad
    )

    print(
        '   {:}Final training {:}({:} - {:}[{:,} parameters]{:}){:}'.format(
            c['g'], c['y'], c['b'] + display_name,
            c['nc'], n_param, c['y'], c['nc']
        )
    )
    print('   |_ Training', len(training_set), 'images')
    print('   |_ Validation', len(validation_set), 'images')
    print('   |_ Testing', len(test_subs), 'images')

    training_loader = DataLoader(
        training_set, train_batch, True
    )
    validation_loader = DataLoader(
        validation_set, test_batch
    )
    model_path = os.path.join(
        weight_path, '{:}-_s{:05d}_final.pt'.format(network_name, master_seed)
    )

    # For efficiency, we only run the code once. If the weights are
    # stored on disk, we do not need to train again.
    try:
        test_net.load_model(model_path)
    except IOError:
        test_net.train()
        test_net.fit(
            training_loader, validation_loader,
            epochs=n_epochs, patience=patience
        )
        test_net.save_model(model_path)


    print(
        'Image ID    |   Pred   |', ' | '.join([
            f' {e.lower():<6} ' for e in experts
        ]), '|'
    )
    tcd_csv = os.path.join(out_path, 'tcd_comparison_results.csv')
    header_info = ['image_id', network_name] + [
        e.lower() for e in experts
    ]

    test_net.eval()
    all_tcd_dict = {
        e_id: []
        for e_id in [network_name] + experts
    }
    with open(tcd_csv, 'w') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(header_info)
        for tst_file in e_files:
            im_id = '.'.join(tst_file.split('.')[:-1])
            dicom_f = im_id + '.dcm'
            dicom_filepath = os.path.join(dicom_path, dicom_f)
            dcm_data = dicom.dcmread(dicom_filepath)
            spacing = dcm_data[0x00280030].value
            im = np.expand_dims(
                np.moveaxis(dcm_data.pixel_array, -1, 0).astype(np.float32),
                axis=0
            )

            with torch.no_grad():
                pred_mask = np.argmax(
                    test_net.inference(im)[0], axis=0
                ).astype(np.uint8)
            pred_cerebellum = pred_mask == 2
            pred_tcd, pred_tcd_axis, _ = compute_tcd(
                pred_cerebellum, spacing, lr=axes_lr,
                n_iters=axes_iterations
            )
            all_tcd_dict[network_name].append(pred_tcd)

            plt.figure(figsize=(20, 30))
            plt.imshow(dcm_data.pixel_array, cmap='gray')
            plt.imshow(
                pred_cerebellum, vmin=0, alpha=0.50,
                cmap=ListedColormap(
                    [[0, 0, 0, 0], 'orange'], name='foetal_cereb'
                ),
            )
            plt.plot(pred_tcd_axis[0], pred_tcd_axis[1], color='teal')
            plt.gca().set_title(
                f'{display_name} segmentation (ID: {im_id}) [TCD: {pred_tcd:.3f} mm]'
            )
            plt.gca().axis("off")
            plt.savefig(
                os.path.join(
                    metrics_path, f'{im_id}_{network_name}_cerebellum.png'
                )
            )
            plt.close()

            e_tcds = []
            for expert in experts:
                xcf_filepath = os.path.join(test_path, expert, tst_file)
                names, data = load_xcf(xcf_filepath)
                layer_dict = {
                    name.lower(): d_i
                    for name, d_i in zip(names, data)
                    if '.dcm' not in name
                }

                #### Data preparation
                # > Area segmentations
                e_mask = np.max([
                    (i + 1) * binary_fill_holes(np.array(layer_dict[name])[..., -1] > 0)
                    for i, name in enumerate(layer_list[:-2])
                ], axis=0)
                try:
                    e_cerebellum = e_mask == 2
                    e_tcd, e_tcd_axis, _ = compute_tcd(
                        e_cerebellum, spacing, lr=axes_lr,
                        n_iters=axes_iterations
                    )
                    e_tcds.append(e_tcd)
                    all_tcd_dict[expert].append(e_tcd)
                    plt.figure(figsize=(20, 30))
                    plt.imshow(dcm_data.pixel_array, cmap='gray')
                    plt.imshow(
                        e_cerebellum, vmin=0, alpha=0.50,
                        cmap=ListedColormap(
                            [[0, 0, 0, 0], 'orange'], name='foetal_cereb'
                        ),
                    )
                    plt.plot(e_tcd_axis[0], e_tcd_axis[1], color='teal')
                    plt.gca().set_title(
                        f'{expert} segmentation (ID: {im_id}) [TCD: {e_tcd:.3f} mm]'
                    )
                    plt.gca().axis("off")
                    plt.savefig(
                        os.path.join(
                            metrics_path, f'{im_id}_{expert}_cerebellum.png'
                        )
                    )
                    plt.close()
                except Exception as e:
                    print(f'Error with image {im_id} for expert {expert}', e)

            print(
                f'Image {im_id:>5} | {pred_tcd:6.3f}mm |',
                ' | '.join(f'{e_i:6.3f}mm' for e_i in e_tcds),
                '|'
            )
            writer.writerow([im_id, pred_tcd] + [e_i for e_i in e_tcds])
        print(
            f'Mean        |',
            ' | '.join([
                f'{np.mean(tcd_list):6.3f}mm'
                for tcd_list in all_tcd_dict.values()
            ]),
            '|'
        )


if __name__ == '__main__':
    main()