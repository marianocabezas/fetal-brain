import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from time import strftime
from torch.utils.data import DataLoader

from datasets import FetalDataset


class SegmentationExperiment:
    """
    Runs cross-validated segmentation experiments over multiple random seeds.
    Supports any network that follows the BaseModel interface.
    """

    def __init__(
        self, network_name, display_name, experiment_name, network_f,
        data_set, weight_path, maps_path, classes,
        n_folds=5, val_split=0.8, epochs=10, patience=5,
        n_seeds=1, n_inputs=3, n_classes=5,
        train_batch=2, test_batch=2, verbose=1, save_plots=True
    ):
        self.network_name    = network_name
        self.display_name    = display_name
        self.experiment_name = experiment_name
        self.network_f       = network_f
        self.data_set        = data_set
        self.weight_path     = weight_path
        self.maps_path       = maps_path
        self.classes         = classes
        self.n_folds         = n_folds
        self.val_split       = val_split
        self.epochs          = epochs
        self.patience        = patience
        self.n_seeds         = n_seeds
        self.n_inputs        = n_inputs
        self.n_classes       = n_classes
        self.train_batch     = train_batch
        self.test_batch      = test_batch
        self.verbose         = verbose
        self.save_plots      = save_plots

        for d in [weight_path, maps_path]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _split_fold(self, rand_perm, f_i):
        """Return train, val, test subject id lists for a given fold."""
        n           = len(rand_perm)
        subs_x_fold = n / self.n_folds
        test_ini    = int(round(f_i * subs_x_fold))
        test_end    = int(round((f_i + 1) * subs_x_fold))
        test_ids      = rand_perm[test_ini:test_end]
        train_val_ids = rand_perm[:test_ini] + rand_perm[test_end:]
        split         = int(len(train_val_ids) * self.val_split)
        return train_val_ids[:split], train_val_ids[split:], test_ids

    def _make_datasets(self, train_ids, val_ids, test_ids):
        """Build FetalDataset objects for each split."""
        def collect(ids):
            images = [im   for s in ids for im,   _ in self.data_set[s]]
            masks  = [mask for s in ids for _,  mask in self.data_set[s]]
            return FetalDataset(images, masks)
        return collect(train_ids), collect(val_ids), collect(test_ids)

    def _train_or_load(self, net, seed, f_i, training_loader, validation_loader):
        """Load weights from disk if available, otherwise train and save."""
        import sys, io
        model_path = os.model_path = os.path.join(self.weight_path, '{:}-balanced_s{:05d}_f{:01d}_e{:02d}.pt'.format(self.network_name, seed, f_i, self.epochs))
        try:
            net.load_model(model_path)
        except IOError:
            net.train()
            sys.stdout = io.StringIO()
            try:
                net.fit(training_loader, validation_loader, epochs=self.epochs, patience=self.patience)
            except torch.cuda.OutOfMemoryError:
                sys.stdout = sys.__stdout__
                print('OOM: {:} skipped'.format(self.network_name))
                torch.cuda.empty_cache()
                return
            finally:
                sys.stdout = sys.__stdout__
            net.save_model(model_path)





    def _save_prediction_plot(self, input_mosaic, mask_i, pred_map, tst_i, seed, f_i):
        """Save a side-by-side ground truth vs prediction plot to maps_path."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 40))
        plt.subplot(1, 2, 1)
        plt.imshow(input_mosaic[0, ...], cmap='gray')
        plt.imshow(mask_i, cmap='tab10', vmin=0, alpha=0.50)
        axes[0].set_title('Ground truth (ID: {:02d} / seed {:} / fold {:02d})'.format(tst_i, seed, f_i))
        axes[0].axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(input_mosaic[0, ...], cmap='gray')
        plt.imshow(np.argmax(pred_map, axis=0), cmap='tab10', vmin=0, alpha=0.50)
        axes[1].set_title('Prediction (ID: {:02d} / seed {:} / fold {:02d})'.format(tst_i, seed, f_i))
        axes[1].axis('off')
        fname = 'pred_s{:05d}_f{:02d}_i{:03d}.png'.format(seed, f_i, tst_i)
        plt.savefig(os.path.join(self.maps_path, fname), bbox_inches='tight')
        plt.close(fig)

    def _compute_dsc(self, net, testing_set, seed, f_i):
        """Run inference on the test set, optionally save prediction plots, return DSC."""
        mosaic_dsc, mosaic_class_dsc = [], []
        net.eval()
        with torch.no_grad():
            for tst_i, (input_mosaic, mask_i) in enumerate(testing_set):
                pred_map = net.inference(
                    np.expand_dims(input_mosaic.astype(np.float32), axis=0)
                )[0]
                pred_y = np.argmax(pred_map, axis=0).astype(np.uint8)
                y      = mask_i.astype(np.uint8)
                intersection = np.stack([
                    2 * np.sum(np.logical_and(pred_y == lab, y == lab))
                    for lab in range(self.n_classes)
                ])
                card_pred_y = np.stack([np.sum(pred_y == lab) for lab in range(self.n_classes)])
                card_y      = np.stack([np.sum(y == lab)      for lab in range(self.n_classes)])
                dsc_k = intersection / (card_pred_y + card_y)
                mosaic_dsc.append(np.nanmean(dsc_k))
                mosaic_class_dsc.append(dsc_k.tolist())
                if self.save_plots:
                    self._save_prediction_plot(input_mosaic, mask_i, pred_map, tst_i, seed, f_i)
        return mosaic_dsc, mosaic_class_dsc

    def _print_fold_header(self, f_i, n_param, n_train, n_val, n_test):
        print('   Fold {:02d}/{:02d} ({:} - [{:,} parameters])'.format(
            f_i + 1, self.n_folds, self.display_name, n_param
        ))
        print('   |_ Training', n_train)
        print('   |_ Validation', n_val)
        print('   |_ Testing', n_test)

    def _print_fold_results(self, f_i, seed, test_n, n_seeds, dsc, class_dsc):
        print('   Fold {:02d}/{:02d} DSC (seed {:05d}) [{:02d}/{:02d}] {:.3f}'.format(
            f_i + 1, self.n_folds, seed, test_n + 1, n_seeds, dsc
        ))
        class_dsc_s = ', '.join('{:} {:.3f}'.format(k, v) for k, v in zip(self.classes, class_dsc))
        print('   Fold {:02d}/{:02d} Class DSC (seed {:05d}) [{:02d}/{:02d}] {:}'.format(
            f_i + 1, self.n_folds, seed, test_n + 1, n_seeds, class_dsc_s
        ))

    def _print_summary(self, mean_dsc, mean_class_dsc):
        class_dsc_s = ', '.join('{:} {:.3f}'.format(k, v) for k, v in zip(self.classes, mean_class_dsc))
        print('[{:}] {:} Mean DSC {:.3f}'.format(strftime("%d/%m/%Y - %H:%M:%S"), self.display_name, mean_dsc))
        print('[{:}] {:} Mean class DSC {:}'.format(strftime("%d/%m/%Y - %H:%M:%S"), self.display_name, class_dsc_s))

    def _build_dataframe(self, records):
        """
        Build a tidy DataFrame from the per-fold records collected during run().
        Columns: seed, fold, mean_dsc, <class>_dsc for each class.
        A summary row per seed and a global summary row are appended at the end.
        """
        df = pd.DataFrame(records)
        seed_summary = (
            df.groupby('seed').mean(numeric_only=True)
              .reset_index()
              .assign(fold='mean')
        )
        global_summary = (
            df.mean(numeric_only=True)
              .to_frame().T
              .assign(seed='all', fold='mean')
        )
        return pd.concat([df, seed_summary, global_summary], ignore_index=True)

    def run(self, master_seed):
        """
        Run all seeds and folds.
        Returns (dsc_list, class_dsc_list, results_df).
        """
        subj_ids = list(self.data_set.keys())
        np.random.seed(master_seed)
        seeds = np.random.randint(0, 100000, self.n_seeds)

        dsc_list, class_dsc_list = [], []
        records = []

        for test_n, seed in enumerate(seeds):
            print('[{:}] Starting experiment (seed {:05d}) [{:02d}/{:02d}] for {:}'.format(
                strftime("%d/%m/%Y - %H:%M:%S"), seed, test_n + 1, len(seeds), self.experiment_name
            ))
            np.random.seed(seed)
            torch.manual_seed(seed)
            rand_perm = np.random.permutation(subj_ids).tolist()

            seed_dsc_list, seed_class_dsc_list = [], []

            for f_i in range(self.n_folds):
                train_ids, val_ids, test_ids = self._split_fold(rand_perm, f_i)
                train_set, val_set, test_set = self._make_datasets(train_ids, val_ids, test_ids)

                net      = self.network_f(n_inputs=self.n_inputs, n_outputs=self.n_classes)
                net.init = False
                n_param  = sum(p.numel() for p in net.parameters() if p.requires_grad)

                self._print_fold_header(f_i, n_param, len(train_set), len(val_set), len(test_set))

                training_loader   = DataLoader(train_set, self.train_batch, shuffle=True)
                validation_loader = DataLoader(val_set, self.test_batch)

                self._train_or_load(net, seed, f_i, training_loader, validation_loader)

                print('   Testing <{:d} samples>'.format(len(test_set)))
                mosaic_dsc, mosaic_class_dsc = self._compute_dsc(net, test_set, seed, f_i)
                net = None
                torch.cuda.empty_cache()

                dsc       = np.nanmean(mosaic_dsc)
                class_dsc = np.nanmean(mosaic_class_dsc, axis=0)
                self._print_fold_results(f_i, seed, test_n, len(seeds), dsc, class_dsc)

                record = {'seed': seed, 'fold': f_i + 1, 'mean_dsc': dsc}
                record.update({'{:}_dsc'.format(k): v for k, v in zip(self.classes, class_dsc)})
                records.append(record)

                self._build_dataframe(records).to_csv(
                    os.path.join(self.maps_path, 'results_partial.csv'), index=False
                )

                seed_dsc_list.append(mosaic_dsc)
                seed_class_dsc_list.append(mosaic_class_dsc)

            seed_mean_dsc       = np.nanmean(np.concatenate(seed_dsc_list))
            seed_class_mean_dsc = np.nanmean(np.concatenate(seed_class_dsc_list, axis=0), axis=0)
            self._print_summary(seed_mean_dsc, seed_class_mean_dsc)

            dsc_list.append(seed_mean_dsc)
            class_dsc_list.append(seed_class_mean_dsc)

        self._print_summary(np.nanmean(dsc_list), np.nanmean(class_dsc_list, axis=0))

        results_df = self._build_dataframe(records)
        return dsc_list, class_dsc_list, results_df

    def save_results(self, results_df, path):
        """Save the results DataFrame to a CSV file."""
        results_df.to_csv(path, index=False)
        print('Results saved to', path)