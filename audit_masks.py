"""
audit_masks.py
Diagnostic tool for the fetal brain ultrasound segmentation pipeline.

Run this alongside foetal_yago.py (same folder, same conda env) to:
  1. List the raw layer names found in every XCF file, and flag any that
     don't match an expected area/curve layer name or a known alias in
     LAYER_NAME_ALIASES. This catches typos like 'cerbel' vs 'cerebel'
     BEFORE they silently drop a structure from the mask.
  2. Save a per-class mask visualization for every subject, so you can
     visually confirm each expected layer actually painted something
     (an all-black panel for a given class = that layer was empty,
     missing, or misspelled for that subject).

Usage:
    python audit_masks.py

Output:
    AuditOutput/
        layer_report.txt          <- text summary of every file's layers
        masks/<subj_code>_masks.png   <- one panel per class, per subject
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — avoids slow/blocking GUI rendering
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes

from utils import load_xcf

# ── import the same config used by foetal_yago.py so this stays in sync ──────
# Adjust these imports/paths if your foetal_yago.py defines them differently.
try:
    from foetal_yago import (
        LAYER_NAME_ALIASES, normalize_layer_dict, get_xcf_files,
    )
except ImportError:
    # Fallback minimal copies, in case foetal_yago.py hasn't been updated yet.
    LAYER_NAME_ALIASES = {'cerbel': 'cerebel'}

    def normalize_layer_dict(layer_dict):
        for alias, canonical in LAYER_NAME_ALIASES.items():
            if alias in layer_dict:
                if canonical in layer_dict:
                    raise ValueError(
                        f"Both '{alias}' and '{canonical}' present — ambiguous."
                    )
                layer_dict[canonical] = layer_dict.pop(alias)
        return layer_dict

    def get_xcf_files(xcf_path):
        return sorted(
            [f for f in os.listdir(xcf_path) if f.endswith('.xcf')],
            key=lambda x: int(x.split('.')[0])
        )


# ── EDIT THESE to match your current foetal_yago.py __main__ config ─────────
XCF_PATH = 'Anotacions 2026AnnaPART/'

EXPECTED_AREA_LAYERS = ['cavum', 'cerebel', 'cisterna magna', 'plec nucal']
EXPECTED_CURVE_LAYERS = []   # e.g. ['midline', 'silvio'] if line structures are active

OUT_DIR = 'AuditOutput'
# ──────────────────────────────────────────────────────────────────────────


def layer_pixel_count(layer_data):
    """Pixel count of a layer's painted region: alpha/last channel > 0, then
    fill holes. A present-but-unpainted layer returns 0 (the 'empty layer'
    case that produces an empty-both class at DSC time)."""
    return int(binary_fill_holes(np.array(layer_data)[..., -1] > 0).sum())



def audit_layers(xcf_path):
    """
    Read every XCF file's raw layer names and compare against what's expected.
    Returns (report_lines, per_subject_layer_dicts) where per_subject_layer_dicts
    maps subj_code -> {canonical_layer_name: mask_layer_data}.
    """
    expected = set(EXPECTED_AREA_LAYERS) | set(EXPECTED_CURVE_LAYERS)
    known_aliases = set(LAYER_NAME_ALIASES.keys())

    report_lines = []
    per_subject = {}
    all_raw_names = set()
    missing_summary = {name: [] for name in expected}
    empty_summary = {name: [] for name in expected}
    unexpected_summary = {}

    for xcf_f in get_xcf_files(xcf_path):
        subj_code = '.'.join(xcf_f.split('.')[:-1])
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Loading {xcf_f} ...", end=' ', flush=True)

        names, data = load_xcf(os.path.join(xcf_path, xcf_f))
        raw_layer_dict = {
            name.lower(): d_i for name, d_i in zip(names, data) if '.dcm' not in name
        }
        print(f"done ({time.time() - t0:.1f}s)", flush=True)
        raw_names = set(raw_layer_dict.keys())
        all_raw_names |= raw_names

        # What does normalization do to this file?
        normalized_dict = normalize_layer_dict(dict(raw_layer_dict))
        normalized_names = set(normalized_dict.keys())

        aliased_here = raw_names & known_aliases
        unexpected_here = normalized_names - expected
        missing_here = expected - normalized_names

        # Present expected layers painted with zero pixels silently become an
        # absent class at training/DSC time.
        empty_here = set()
        for name in expected & normalized_names:
            if layer_pixel_count(normalized_dict[name]) == 0:
                empty_here.add(name)

        for name in missing_here:
            missing_summary[name].append(subj_code)
        for name in empty_here:
            empty_summary[name].append(subj_code)
        for name in unexpected_here:
            unexpected_summary.setdefault(name, []).append(subj_code)

        line = f"{subj_code:<20} raw_layers={sorted(raw_names)}"
        if aliased_here:
            line += f"  [ALIAS FIXED: {sorted(aliased_here)} -> {[LAYER_NAME_ALIASES[a] for a in aliased_here]}]"
        if missing_here:
            line += f"  [MISSING after normalization: {sorted(missing_here)}]"
        if empty_here:
            line += f"  [EMPTY (present but 0 px): {sorted(empty_here)}]"
        if unexpected_here:
            line += f"  [UNEXPECTED (possible new typo): {sorted(unexpected_here)}]"
        print(line, flush=True)
        report_lines.append(line)

        per_subject[subj_code] = normalized_dict

    report_lines.append("")
    report_lines.append("── Summary ──────────────────────────────────────────")
    report_lines.append(f"All raw layer names ever seen: {sorted(all_raw_names)}")
    report_lines.append("")
    report_lines.append("Subjects missing each expected layer (after alias fix):")
    for name, subjects in missing_summary.items():
        status = "OK (present in all subjects)" if not subjects else f"MISSING in {len(subjects)}: {subjects}"
        report_lines.append(f"  - {name}: {status}")
    report_lines.append("")
    report_lines.append("Subjects where each expected layer is PRESENT but EMPTY (0 px):")
    for name, subjects in empty_summary.items():
        status = "OK (none empty)" if not subjects else f"EMPTY in {len(subjects)}: {subjects}"
        report_lines.append(f"  - {name}: {status}")
    report_lines.append("")
    report_lines.append(
        "Effectively ABSENT per layer (missing OR empty — these are the "
        "(subject, class) pairs that become empty-both at DSC time):"
    )
    for name in sorted(expected):
        absent = sorted(set(missing_summary[name]) | set(empty_summary[name]))
        status = "OK (present & painted in all subjects)" if not absent else f"{len(absent)}: {absent}"
        report_lines.append(f"  - {name}: {status}")
    report_lines.append("")
    if unexpected_summary:
        report_lines.append("Unexpected layer names NOT in expected list or known aliases")
        report_lines.append("(these are candidates for new typos to add to LAYER_NAME_ALIASES):")
        for name, subjects in unexpected_summary.items():
            report_lines.append(f"  - '{name}' found in {len(subjects)} subject(s): {subjects}")
    else:
        report_lines.append("No unexpected/unrecognized layer names found.")

    return report_lines, per_subject


def save_mask_panels(subj_code, layer_dict, out_dir):
    """
    Save one figure per subject with one panel per expected area layer,
    so an empty/black panel makes a missing or mislabeled layer visually
    obvious at a glance.
    """
    layers_to_show = EXPECTED_AREA_LAYERS + EXPECTED_CURVE_LAYERS
    n = len(layers_to_show)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, layer_name in zip(axes, layers_to_show):
        if layer_name in layer_dict:
            filled = binary_fill_holes(np.array(layer_dict[layer_name])[..., -1] > 0)
            pixel_count = int(filled.sum())
            ax.imshow(filled, cmap='gray')
            title = f"{layer_name}\n({pixel_count} px)"
            if pixel_count == 0:
                title += "\n** EMPTY MASK **"
                ax.set_title(title, color='red')
            else:
                ax.set_title(title)
        else:
            blank = np.zeros((10, 10))
            ax.imshow(blank, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"{layer_name}\n** LAYER NOT FOUND **", color='red')
        ax.axis('off')

    fig.suptitle(subj_code)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{subj_code}_masks.png"), bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    masks_dir = os.path.join(OUT_DIR, 'masks')

    print(f"Auditing XCF layers in: {XCF_PATH}", flush=True)
    report_lines, per_subject = audit_layers(XCF_PATH)

    report_path = os.path.join(OUT_DIR, 'layer_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\nFull layer report written to: {report_path}")

    print(f"\nSaving per-subject mask panels to: {masks_dir}")
    n_subjects = len(per_subject)
    for i, (subj_code, layer_dict) in enumerate(per_subject.items(), start=1):
        t0 = time.time()
        print(f"[{i}/{n_subjects}] {subj_code} ...", end=' ', flush=True)
        save_mask_panels(subj_code, layer_dict, masks_dir)
        print(f"saved ({time.time() - t0:.1f}s)", flush=True)

    print(f"\nDone. Check {report_path} for any 'MISSING' or 'UNEXPECTED' lines,")
    print(f"and scan {masks_dir}/*.png for panels marked 'EMPTY MASK' or 'LAYER NOT FOUND'.")


if __name__ == '__main__':
    main()
