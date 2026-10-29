"""
Register the skull dataset in a local EllipseNet checkout (run once):
    python patch_ellipsenet.py /path/to/ellipsenet
Copies coco_skull.py next to coco_fhd.py, adds 'coco_skull' to the dataset
factory and gives the debugger a class name for it. Safe to run twice.
"""
import os
import sys
import shutil

root = sys.argv[1]
lib = os.path.join(root, 'src', 'lib')
here = os.path.dirname(os.path.abspath(__file__))

shutil.copy(os.path.join(here, 'coco_skull.py'),
            os.path.join(lib, 'datasets', 'dataset', 'coco_skull.py'))


def patch(path, anchor, addition, marker):
    text = open(path).read()
    if marker in text:
        return
    if anchor not in text:
        raise RuntimeError('Anchor not found in {}: {!r}'.format(path, anchor))
    open(path, 'w').write(text.replace(anchor, anchor + addition, 1))


patch(os.path.join(lib, 'datasets', 'dataset_factory.py'),
      'from .dataset.coco_fhd import COCOFHD\n',
      'from .dataset.coco_skull import COCOSkull\n', 'COCOSkull')
patch(os.path.join(lib, 'datasets', 'dataset_factory.py'),
      "'coco_fhd': COCOFHD",
      ",\n  'coco_skull': COCOSkull", "'coco_skull'")
patch(os.path.join(lib, 'utils', 'debugger.py'),
      "      self.names = fhd_class_name\n",
      "    elif dataset == 'coco_skull':\n      self.names = ['skull']\n",
      "'coco_skull'")

# DCNv2: replace the compiled extension wrapper with a torchvision version
dcn = os.path.join(lib, 'models', 'networks', 'DCNv2', 'dcn_v2.py')
backup = os.path.join(lib, 'models', 'networks', 'DCNv2', 'dcn_v2_original.py')
if 'torchvision.ops' not in open(dcn).read():
    shutil.copy(dcn, backup)
    shutil.copy(os.path.join(here, 'dcn_v2.py'), dcn)

# Rotated IoU: use the compiled sort_vertices if it exists, else pure PyTorch
cuda_op = os.path.join(lib, 'models', 'Rotated_IoU', 'cuda_op')
shutil.copy(os.path.join(here, 'sort_vertices_torch.py'),
            os.path.join(cuda_op, 'sort_vertices_torch.py'))
ext = os.path.join(cuda_op, 'cuda_ext.py')
text = open(ext).read()
if 'sort_vertices_torch' not in text:
    text = text.replace(
        'import sort_vertices\n',
        'try:\n    import sort_vertices\nexcept ImportError:  # not compiled: pure-PyTorch fallback\n'
        '    from . import sort_vertices_torch as sort_vertices\n', 1)
    open(ext, 'w').write(text)

print('EllipseNet patched for coco_skull (torchvision DCN, rotated-IoU fallback)')

# NumPy >= 1.24 removed np.int / np.float / np.bool / np.object
import re
alias = re.compile(r'\bnp\.(int|float|bool|object)\b(?![_0-9])')
for dirpath, _, files in os.walk(lib):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(dirpath, f)
            text = open(path).read()
            new = alias.sub(lambda m: m.group(1), text)
            if new != text:
                open(path, 'w').write(new)
print('NumPy aliases fixed')


def replace(path, old, new, count_expected=None):
    text = open(path).read()
    if new in text:
        return
    n = text.count(old)
    if n == 0 or (count_expected and n != count_expected):
        raise RuntimeError('Expected text not found in {}: {!r}'.format(path, old[:60]))
    open(path, 'w').write(text.replace(old, new))


# NumPy 2: one-element arrays are no longer accepted as Python scalars
eldet = os.path.join(lib, 'datasets', 'sample', 'eldet.py')
replace(eldet, 'l[k] = 2 * math.sqrt(a[k] * a[k] + b[k] * b[k])',
        'l[k] = 2 * math.sqrt(float(a[k][0]) ** 2 + float(b[k][0]) ** 2)')
replace(eldet, 'sincos[k] = [np.sin(theta[k] * math.pi), np.cos(theta[k] * math.pi)]',
        'sincos[k] = [np.sin(theta[k][0] * math.pi), np.cos(theta[k][0] * math.pi)]')

# IoU loss: the decoders assumed exactly 2 objects per image (lung + heart)
losses = os.path.join(lib, 'models', 'losses.py')
for f in ['decode_output_4iou_sincos(output)', 'decode_output_4iou(output)',
          'decode_label_4iou(target)']:
    text = open(losses).read()
    if f in text:
        open(losses, 'w').write(text.replace(f, f[:-1] + ", K=target['ind'].shape[1])"))
replace(os.path.join(lib, 'models', 'decode.py'),
        '        temp_box.append(bboxes[batch_idx, clses[batch_idx, 0], :])\n'
        '        temp_box.append(bboxes[batch_idx, clses[batch_idx, 1], :])\n',
        '        for k in range(K):  # was hard-coded for 2 objects\n'
        '            temp_box.append(bboxes[batch_idx, clses[batch_idx, k], :])\n')
print('NumPy 2 and one-object fixes applied')
