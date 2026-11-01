import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuralbunch import scaled
from neuralbunch.fast_reference import load_fast_reference
from neuralbunch.reference_data import call_tile_field
from neuralbunch.reference_loader import flatten_tile

mod = load_fast_reference()
split = 'train'; seed = 20261101
descs = [d for d in scaled.descriptors_for_split(split)
         if d['kind'] == 'scenario' and d['direction'] in ('axis_x','random')
         and d['position_mode'] in ('near','far','subcell')]
configs = [
    ('o4_1e-7', {'order': 4, 'tol': 1e-7, 'max_level': 5, 'max_order': 16}),
    ('o6_1e-7', {'order': 6, 'tol': 1e-7, 'max_level': 5, 'max_order': 16}),
    ('o6_1e-6', {'order': 6, 'tol': 1e-6, 'max_level': 5, 'max_order': 16}),
    ('o6_1e-5', {'order': 6, 'tol': 1e-5, 'max_level': 5, 'max_order': 16}),
    ('o4_1e-5', {'order': 4, 'tol': 1e-5, 'max_level': 4, 'max_order': 16}),
    ('o6_1e-4', {'order': 6, 'tol': 1e-4, 'max_level': 4, 'max_order': 16}),
]
for idx in (108, 105, 94, 82):
    d = descs[idx]; s = scaled.make_sample(d, split, seed)
    print('CASE', idx, d['aspects'], d['position_mode'], 'g%.2f' % d['gamma'],
          'dmin %.3f' % scaled._min_distance_cells(s), flush=True)
    refs = {}
    for label, q in configs:
        t = time.time()
        try:
            flat = flatten_tile(call_tile_field(mod, s, q))
        except Exception as exc:
            print('  ', label, 'ERR', exc, flush=True); continue
        dt = time.time() - t
        refs[label] = flat
        print('  %-8s %.3fs' % (label, dt), flush=True)
    base = refs.get('o4_1e-7')
    if base is not None:
        for label, v in refs.items():
            rel = float(np.linalg.norm(v - base) / max(np.linalg.norm(base), 1e-300))
            print('    rel %-8s %.3e' % (label, rel), flush=True)
