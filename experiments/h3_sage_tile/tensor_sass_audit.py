"""Reject known invalid tensor calibration lowering; not a dynamic counter.

Also inspect the loop increment and unrolled full-matrix groups when changing
compiler/architecture. Absence of dummy work alone cannot prove loop trip count.
"""
import re


def audit_tensor_sass(text):
    rows = []
    for part in text.split('Function : ')[1:]:
        name = part.splitlines()[0]
        if 'tensor_probe_kernel' not in name:
            continue
        if re.search(r'HGMMA\.64x8x16\.F16\s+RZ,', part):
            raise ValueError('Invalid nominal work: dummy WGMMA groups in ' + name)
        rows.append({'function': name,
                     'int8_full_instructions': part.count('IGMMA.64x128x32.S8.S8'),
                     'fp8_full_instructions': part.count('QGMMA.64x128x32.F32.E4M3.E4M3')})
    if len(rows) != 6 or any(r['int8_full_instructions'] + r['fp8_full_instructions'] < 4 for r in rows):
        raise ValueError('Expected six diagnostic kernels with full matrix instructions')
    return {'known_dummy_lowering_absent': True, 'functions': rows,
            'scope': 'Static structural guard; loop increments/trip counts still require inspection'}
