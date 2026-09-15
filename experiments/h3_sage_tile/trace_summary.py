"""Summarize an explicit, completed PyTorch trace without reading request prompts."""
import argparse
import collections
import gzip
import json
from pathlib import Path


def kernel_kind(name):
    name = name.lower()
    if 'nccl' in name:
        return 'communication'
    if any(x in name for x in ('attn', 'attention', 'qk_int8')):
        return 'attention'
    if any(x in name for x in ('gemm', 'matmul', 'cutlass', 'wgmma')):
        return 'matrix'
    if any(x in name for x in ('rms', 'norm')):
        return 'normalization'
    if any(x in name for x in ('quant', 'cast')):
        return 'quantization_cast'
    return 'other'


def interval_union(intervals):
    end, total = float('-inf'), 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def aggregate(events, with_shape=False):
    rows = {}
    for e in events:
        shape = e.get('args', {}).get('Input Dims') if with_shape else None
        key = (e['name'], json.dumps(shape))
        row = rows.setdefault(key, dict(name=e['name'], shape=shape, count=0,
                                       total_us=0, max_us=0))
        row['count'] += 1
        row['total_us'] += e['dur']
        row['max_us'] = max(row['max_us'], e['dur'])
    return sorted(rows.values(), key=lambda row: -row['total_us'])


def summarize(events):
    kernels = [e for e in events if e.get('cat') == 'kernel' and 'dur' in e]
    cpu = [e for e in events if e.get('cat') == 'cpu_op' and 'dur' in e]
    groups = collections.defaultdict(float)
    devices = collections.defaultdict(list)
    for e in kernels:
        groups[kernel_kind(e['name'])] += e['dur']
        devices[str(e.get('args', {}).get('device', 'unknown'))].append(
            (e['ts'], e['ts'] + e['dur']))
    busy = {device: dict(busy_union_us=interval_union(intervals),
                        span_us=max(b for a, b in intervals)-min(a for a, b in intervals))
            for device, intervals in devices.items()}
    return dict(kernel_sum_us=sum(e['dur'] for e in kernels),
                groups_sum_us=dict(groups), devices=busy,
                kernels=aggregate(kernels), cpu_ops_inclusive=aggregate(cpu, True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == '.gz' else open
    with opener(args.trace, 'rt') as f:
        result = summarize(json.load(f)['traceEvents'])
    result['trace'] = str(args.trace)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('kernels', 'cpu_ops_inclusive')}, indent=2))


if __name__ == '__main__':
    main()
