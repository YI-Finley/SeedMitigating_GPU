#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_BASE = Path('/root/code/SeedMitigating/output/eval_flashrag500_7models')
DEFAULT_ROOT = DEFAULT_BASE / 'response_level'
DEFAULT_OUT_DIR = DEFAULT_BASE / 'report'
DEFAULT_OUT_XLSX = DEFAULT_OUT_DIR / 'eval_flashrag500_7models_metrics.xlsx'

METRICS = [
    ('SNR Gain', 'snr_gain', 'up'),
    ('Conf AUC', 'confidence_auc', 'up'),
    ('Abs Acc', 'abstention_accuracy', 'up'),
    ('smECE', 'smoothed_ece', 'down'),
    ('Brier', 'brier_score', 'down'),
    ('NLL', 'nll', 'down'),
    ('Pred Acc', 'predictive_accuracy', 'up'),
]

TEST_DATASETS = [
    'flashrag_nq',
    'flashrag_triviaqa',
    'flashrag_popqa',
    'flashrag_hotpotqa',
    'flashrag_2wikimultihopqa',
    'flashrag_musique',
    'flashrag_bamboogle',
]

TEST_DISPLAY = {
    'flashrag_nq': 'NQ',
    'flashrag_triviaqa': 'TriviaQA',
    'flashrag_popqa': 'PopQA',
    'flashrag_hotpotqa': 'HotpotQA',
    'flashrag_2wikimultihopqa': '2wiki',
    'flashrag_musique': 'Musique',
    'flashrag_bamboogle': 'Bamboogle',
}

TEST_GROUPS = {
    'General QA': ['flashrag_nq', 'flashrag_triviaqa', 'flashrag_popqa'],
    'Multi-Hop QA': ['flashrag_hotpotqa', 'flashrag_2wikimultihopqa', 'flashrag_musique', 'flashrag_bamboogle'],
}


def simplify_method_label(method: str) -> str:
    if method.startswith('Binary Reward'):
        return 'Binary Reward'
    if method.startswith('Verbalized Brier'):
        return 'Verbalized Brier'
    if method.startswith('Verbalized CE'):
        return 'Verbalized CE'
    return method


def extract_train_group(method: str) -> str:
    if method == 'Qwen3-4B-Instruct-2507 (Base)':
        return 'Base'
    if '(' in method and method.endswith(')'):
        return method.rsplit('(', 1)[-1][:-1].strip()
    return 'Other'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export comparison tables from a response_level results directory.'
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=DEFAULT_ROOT,
        help='Input response_level directory. Default: %(default)s',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=DEFAULT_OUT_XLSX,
        help='Output xlsx path. Default: %(default)s',
    )
    return parser.parse_args()


def collect_df(root: Path) -> pd.DataFrame:
    rows = []
    for mdir in sorted([p for p in root.iterdir() if p.is_dir()]):
        method = mdir.name
        for ddir in sorted([p for p in mdir.iterdir() if p.is_dir()]):
            ds = ddir.name
            f = ddir / 'results.json'
            if not f.exists():
                continue
            data = json.loads(f.read_text(encoding='utf-8'))
            m = data.get('metrics') or data.get('metrics_confidence') or {}
            row = {'Method': method, 'Dataset': ds}
            for show, key, _ in METRICS:
                row[show] = float(m.get(key, 0.0) or 0.0)
            rows.append(row)
    if not rows:
        raise RuntimeError(f'No results found under {root}')
    return pd.DataFrame(rows)


def metric_label(metric_name: str, direction: str) -> str:
    return f'{metric_name} (↑)' if direction == 'up' else f'{metric_name} (↓)'


def get_method_order(df: pd.DataFrame):
    preferred_prefixes = [
        'Qwen3-4B-Instruct-2507 (Base)',
        'Binary Reward',
        'Verbalized Brier',
        'Verbalized CE',
    ]
    existing = df['Method'].unique().tolist()
    ordered = []
    for prefix in preferred_prefixes:
        ordered.extend([m for m in existing if m == prefix or m.startswith(prefix)])
    ordered += [m for m in existing if m not in ordered]
    return ordered


def get_train_groups(df: pd.DataFrame):
    groups = {}
    methods = get_method_order(df)
    train_labels = []
    for method in methods:
        label = extract_train_group(method)
        if label not in train_labels and label not in ('Base', 'Other'):
            train_labels.append(label)
    for label in train_labels:
        grouped = ['Qwen3-4B-Instruct-2507 (Base)']
        grouped.extend([m for m in methods if extract_train_group(m) == label])
        groups[label] = grouped
    return groups


def get_dataset_order(df: pd.DataFrame):
    existing = df['Dataset'].unique().tolist()
    ordered = [d for d in TEST_DATASETS if d in existing]
    ordered += [d for d in existing if d not in ordered]
    return ordered


def get_test_group_spans(dataset_order):
    spans = []
    start = 1
    for label, members in TEST_GROUPS.items():
        present = [d for d in dataset_order if d in members]
        if not present:
            continue
        end = start + len(present) - 1
        spans.append((label, start, end, present))
        start = end + 1
    return spans


def write_all_in_one_sheet(writer, df: pd.DataFrame):
    wb = writer.book
    ws = wb.add_worksheet('all_in_one')
    writer.sheets['all_in_one'] = ws

    # formats
    title_fmt = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'align': 'left'})
    section_fmt = wb.add_format({'bold': True, 'bg_color': '#E2EFDA', 'align': 'left'})
    metric_fmt = wb.add_format({'bold': True, 'bg_color': '#FFF2CC', 'align': 'left'})
    header_fmt = wb.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1, 'align': 'center'})
    text_fmt = wb.add_format({'align': 'left'})
    num_fmt = wb.add_format({'num_format': '0.0000'})
    best_fmt = wb.add_format({'bold': True, 'num_format': '0.0000'})

    # widths
    ws.set_column(0, 0, 38)  # method
    for c in range(1, 1 + len(TEST_DATASETS)):
        ws.set_column(c, c, 13)

    method_order = get_method_order(df)
    dataset_order = get_dataset_order(df)
    train_groups = get_train_groups(df)

    row = 0
    total_cols = 1 + len(dataset_order)
    ws.merge_range(row, 0, row, total_cols - 1,
                   'Compare methods within the same train dataset (best in bold)', title_fmt)
    row += 2

    for train_label, grouped_methods in train_groups.items():
        ws.merge_range(row, 0, row, total_cols - 1,
                       f'Train: {train_label}', section_fmt)
        row += 1

        for metric_name, _metric_key, direction in METRICS:
            ws.merge_range(row, 0, row, total_cols - 1,
                           f'Metric: {metric_name}', metric_fmt)
            row += 1

            spans = get_test_group_spans(dataset_order)
            ws.write(row, 0, '', header_fmt)
            for label, start_col, end_col, _present in spans:
                ws.merge_range(row, start_col, row, end_col, label, header_fmt)
            row += 1

            ws.write(row, 0, 'Method', header_fmt)
            for j, ds in enumerate(dataset_order, start=1):
                ws.write(row, j, TEST_DISPLAY.get(ds, ds), header_fmt)
            row += 1

            values = []
            for method in grouped_methods:
                row_vals = []
                for ds in dataset_order:
                    s = df[(df['Method'] == method) & (df['Dataset'] == ds)][metric_name]
                    row_vals.append(float(s.iloc[0]) if len(s) > 0 else None)
                values.append(row_vals)

            best_per_col = []
            for col_idx in range(len(dataset_order)):
                col = [v[col_idx] for v in values if v[col_idx] is not None]
                if not col:
                    best_per_col.append(None)
                else:
                    best_per_col.append(max(col) if direction == 'up' else min(col))

            for i, method in enumerate(grouped_methods):
                ws.write(row + i, 0, simplify_method_label(method), text_fmt)
                for j, val in enumerate(values[i], start=1):
                    if val is None:
                        continue
                    best_val = best_per_col[j - 1]
                    fmt = best_fmt if (best_val is not None and abs(val - best_val) < 1e-12) else num_fmt
                    ws.write_number(row + i, j, val, fmt)

            row += len(grouped_methods) + 2

    ws.freeze_panes(4, 1)


def write_right_compact_sheet(writer, df: pd.DataFrame):
    wb = writer.book
    ws = wb.add_worksheet('right_compact')
    writer.sheets['right_compact'] = ws

    # formats
    title_fmt = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'align': 'left'})
    section_fmt = wb.add_format({'bold': True, 'bg_color': '#E2EFDA', 'align': 'left'})
    test_fmt = wb.add_format({'bold': True, 'bg_color': '#FFF2CC', 'align': 'left'})
    header_fmt = wb.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1, 'align': 'center'})
    text_fmt = wb.add_format({'align': 'left'})
    num_fmt = wb.add_format({'num_format': '0.0000'})
    best_fmt = wb.add_format({'bold': True, 'num_format': '0.0000'})

    # widths
    ws.set_column(0, 0, 36)  # method
    for c in range(1, 1 + len(METRICS)):
        ws.set_column(c, c, 14)

    method_order = get_method_order(df)
    dataset_order = get_dataset_order(df)
    train_groups = get_train_groups(df)

    row = 0
    total_cols = 1 + len(METRICS)
    ws.merge_range(row, 0, row, total_cols - 1,
                   'Right layout: Train block -> Test block -> Method x Metrics (best in bold)',
                   title_fmt)
    row += 2

    for train_label, grouped_methods in train_groups.items():
        ws.merge_range(row, 0, row, total_cols - 1, f'Train: {train_label}', section_fmt)
        row += 1

        for group_label, _start_col, _end_col, present in get_test_group_spans(dataset_order):
            ws.merge_range(row, 0, row, total_cols - 1, f'Test Group: {group_label}', test_fmt)
            row += 1

            for ds in present:
                ws.merge_range(row, 0, row, total_cols - 1, f'Test: {TEST_DISPLAY.get(ds, ds)}', test_fmt)
                row += 1

                ws.write(row, 0, 'Method', header_fmt)
                for j, (metric_name, _metric_key, direction) in enumerate(METRICS, start=1):
                    ws.write(row, j, metric_label(metric_name, direction), header_fmt)
                row += 1

                values = []
                for method in grouped_methods:
                    row_vals = []
                    for metric_name, _metric_key, _direction in METRICS:
                        s = df[(df['Method'] == method) & (df['Dataset'] == ds)][metric_name]
                        row_vals.append(float(s.iloc[0]) if len(s) > 0 else None)
                    values.append(row_vals)

                best_per_metric = []
                for metric_idx, (_metric_name, _metric_key, direction) in enumerate(METRICS):
                    col = [v[metric_idx] for v in values if v[metric_idx] is not None]
                    if not col:
                        best_per_metric.append(None)
                    else:
                        best_per_metric.append(max(col) if direction == 'up' else min(col))

                for i, method in enumerate(grouped_methods):
                    ws.write(row + i, 0, simplify_method_label(method), text_fmt)
                    for j, val in enumerate(values[i], start=1):
                        if val is None:
                            continue
                        best_val = best_per_metric[j - 1]
                        fmt = best_fmt if (best_val is not None and abs(val - best_val) < 1e-12) else num_fmt
                        ws.write_number(row + i, j, val, fmt)

                row += len(grouped_methods) + 2

    ws.freeze_panes(3, 1)


def main():
    args = parse_args()
    out_xlsx = args.out.resolve()
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df = collect_df(args.root.resolve())

    with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as writer:
        write_all_in_one_sheet(writer, df)
        write_right_compact_sheet(writer, df)

    print(f'[OK] rewritten xlsx: {out_xlsx}')


if __name__ == '__main__':
    main()
