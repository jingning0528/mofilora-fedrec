#!/usr/bin/env python3
"""Copy results into a dataset-oriented layout, preserving originals and hashes."""
import argparse
import csv
import hashlib
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Paper sources confirmed by the user; Industrial-less remains an archive.
DATASETS = ('ML1M', 'Instrument', 'Industrial', 'Software')
AB_METHODS = {
    'analyze_1000__FedNCF-LoRA': 'LoRA',
    'analyze_1000__FedNCF-FixedB': 'FixedB',
    'analyze_1000__MofiLoRA': 'MoFiLoRA',
    'instrument_AB__LoRA': 'LoRA',
    'instrument_AB__LoRA-FixedB': 'FixedB',
    'instrument_AB__MoFiLoRA': 'MoFiLoRA',
}
TABLE_NAMES = {
    '': 'method_comparison',
    '_mean_std': 'method_comparison_mean_std',
    '_vs_LoRA': 'relative_to_lora',
    '_All_Result': 'all_method_results',
}


def destination(source):
    """Return destination, descriptive title, and evidence/status."""
    parts = source.parts
    if parts[0] == 'results':
        return Path(*parts[1:]), source.stem.replace('_', ' '), 'Paper export relocated from repository root; contents preserved'
    if parts[0] == 'log' and len(parts) == 4 and parts[1] == 'multi_seed' and parts[2] in DATASETS and source.suffix == '.txt':
        dataset = parts[2]
        return Path(dataset) / 'multi_seed/logs' / source.name, f'{dataset}: final multi-seed run {source.stem}', 'Final paper result source: multi_seed (user confirmed)'
    if parts[0] == 'csv' and len(parts) == 2:
        if source.name == 'Paper_Table.csv':
            return Path('paper/tables/paper_table.csv'), 'Cross-dataset paper table', 'Existing paper table; verify against final multi_seed logs before publication'
        for dataset in DATASETS:
            if source.stem.startswith(dataset):
                suffix = source.stem[len(dataset):]
                if suffix in TABLE_NAMES:
                    title = TABLE_NAMES[suffix]
                    return Path(dataset) / 'comparison/tables' / f'{dataset}_{title}.csv', f'{dataset}: {title.replace("_", " ")}', 'Existing summary table; verify against final multi_seed logs before publication'
                return Path('archive') / source, f'{dataset}: historical table {source.stem}', 'Historical/alternate version; not selected for paper'
    if parts[0] in ('csv', 'figures') and len(parts) > 2:
        kind = 'tables' if parts[0] == 'csv' else 'figures'
        if parts[1] in ('ML1M_AB', 'instrument_AB'):
            dataset = 'ML1M' if parts[1] == 'ML1M_AB' else 'Instrument'
            # Older short-run experiments stay separate from the 1000-round paper analysis.
            if parts[2].startswith('analyze__'):
                return Path('archive') / source, f'{dataset}: historical A/B analysis', 'Historical run'
            if parts[2] in AB_METHODS:
                method = AB_METHODS[parts[2]]
                return Path(dataset) / 'AB' / method / kind / Path(*parts[3:]), f'{dataset}: {method} A/B {source.stem.replace("_", " ")}', 'Dataset and method identified by existing analysis folder'
            return Path(dataset) / 'AB/comparison' / kind / Path(*parts[3:] if parts[2] == 'comparison' else parts[2:]), f'{dataset}: A/B comparison {source.stem.replace("_", " ")}', 'Existing comparison artifact; original contents retained'
        if parts[1] == 'Efficiency':
            return Path('paper/efficiency/figures') / Path(*parts[2:]), f'Efficiency: {source.stem.replace("_", " ")}', 'Existing efficiency figures; run provenance unconfirmed'
    if parts[0] == 'log' and len(parts) > 2 and parts[1] == 'analyze_1000':
        text = (ROOT / source).read_text(errors='replace')[:20000]
        if re.search(r'"dataset_id"\s*:\s*"ML1M_t5"', text):
            method = {'FedNCF-LoRA': 'LoRA', 'FedNCF-FixedB': 'FixedB', 'MofiLoRA': 'MoFiLoRA'}.get(source.stem)
            if method:
                return Path('ML1M/AB') / method / 'logs' / source.name, f'ML1M: {method} A/B training log', 'Dataset verified from logged parameters'
            experiment = 'full_model'
            return Path('ML1M') / experiment / 'logs' / source.name, f'ML1M: {source.stem} training log', 'Dataset verified from logged parameters'
    if len(parts) == 1 and source.suffix in ('.pdf', '.png'):
        return Path('paper/efficiency/figures/root_exports') / source, source.stem.replace('_', ' '), 'Root-level paper figure; retained separately from other versions'
    return Path('archive') / source, source.stem.replace('_', ' '), 'Historical or provenance unconfirmed; original hierarchy retained'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Copy files and write the index; otherwise preview counts')
    args = parser.parse_args()
    sources = []
    for folder in ('csv', 'figures', 'log'):
        sources.extend(p.relative_to(ROOT) for p in (ROOT / folder).rglob('*')
                       if p.is_file() and not p.name.startswith('.') and p.suffix != '.py')
    sources.extend(p.relative_to(ROOT) for pattern in ('efficiency_comparison.*', 'model_size.*')
                   for p in ROOT.glob(pattern) if p.suffix in ('.pdf', '.png'))
    old_index = ROOT / 'results/index.csv'
    previous = {}
    if old_index.exists():
        with old_index.open() as f:
            previous = {r['source']: r for r in csv.DictReader(f)}
    exports = ROOT / 'results/paper/efficiency/figures/root_exports'
    if exports.exists():
        sources.extend(p.relative_to(ROOT) for p in exports.iterdir() if p.suffix in ('.pdf', '.png'))
    rows = []
    targets = set()
    for source in sorted(sources):
        dest, title, status = destination(source)
        if dest in targets:
            raise ValueError(f'Duplicate destination: {dest}')
        targets.add(dest)
        original = ROOT / source
        target = ROOT / 'results' / dest
        sha = digest(original)
        rows.append({'title': title, 'source': str(source), 'organized_path': str(Path('results') / dest),
                     'status': status, 'bytes': original.stat().st_size, 'sha256': sha})
        if args.apply:
            if target.exists() and digest(target) != sha:
                raise ValueError(f'Refusing to overwrite changed result: {target}')
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
            if digest(target) != sha:
                raise ValueError(f'Copy verification failed: {target}')
    if args.apply:
        # Relocate only verified organized copies; original files remain untouched.
        for row in rows:
            old = previous.get(row['source'])
            if old and old['organized_path'] != row['organized_path']:
                old_path = ROOT / old['organized_path']
                if old_path.is_file() and digest(old_path) == row['sha256']:
                    old_path.unlink()
        # Remove only redundant organized archive copies promoted to final results.
        # Original files under log/ are never removed.
        for row in rows:
            if '/multi_seed/logs/' in row['organized_path']:
                old = ROOT / 'results/archive' / row['source']
                if old.is_file() and digest(old) == row['sha256']:
                    old.unlink()
        runs = []
        for row in rows:
            if '/multi_seed/logs/' not in row['organized_path']:
                continue
            text = (ROOT / row['source']).read_text(errors='replace')
            def param(name):
                match = re.search(r'"' + name + r'"\s*:\s*"([^"\n]*)"', text)
                return match.group(1) if match else ''
            turns = re.findall(r'Train Turn\s+(\d+)', text)
            runs.append({'dataset': Path(row['organized_path']).parts[1],
                         'method': param('model'), 'seed': param('seed'),
                         'configured_rounds': param('train_turn'),
                         'last_logged_round': max(map(int, turns)) if turns else '',
                         'has_final_test': 'Final Test' in text,
                         'log': row['organized_path']})
        if runs:
            with (ROOT / 'results/paper/multi_seed_runs.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(runs[0]))
                writer.writeheader()
                writer.writerows(runs)
        with (ROOT / 'results/index.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    archive = sum('/archive/' in row['organized_path'] for row in rows)
    print(f'{len(rows)} files: {len(rows) - archive} organized paper candidates; {archive} archived files.')
    print('Originals preserved. SHA-256 verified.' if args.apply else 'Preview only; use --apply to copy.')


if __name__ == '__main__':
    main()
