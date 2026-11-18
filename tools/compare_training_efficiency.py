"""Apply VertiKV's final-200 train CE score to the completed YOCO cohorts."""
import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.057
WINDOW = 200
COHORTS = {
    ('gRet', '160M'): 'ab_baseline_20261104',
    ('gRet', '400M'): 'gret_400m_3way_20261108T194918Z',
    ('SWA', '160M'): 'swa_160m_ab_20261107',
    ('SWA', '400M'): 'swa_400m_ab_20261111T054430Z',
}


def efficiency(loss, baseline):
    return (loss / baseline) ** (-1 / ALPHA)


def read_rows(path):
    with path.open(newline='') as source:
        return list(csv.DictReader(source))


def summarize(family, size, arm, points, expected_count=WINDOW):
    if len(points) != expected_count or len({tokens for tokens, _, _ in points}) != expected_count:
        raise ValueError(f'{family} {size} {arm}: expected {expected_count} distinct token counts')
    if any(not math.isfinite(value) or value <= 0 for _, loss, base in points for value in (loss, base)):
        raise ValueError(f'{family} {size} {arm}: expected positive finite train losses')
    tokens, losses, baselines = zip(*points)
    scores = [efficiency(loss, base) for loss, base in zip(losses, baselines)]
    return {
        'family': family, 'size': size, 'arm': arm,
        'points': len(scores), 'first_tokens': min(tokens), 'last_tokens': max(tokens),
        'mean_train_ce': statistics.mean(losses), 'mean_baseline_ce': statistics.mean(baselines),
        'mean_efficiency': statistics.mean(scores), 'median_efficiency': statistics.median(scores),
    }


def yoco_summaries():
    for (family, size), folder in COHORTS.items():
        traces = defaultdict(dict)
        for row in read_rows(ROOT / 'evaluation' / folder / 'training_curves.csv'):
            arm, step, tokens = row['arm'], int(row['step']), int(row['tokens'])
            if tokens in traces[arm] or tokens != step * 262144:
                raise ValueError(f'{folder}: invalid or repeated token count {tokens} for {arm}')
            traces[arm][tokens] = float(row['loss'])
        expected = set(range(262144, 40000 * 262144 + 1, 262144))
        if set(traces) != {'baseline', 'A', 'B'} or any(set(trace) != expected for trace in traces.values()):
            raise ValueError(f'{folder}: expected three complete 40000-update trajectories')
        window = sorted(expected)[-WINDOW:]
        for arm in ('baseline', 'A', 'B'):
            points = [(tokens, traces[arm][tokens], traces['baseline'][tokens]) for tokens in window]
            yield summarize(family, size, arm, points)


def vertikv_summaries(directory):
    published = {(row['size'], row['variant']): row
                 for row in read_rows(directory / 'final200_efficiency_summary.csv')}
    groups = defaultdict(list)
    for row in read_rows(directory / 'final200_efficiency_points.csv'):
        groups[row['size'], row['variant']].append(
            (int(float(row['total_tokens'])), float(row['train_ce']), float(row['aligned_baseline_ce']))
        )
    expected = {(f'{size}M', arm) for size in (150, 300, 500, 700)
                for arm in ('baseline', 'gate_a', 'gate_b', 'gate_c', 'gate_d', 'gate_e', 'gate_f')}
    if groups.keys() != expected or published.keys() != expected:
        raise ValueError('Expected all 28 published VertiKV size/variant pairs')
    for (size, arm), points in groups.items():
        reference = published[size, arm]
        row = summarize('VertiKV', size, arm, points, int(reference['points']))
        if not math.isclose(row['mean_efficiency'], float(reference['mean_efficiency_score']), rel_tol=1e-12):
            raise ValueError(f'VertiKV {size} {arm} disagrees with the archived summary')
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vertikv', type=Path, required=True, help='Path to the cloned VertiKV repository')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.vertikv / 'experiments/01_lm_quality_scaling/final200/data'
    rows = [*yoco_summaries(), *vertikv_summaries(source)]
    with args.output.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
