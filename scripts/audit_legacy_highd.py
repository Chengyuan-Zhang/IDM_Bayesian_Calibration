"""Read-only census of historical 50-second / 5 Hz pair-cache metadata."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle

import pandas as pd

LEGACY_CAR_INDICES = [14, 35, 23, 25, 36, 38, 60, 90, 228, 232]
LEGACY_TRUCK_INDICES = [3, 4, 18, 52, 81, 144, 153, 162, 5, 241]


def audit(data_dir: Path, cache_dir: Path) -> dict:
    rows, fingerprints = [], []
    for recording in range(1, 61):
        cache = cache_dir / ('track_pair_list_temp%02d_MinLength50_freq_5.pkl' % recording)
        metadata_path = data_dir / ('%02d_tracksMeta.csv' % recording)
        content = cache.read_bytes()
        metadata = pd.read_csv(metadata_path)
        by_id = metadata.set_index('id', verify_integrity=True)
        fingerprints.append({'recording': recording, 'cache_sha256': hashlib.sha256(content).hexdigest(),
                             'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest()})
        for pair in pickle.loads(content):
            vehicle = int(pair['id_Foll'])
            rows.append({'recording': recording, 'vehicle': vehicle,
                         'cached_class': pair['class_foll'], 'true_class': by_id.loc[vehicle, 'class'],
                         'shifted_class': metadata['class'].loc[vehicle],
                         'lane_changes': int(by_id.loc[vehicle, 'numLaneChanges'])})
    return {
        'historical_cache_files': len(fingerprints), 'pair_count': len(rows),
        'correct_class_count': sum(r['cached_class'] == r['true_class'] for r in rows),
        'incorrect_class_count': sum(r['cached_class'] != r['true_class'] for r in rows),
        'next_vehicle_class_match_count': sum(r['cached_class'] == r['shifted_class'] for r in rows),
        'confusion_cached_to_true': {'%s -> %s' % key: count for key, count in
                                   Counter((r['cached_class'], r['true_class']) for r in rows).items()},
        'original_car_group_true_classes': dict(Counter(rows[i]['true_class'] for i in LEGACY_CAR_INDICES)),
        'original_truck_group_true_classes': dict(Counter(rows[i]['true_class'] for i in LEGACY_TRUCK_INDICES)),
        'cached_followers_with_lane_changes': sum(r['lane_changes'] > 0 for r in rows),
        'input_fingerprints': fingerprints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=Path('data/cache'))
    args = parser.parse_args()
    print(json.dumps(audit(args.data_dir, args.cache_dir), indent=2))


if __name__ == '__main__':
    main()
