"""Regression checks for vehicle IDs, corrected grouping, and cache isolation."""

import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import Config
from data import data_loader, read_csv


def make_tracks(follower_id: int = 211, leader_id: int = 212, frames: int = 16) -> list:
    def track(vehicle_id: int, x_offset: float, preceding: int, following: int) -> dict:
        return {
            'id': vehicle_id, 'frame': np.arange(frames),
            'bbox': np.column_stack((np.arange(frames) + x_offset, np.zeros(frames),
                                    np.full(frames, 4.14), np.full(frames, 1.82))),
            'xVelocity': np.full(frames, 5.0), 'xAcceleration': np.zeros(frames),
            'precedingId': np.full(frames, preceding), 'followingId': np.full(frames, following),
        }
    return [track(leader_id, 15.0, 0, follower_id), track(follower_id, 0.0, leader_id, 0)]


class MetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_metadata(self, ids: list, classes: list, changes: list, name: str = 'meta.csv') -> Path:
        file = self.root / name
        pd.DataFrame({'id': ids, 'class': classes, 'numLaneChanges': changes}).to_csv(file, index=False)
        return file

    def test_classes_use_vehicle_ids_with_reordered_rows(self) -> None:
        file = self.write_metadata([212, 211], ['Truck', 'Car'], [0, 0])
        lane_changes, classes = read_csv.read_track_meta_csv(str(file))
        self.assertEqual(classes.loc[211], 'Car')
        self.assertEqual(classes.loc[212], 'Truck')
        self.assertEqual(lane_changes.tolist(), [])

    def test_all_lane_changes_return_actual_vehicle_ids(self) -> None:
        file = self.write_metadata([40, 10, 900], ['Truck', 'Car', 'Car'], [2, 0, 1])
        ids, _ = read_csv.read_track_meta_csv(str(file))
        self.assertEqual(ids.tolist(), [40, 900])

    def test_duplicate_vehicle_ids_are_rejected(self) -> None:
        file = self.write_metadata([211, 211], ['Car', 'Truck'], [0, 0])
        with self.assertRaises(ValueError):
            read_csv.read_track_meta_csv(str(file))

    def test_extraction_uses_follower_and_leader_ids(self) -> None:
        pairs = data_loader.load_CF_pairs(make_tracks(), np.array([], dtype=int),
                                         pd.Series({211: 'Car', 212: 'Truck'}), min_traj_lenth=5)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['id_Foll'], 211)
        self.assertEqual(pairs[0]['class_foll'], 'Car')
        np.testing.assert_allclose(pairs[0]['sReal'], 15.0 - 4.14)
        np.testing.assert_array_equal(pairs[0]['precedingId_Foll'], 212)

    def test_last_noncontiguous_vehicle_id_is_supported(self) -> None:
        pairs = data_loader.load_CF_pairs(make_tracks(900, 40), np.array([], dtype=int),
                                         pd.Series({40: 'Truck', 900: 'Car'}), min_traj_lenth=5)
        self.assertEqual(pairs[0]['class_foll'], 'Car')

    def test_lane_changing_follower_is_excluded(self) -> None:
        pairs = data_loader.load_CF_pairs(make_tracks(), np.array([211]),
                                         pd.Series({211: 'Car', 212: 'Truck'}), min_traj_lenth=5)
        self.assertEqual(pairs, [])

    def test_stable_lookup_is_independent_of_list_order(self) -> None:
        first = {'recording_id': 25, 'id_Foll': 211}
        second = {'recording_id': 26, 'id_Foll': 211}
        pairs = data_loader.CFPairList([second, first])
        self.assertIs(pairs[(25, 211)], first)
        self.assertIs(pairs[0], second)
        with self.assertRaises(KeyError):
            pairs[(36, 211)]
        with self.assertRaises(KeyError):
            data_loader.CFPairList([first, first])[(25, 211)]

    def test_historical_cache_is_preserved_and_not_loaded(self) -> None:
        self.write_metadata([212, 211], ['Truck', 'Car'], [0, 0], '25_tracksMeta.csv')
        old = self.root / 'track_pair_list_temp25_MinLength0_freq_5.pkl'
        old.write_bytes(pickle.dumps([{'class_foll': 'wrong historical class'}]))
        before = old.read_bytes()
        with patch.object(data_loader, 'read_dataset', return_value=make_tracks()) as reader:
            first = data_loader.read_training_data(str(self.root), [25], 5, str(self.root))
            second = data_loader.read_training_data(str(self.root), [25], 5, str(self.root))
            reader.assert_called_once()
        self.assertEqual(first[(25, 211)]['class_foll'], 'Car')
        self.assertEqual(second[(25, 211)]['class_foll'], 'Car')
        self.assertEqual(old.read_bytes(), before)
        self.assertEqual(first[0]['pair_No'], second[0]['pair_No'])
        new = self.root / 'track_pair_list_temp25_MinLength0_freq_5_metadata_v2.pkl'
        with new.open('rb') as stream:
            self.assertEqual(pickle.load(stream)['preprocessing_version'], 2)

    def test_unversioned_payload_is_rejected_and_preserved(self) -> None:
        cache = self.root / 'track_pair_list_temp25_MinLength0_freq_5_metadata_v2.pkl'
        before = pickle.dumps([])
        cache.write_bytes(before)
        with self.assertRaises(ValueError):
            data_loader.read_training_data(str(self.root), [25], 5, str(self.root))
        self.assertEqual(cache.read_bytes(), before)

    def test_raw_reader_does_not_write_cache(self) -> None:
        with patch.object(read_csv, 'read_track_csv', return_value=make_tracks()) as reader:
            tracks = read_csv.read_dataset(str(self.root), 25)
        reader.assert_called_once_with(str(self.root / '25_tracks.csv'))
        self.assertEqual(tracks[1]['id'], 211)
        self.assertFalse((self.root / 'dataset_25.pkl').exists())

    def test_id_60_always_returns_one_pair(self) -> None:
        pairs = data_loader.CFPairList([{'id_Foll': i} for i in range(61)])
        with patch.object(data_loader, 'read_training_data', return_value=pairs):
            self.assertIs(data_loader.read_id_60('unused'), pairs[60])

    def test_config_preserves_original_twenty_vehicles(self) -> None:
        root = Path(__file__).resolve().parents[1]
        legacy = []
        for recording in range(1, 61):
            file = root / 'data/cache' / ('track_pair_list_temp%02d_MinLength50_freq_5.pkl' % recording)
            with file.open('rb') as stream:
                legacy.extend((recording, int(pair['id_Foll'])) for pair in pickle.load(stream))
        indices = [14, 35, 23, 25, 36, 38, 60, 90, 228, 232, 3, 4, 18, 52, 81, 144, 153, 162, 5, 241]
        expected = {legacy[index] for index in indices}
        corrected = Config.car_interactive_pair_list + Config.truck_interactive_pair_list
        self.assertEqual(set(corrected), expected)
        self.assertEqual(len(corrected), 20)
        self.assertEqual(len(Config.car_interactive_pair_list), 16)
        self.assertEqual(len(Config.truck_interactive_pair_list), 4)

    def test_notebook_trace_names_are_versioned(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for folder in ['PGM_highD', 'PGM_highD_joint']:
            for file in (root / folder).glob('*.ipynb'):
                relative = file.relative_to(root).as_posix()
                current = json.loads(file.read_text(encoding='utf-8'))
                for after in current['cells']:
                    for line in after.get('source', []):
                        if '.pkl' in line:
                            self.assertNotIn('.pkl', line.replace('_metadata_v2.pkl', ''), relative)


if __name__ == '__main__':
    unittest.main()
