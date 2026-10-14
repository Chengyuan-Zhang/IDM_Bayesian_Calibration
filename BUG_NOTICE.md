# Confirmed highD preprocessing bug

**Date: 14 October 2026. Status: code corrected; published-result impact assessment pending.**

We confirmed an indexing error in the highD preprocessing distributed with this repository. It affects the vehicle-class labels in the historical pair caches and the interpretation of the configured car/truck groups. We also confirmed an error in the lane-change exclusion rule. We apologize for these errors.

We thank **Wuping Xin** for discovering and reporting the vehicle-class indexing error. His detailed report and concrete example helped us verify and correct the issue.

## What went wrong

At commit [`7520b15`](https://github.com/Chengyuan-Zhang/IDM_Bayesian_Calibration/commit/7520b15a8b163f52532cd3e5c7f7d4db05fc43f0), `read_track_meta_csv` returned the `class` column with its default zero-based row index. `load_CF_pairs` queried that column using the one-based vehicle ID. Vehicle `k` consequently received vehicle `k+1`'s class, while its trajectory and vehicle length still belonged to vehicle `k`.

In recording 25, follower 211 is a 4.14 m **Car**, but its cached class is **Truck**. Vehicle 212 is the 6.27 m Truck whose class was incorrectly assigned to follower 211.

The lane-change reader similarly returned row indices rather than vehicle IDs. It selected only the group with exactly one lane change, rather than all vehicles with at least one lane change. The intended exclusion of lane-changing vehicles was therefore not applied correctly.

## Verified scope

The audit examined all 60 historical `MinLength50_freq_5.pkl` pair-cache files. Their Git blob hashes matched commit `7520b15`. Re-extraction with that commit's code from the original highD CSVs reproduced all 251 cached pairs exactly, including every stored numerical field. The nonempty caches contain 212 pairs from recording 25, 38 from recording 26, and one from recording 36.

All 251 class queries used the shifted index. **84 followers have an incorrect cached class; 167 followers happen to share the next vehicle's class.**

| Cached class | Actual Car | Actual Truck | Total |
| --- | ---: | ---: | ---: |
| Car | 157 | 60 | 217 |
| Truck | 24 | 10 | 34 |
| Total | 181 | 70 | 251 |

The original configuration's ten nominal cars comprise **nine Cars and one Truck**. Its ten nominal trucks comprise **seven Cars and three Trucks**. The single-vehicle truck notebooks selected pair 3, which corresponds to the Car with ID 211 in recording 25. These are confirmed errors in the public implementation's vehicle grouping.

Eight cached followers have `numLaneChanges > 0` in their original metadata. This count concerns their full recorded trajectories; it does not establish that a lane change occurred inside every extracted following segment.

Changing only the class lookup left all non-class fields of the 251 extracted pairs unchanged in the audit. Correcting the lane-change rule can change pair membership and list order, so it requires separate preprocessing caches.

## What the fix changes

- Vehicle classes are indexed and queried by the explicit `id` column. Leader trajectories are also looked up by vehicle ID.
- Lane-change exclusions use actual vehicle IDs and include `numLaneChanges > 0`.
- Corrected pair caches use the suffix `_metadata_v2.pkl` and a versioned payload. Historical pair caches are preserved and are never automatically reused by the corrected loader.
- The original set of 20 configured vehicles is preserved, regrouped into **16 Cars and four Trucks**. Configuration keys are now `(recording_id, follower_id)` tuples, independent of extraction/list order. No replacement vehicles are introduced.
- Notebook and simulation cache filenames also use `_metadata_v2.pkl`, preventing reuse or overwrite of historical fitted traces and simulation caches. Simulation indexing uses the actual car-group size instead of assuming ten cars.
- The raw-track reader no longer writes or deletes files in the raw dataset directory.

## Scope of scientific conclusions

This repository provides code for *Bayesian Calibration of the Intelligent Driver Model* and *Calibrating Car-following Models via Bayesian Dynamic Regression*. The public implementation's car/truck labels and comparisons require reassessment. **This patch does not provide recalibrated models, corrected performance metrics, or a completed assessment of either paper's published conclusions.** Historical notebook outputs and figures have not been recomputed and remain historical results.

This audit does not determine whether the preprocessing used for the separate 2026 paper *When context is not enough* shares this error. That question requires its actual preprocessing version and input-data provenance. No claim of either impact or absence of impact on that paper is made here.

## Using the corrected code

1. Keep historical caches and model outputs for reproducibility. To reproduce the historical public implementation, use commit `7520b15` and interpret its groups with the corrected identities above.
2. For corrected preprocessing, provide your licensed highD files under `data/highD/`. New pair caches are generated separately. The corrected caches are not distributed by this patch.
3. Run preprocessing from the repository root:

   ```python
   from data.data_loader import read_training_data
   from config import Config

   pairs = read_training_data(base_path="data/highD/", dataset_list=[25, 26, 36])
   car = pairs[Config.car_interactive_pair_list[0]]
   truck = pairs[Config.truck_interactive_pair_list[0]]
   assert car["class_foll"] == "Car"
   assert truck["class_foll"] == "Truck"
   ```

4. Refit models on the corrected inputs before using the new trace filenames for analysis or simulation. The historical 10/10 group sizes must not be interpreted as the corrected cohort. Changing the membership or selecting a new balanced cohort requires a separately documented study design.

## Reproducing the checks

The regression suite uses the standard library test runner and the repository's existing NumPy/pandas dependencies:

```sh
python -m unittest discover -s tests -v
```

With licensed raw highD metadata and the historical public caches, reproduce the class and cohort census without changing either input:

```sh
python scripts/audit_legacy_highd.py --data-dir data/highD --cache-dir data/cache
```

The disclosure's counts refer to the historical 50-second, 5 Hz caches, not to a new fitted model or the size of the corrected extraction.
