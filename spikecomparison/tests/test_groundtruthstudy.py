import os
import shutil
import time
import pickle

import pytest

import spikeextractors as se

import spikesorters as ss
from spikecomparison.groundtruthstudy import GroundTruthStudy

study_folder = 'test_groundtruthstudy/'


def setup_module():
    if os.path.exists(study_folder):
        shutil.rmtree(study_folder)
    _setup_comparison_study()
    _run_study_sorters()


def _setup_comparison_study():
    rec0, gt_sorting0 = se.example_datasets.toy_example(num_channels=4, duration=30, seed=0)
    rec1, gt_sorting1 = se.example_datasets.toy_example(num_channels=32, duration=30, seed=0)

    gt_dict = {
        'toy_tetrode': (rec0, gt_sorting0),
        'toy_probe32': (rec1, gt_sorting1),
    }

    study = GroundTruthStudy.create(study_folder, gt_dict)


def _run_study_sorters():
    study = GroundTruthStudy(study_folder)
    sorter_list = ['tridesclous', 'herdingspikes']
    print(f"#################################"
          f"INSTALLED SORTERS"
          f"#################################\n{ss.installed_sorters()}")
    study.run_sorters(sorter_list)


def test_extract_sortings():
    study = GroundTruthStudy(study_folder)
    print(study)

    for rec_name in study.rec_names:
        gt_sorting = study.get_ground_truth(rec_name)
        # ~ print(rec_name, gt_sorting)

    for rec_name in study.rec_names:
        snr = study.get_units_snr(rec_name=rec_name)
        # print(snr)

    study.copy_sortings()
    study.run_comparisons(exhaustive_gt=True)

    run_times = study.aggregate_run_times()
    perf = study.aggregate_performance_by_units()
    count_units = study.aggregate_count_units()
    dataframes = study.aggregate_dataframes()

    shutil.rmtree(study_folder)


if __name__ == '__main__':
    #~ setup_module()
    test_extract_sortings()
