from .version import version as __version__

from .comparisontools import (count_matching_events, compute_agreement_score, count_match_spikes,
        make_possible_match, make_best_match, make_hungarian_match, 
        do_score_labels, do_confusion_matrix, compare_spike_trains)
from .symmetricsortingcomparison import compare_two_sorters, SymmetricSortingComparison
from .groundtruthcomparison import compare_sorter_to_ground_truth, GroundTruthComparison
from .multisortingcomparison import compare_multiple_sorters, MultiSortingComparison

from .groundtruthstudy import GroundTruthStudy
