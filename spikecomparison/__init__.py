from .version import version as __version__

from .comparisontools import (count_matching_events, compute_agreement_score, match_spikes,
        do_matching, do_score_labels, do_confusion_matrix, compare_spike_trains)
from .sortingcomparison import compare_two_sorters, SortingComparison
from .groundtruthcomparison import compare_sorter_to_ground_truth, GroundTruthComparison
from .multisortingcomparison import compare_multiple_sorters, MultiSortingComparison

from .groundtruthstudy import GroundTruthStudy
