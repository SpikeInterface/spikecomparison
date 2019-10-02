import numpy as np
import pandas as pd

from .basecomparison import BaseTwoSorterComparison
from .comparisontools import (compute_agreement_score, do_score_labels, make_possible_match,
                              make_best_match, make_hungarian_match, do_confusion_matrix, do_count_score,
                              compute_performence)


# Note for dev,  because of  BaseTwoSorterComparison internally:
#     sorting1 = gt_sorting
#     sorting2 = tested_sorting

class GroundTruthComparison(BaseTwoSorterComparison):
    """
    Class to compare a sorter to ground truth (GT)
    
    This class can:
      * compute a "macth between gt_sorting and tested_sorting
      * compte th score label (TP, FN, CL, FP) for each spike
      * count by unit of GT the total of each (TP, FN, CL, FP) into a Dataframe 
        GroundTruthComparison.count
      * compute the confusion matrix .get_confusion_matrix()
      * compute some performance metric with several strategy based on 
        the count score by unit
      * count how much well detected units with some threshold selection
      * count false positve detected units
      * count units detected twice (or more)
      * summary all this
    """

    def __init__(self, gt_sorting, tested_sorting, gt_name=None, tested_name=None,
                 delta_time=0.4, sampling_frequency=None, min_accuracy=0.5, exhaustive_gt=False,
                 bad_redundant_threshold=0.2,
                 n_jobs=-1, match_mode='hungarian', compute_labels=False, verbose=False):

        if gt_name is None:
            gt_name = 'ground truth'
        if tested_name is None:
            tested_name = 'tested'
        BaseTwoSorterComparison.__init__(self, gt_sorting, tested_sorting, sorting1_name=gt_name,
                                         sorting2_name=tested_name,
                                         delta_time=delta_time, sampling_frequency=sampling_frequency,
                                         min_accuracy=min_accuracy, n_jobs=n_jobs,
                                         verbose=verbose)
        self.exhaustive_gt = exhaustive_gt
        self._bad_redundant_threshold = bad_redundant_threshold

        assert match_mode in ['hungarian', 'best']
        self.match_mode = match_mode
        self._compute_labels = compute_labels

        self._do_count()

        self._labels_st1 = None
        self._labels_st2 = None
        if self._compute_labels:
            self._do_score_labels()

        # confusion matrix is compute on demand
        self._confusion_matrix = None

    def get_labels1(self, unit_id):
        if self._labels_st1 is None:
            self._do_score_labels()

        if unit_id in self.sorting1.get_unit_ids():
            return self._labels_st1[unit_id]
        else:
            raise Exception("Unit_id is not a valid unit")

    def get_labels2(self, unit_id):
        if self._labels_st1 is None:
            self._do_score_labels()

        if unit_id in self.sorting2.get_unit_ids():
            return self._labels_st2[unit_id]
        else:
            raise Exception("Unit_id is not a valid unit")

    def _do_matching(self):
        if self._verbose:
            print("Matching...")

        self.possible_match_12, self.possible_match_21 = make_possible_match(self.agreement_scores, self.min_accuracy)
        self.best_match_12, self.best_match_21 = make_best_match(self.agreement_scores, self.min_accuracy)
        self.hungarian_match_12, self.hungarian_match_21 = make_hungarian_match(self.agreement_scores,
                                                                                self.min_accuracy)

    def _do_count(self):
        """
        Do raw count into a dataframe.
        
        Internally use hugarian match or best match.
        
        """
        if self.match_mode == 'hungarian':
            match_12 = self.hungarian_match_12
        elif self.match_mode == 'best':
            match_12 = self.best_match_12
        self.count_score = do_count_score(self.event_counts1, self.event_counts2,
                                          match_12, self.match_event_count)

    def _do_confusion_matrix(self):
        if self._verbose:
            print("Computing confusion matrix...")

        if self.match_mode == 'hungarian':
            match_12 = self.hungarian_match_12
        elif self.match_mode == 'best':
            match_12 = self.best_match_12

        self._confusion_matrix = do_confusion_matrix(self.event_counts1, self.event_counts2, match_12,
                                                     self.match_event_count)

    def get_confusion_matrix(self):
        """
        Computes the confusion matrix.

        Returns
        ------
        confusion_matrix: np.array
            The confusion matrix
        st1_idxs: np.array
            Array with order of units1 in confusion matrix
        st2_idxs: np.array
            Array with order of units2 in confusion matrix
        """
        if self._confusion_matrix is None:
            self._do_confusion_matrix()
        return self._confusion_matrix

    def _do_score_labels(self):
        assert self.match_mode == 'hungarian', \
            'Labels (TP, FP, FN) can be computed only with hungarian match'

        if self._verbose:
            print("Adding labels...")

        self._labels_st1, self._labels_st2 = do_score_labels(self.sorting1, self.sorting2,
                                                             self.delta_frames, self.hungarian_match_12, True)

    def get_performance(self, method='by_unit', output='pandas'):
        """
        Get performance rate with several method:
          * 'raw_count' : just render the raw count table
          * 'by_unit' : render perf as rate unit by unit of the GT
          * 'pooled_with_average' : compute rate unit by unit and average

        Parameters
        ----------
        method: str
            'by_unit',  or 'pooled_with_average'
        output: str
            'pandas' or 'dict'

        Returns
        -------
        perf: pandas dataframe/series (or dict)
            dataframe/series (based on 'output') with performance entries
        """
        possibles = ('raw_count', 'by_unit', 'pooled_with_average')
        if method not in possibles:
            raise Exception("'method' can be " + ' or '.join(possibles))

        if method == 'raw_count':
            perf = self.count_score

        elif method == 'by_unit':
            perf = compute_performence(self.count_score)

        elif method == 'pooled_with_average':
            perf = self.get_performance(method='by_unit').mean(axis=0)

        if output == 'dict' and isinstance(perf, pd.Series):
            perf = perf.to_dict()

        return perf

    def print_performance(self, method='pooled_with_average'):
        """
        Print performance with the selected method
        """

        template_txt_performance = _template_txt_performance

        if method == 'by_unit':
            perf = self.get_performance(method=method, output='pandas')
            perf = perf * 100
            # ~ print(perf)
            d = {k: perf[k].tolist() for k in perf.columns}
            txt = template_txt_performance.format(method=method, **d)
            print(txt)

        elif method == 'pooled_with_average':
            perf = self.get_performance(method=method, output='pandas')
            perf = perf * 100
            txt = template_txt_performance.format(method=method, **perf.to_dict())
            print(txt)

    def print_summary(self, min_redundant_agreement=0.4, **kargs_well_detected):
        """
        Print a global performance summary that depend on the context:
          * exhaustive= True/False
          * how many gt units (one or several)
        
        This summary mix several performance metrics.
        """
        txt = _template_summary_part1

        d = dict(
            num_gt=len(self.unit1_ids),
            num_tested=len(self.unit2_ids),
            num_well_detected=self.count_well_detected_units(**kargs_well_detected),
            num_redundant=self.count_redundant_units(),
        )

        if self.exhaustive_gt:
            txt = txt + _template_summary_part2
            d['num_false_positive_units'] = self.count_false_positive_units()
            d['num_bad'] = self.count_bad_units()

        txt = txt.format(**d)

        print(txt)

    def get_well_detected_units(self, **thresholds):
        """
        Get the units in GT that are well detected with a comninaison a treshold level
        on some columns (accuracy, recall, precision, miss_rate, ...):
        
        
        
        By default threshold is {'accuray'=0.95} meaning that all units with
        accuracy above 0.95 are selected.
        
        For some thresholds columns units are below the threshold for instance
        'miss_rate', 'false_discovery_rate'
        
        If several thresh are given the the intersect of selection is kept.
        
        For instance threholds = {'accuracy':0.9, 'miss_rate':0.1 }
        give units with accuracy>0.9 AND miss<0.1
        Parameters
        ----------
        **thresholds : dict
            A dict that contains some threshold of columns of perf Dataframe.
            If sevral threhold they are combined.
        """
        if len(thresholds) == 0:
            thresholds = {'accuracy': 0.95}

        _above = ['accuracy', 'recall', 'precision']
        _below = ['false_discovery_rate', 'miss_rate']

        perf = self.get_performance(method='by_unit')
        keep = perf['accuracy'] >= 0  # tale all

        for col, thresh in thresholds.items():
            if col in _above:
                keep = keep & (perf[col] >= thresh)
            elif col in _below:
                keep = keep & (perf[col] <= thresh)
            else:
                raise ValueError('Threshold column do not exits', col)

        return perf[keep].index.tolist()

    def count_well_detected_units(self, **kargs):
        """
        Count how many well detected units.
        Kargs are the same as get_well_detected_units.
        """
        return len(self.get_well_detected_units(**kargs))

    def get_false_positive_units(self, bad_redundant_threshold=None):
        """
        Return units list of "false positive units" from tested_sorting.
        
        "false positive units" ara defined as units in tested that
        are not matched at all in GT units.
        
        Need exhaustive_gt=True

        Parameters
        ----------
        bad_redundant_threshold: float (default 0.2)
            The minimum agreement between gt and tested units
            that are best match to be counted as "false positive" units and not "redundant".

        """
        assert self.exhaustive_gt, 'false_positive_units list is valid only if exhaustive_gt=True'

        if bad_redundant_threshold is not None:
            self._bad_redundant_threshold = bad_redundant_threshold

        matched_units2 = list(self.hungarian_match_12.values)
        false_positive_ids = []
        for u2 in self.unit2_ids:
            if u2 not in matched_units2:
                if self.best_match_21[u2] == -1:
                    false_positive_ids.append(u2)
                else:
                    u1 = self.best_match_21[u2]
                    score = self.agreement_scores.at[u1, u2]
                    if score < self._bad_redundant_threshold:
                        false_positive_ids.append(u2)

        return false_positive_ids

    def count_false_positive_units(self, bad_redundant_threshold=None):
        """
        See get_false_positive_units.
        """
        return len(self.get_false_positive_units(bad_redundant_threshold))

    def get_redundant_units(self, bad_redundant_threshold=None):
        """
        Return "redundant units"
        
        
        "redundant units" are defined as units in tested
        that match a GT units with a big agreement score
        but it is not the best match.
        In other world units in GT that detected twice or more.
        
        Parameters
        ----------
        bad_redundant_threshold=None: float (default 0.2)
            The minimum agreement between gt and tested units
            that are best match to be counted as "redundant" unit and not "false positive".
        
        """
        if bad_redundant_threshold is not None:
            self._bad_redundant_threshold = bad_redundant_threshold
        matched_units2 = list(self.hungarian_match_12.values)
        redundant_ids = []
        for u2 in self.unit2_ids:
            if u2 not in matched_units2 and self.best_match_21[u2] != -1:
                u1 = self.best_match_21[u2]
                if u2 != self.best_match_12[u1]:
                    score = self.agreement_scores.at[u1, u2]
                    if score >= self._bad_redundant_threshold:
                        redundant_ids.append(u2)

        return redundant_ids

    def count_redundant_units(self, bad_redundant_threshold=None):
        """
        See get_redundant_units.
        """
        return len(self.get_redundant_units(bad_redundant_threshold=bad_redundant_threshold))

    def get_bad_units(self):
        """
        Return units list of "bad units".
        
        "bad units" are defined as units in tested that are not
        in the best match list of GT units.
        
        So it is the union of "false positive units" + "redundant units".
        
        Need exhaustive_gt=True
        """
        assert self.exhaustive_gt, 'bad_units list is valid only if exhaustive_gt=True'
        matched_units2 = list(self.hungarian_match_12.values)
        bad_ids = []
        for u2 in self.unit2_ids:
            if u2 not in matched_units2:
                bad_ids.append(u2)
        return bad_ids

    def count_bad_units(self):
        """
        See get_bad_units
        """
        return len(self.get_bad_units())


# usefull also for gathercomparison


_template_txt_performance = """PERFORMANCE ({method})
-----------
ACCURACY: {accuracy}
RECALL: {recall}
PRECISION: {precision}
FALSE DISCOVERY RATE: {false_discovery_rate}
MISS RATE: {miss_rate}
"""

_template_summary_part1 = """SUMMARY
-------
GT num_units: {num_gt}
TESTED num_units: {num_tested}
num_well_detected: {num_well_detected} 
num_redundant: {num_redundant}
"""

_template_summary_part2 = """num_false_positive_units {num_false_positive_units}
num_bad: {num_bad}
"""


def compare_sorter_to_ground_truth(gt_sorting, tested_sorting, gt_name=None, tested_name=None,
                                   delta_time=0.4, sampling_frequency=None, min_accuracy=0.5, exhaustive_gt=True,
                                   match_mode='hungarian',
                                   n_jobs=-1, bad_redundant_threshold=0.2, compute_labels=False, verbose=False):
    '''
    Compares a sorter to a ground truth.

    - Spike trains are matched based on their agreement scores
    - Individual spikes are labelled as true positives (TP), false negatives (FN),
    false positives 1 (FP), misclassifications (CL)

    It also allows to compute_performance and confusion matrix.

    Parameters
    ----------
    gt_sorting: SortingExtractor
        The first sorting for the comparison
    tested_sorting: SortingExtractor
        The second sorting for the comparison
    gt_name: str
        The name of sorter 1
    tested_name: : str
        The name of sorter 2
    delta_time: float
        Number of ms to consider coincident spikes (default 0.4 ms)
    sampling_frequency: float
        Optional sampling frequency in Hz when not included in sorting
    min_accuracy: float
        Minimum agreement score to match units (default 0.5)
    exhaustive_gt: bool (default True)
        Tell if the ground true is "exhaustive" or not. In other world if the
        GT have all possible units. It allows more performance measurement.
        For instance, MEArec simulated dataset have exhaustive_gt=True
    match_mode: 'hungarian', or 'best'
        What is match used for counting : 'hugarian' or 'best match'.
    n_jobs: int
        Number of cores to use in parallel. Uses all available if -1
    bad_redundant_threshold: float
        Agreement threshold below which a unit is considered 'false positive' and above
        which is considered 'redundant' (default 0.2)
    compute_labels: bool
        If True, labels are computed at instantiation (default True)
    verbose: bool
        If True, output is verbose
    Returns
    -------
    sorting_comparison: SortingComparison
        The SortingComparison object

    '''
    return GroundTruthComparison(gt_sorting=gt_sorting, tested_sorting=tested_sorting, gt_name=gt_name,
                                 tested_name=tested_name, delta_time=delta_time, sampling_frequency=sampling_frequency,
                                 min_accuracy=min_accuracy, exhaustive_gt=exhaustive_gt, n_jobs=n_jobs,
                                 match_mode=match_mode,
                                 compute_labels=compute_labels, bad_redundant_threshold=bad_redundant_threshold,
                                 verbose=verbose)
