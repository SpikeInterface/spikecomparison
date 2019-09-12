import numpy as np
import pandas as pd
from .comparisontools import (do_count_event, make_match_count_matrix, make_agreement_scores_from_count)


class BaseComparison:
    """
    Base class for all comparison class : 
       * GroundTruthComparison
       * MultiSortingComparison
       * SymmetricSortingComparison
    
    Mainly deal with:
      * sampling_frequency
      * sorting names
      * delta_time to delta_frames
    """
    def __init__(self, sorting_list, name_list=None, delta_time=0.3, sampling_frequency=None, 
                min_accuracy=0.5, n_jobs=-1, verbose=False):
        
        self.sorting_list = sorting_list
        if name_list is None:
            name_list = ['sorting {}'.format(i+1) for i in range(len(sorting_list))]
        self.name_list = name_list
        
        if sampling_frequency is None:
            # take sampling frequency from sorting list and test that they are equivalent.
            sampling_freqs_not_none = np.array([s.get_sampling_frequency() for s in self.sorting_list
                              if s.get_sampling_frequency() is not None], dtype='float64')
            assert len(sampling_freqs_not_none) > 0 , ("Sampling frequency information "
                "not found in sorting. Pass it with the 'sampling_frequency' argument")
            
            # Some sorter round the sampling freq lets emit a warning
            sf0 = sampling_freqs_not_none[0]
            if not np.all([sf == sf0 for sf in sampling_freqs_not_none]):
                delta_freq_ratio = np.abs(sampling_freqs_not_none - sf0) / sf0
                # tolerence of 0.1%
                assert np.all(delta_freq_ratio < 0.001), "Inconsintent sampling frequency among sorting list" 

            sampling_frequency = sampling_freqs_not_none[0]

        self.sampling_frequency = sampling_frequency
        self.delta_time = delta_time
        self.delta_frames = int(self.delta_time / 1000 * self.sampling_frequency)
        self.min_accuracy = min_accuracy
        self.n_jobs = n_jobs
        self.verbose = verbose
    


class BaseTwoSorterComparison(BaseComparison):
    """
    Base class shared by SortingComparison and GroundTruthComparison
    """
    def __init__(self, sorting1, sorting2, sorting1_name=None, sorting2_name=None, 
            delta_time=0.3, sampling_frequency=None, min_accuracy=0.5, n_jobs=1, verbose=False):
        
        sorting_list = [sorting1, sorting2]
        if sorting1_name is None:
            sorting1_name = 'sorting 1'
        if sorting2_name is None:
            sorting2_name = 'sorting 2'
        name_list = [sorting1_name, sorting2_name]
        
        BaseComparison.__init__(self, sorting_list, name_list=name_list, delta_time=delta_time,
                sampling_frequency=sampling_frequency, min_accuracy=min_accuracy,
                n_jobs=n_jobs, verbose=verbose)
        
    
        self.unit1_ids = self.sorting1.get_unit_ids()
        self.unit2_ids = self.sorting2.get_unit_ids()
        
        self._do_agreement() 
        self._do_matching()

    @property
    def sorting1(self):
        return self.sorting_list[0]

    @property
    def sorting2(self):
        return self.sorting_list[1]

    @property
    def sorting1_name(self):
        return self.name_list[0]

    @property
    def sorting2_name(self):
        return self.name_list[1]
    
    def _do_agreement(self):
        # common to GroundTruthComparison and SymmetricSortingComparison
        # because it is symetric
        
        # spike count for each spike train
        self.event_counts1 = do_count_event(self.sorting1)
        self.event_counts2 = do_count_event(self.sorting2)

        # matrix of  event match count for each pair
        self.match_event_count = make_match_count_matrix(self.sorting1, self.sorting2, self.delta_frames, n_jobs=self.n_jobs)
        
        # agreement matrix score for each pair
        self.agreement_scores = make_agreement_scores_from_count(self.match_event_count, self.event_counts1, self.event_counts2)

    def _do_matching(self):
        # must be implemented in subclass
        raise(NotImplementedError)
    



