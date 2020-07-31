import pandas as pd
import numpy as np
from .groundtruthcomparison import GroundTruthComparison
#~ from .comparisontools import (do_score_labels, make_possible_match,
                              #~ make_best_match, make_hungarian_match, do_confusion_matrix, do_count_score,
                              #~ compute_performence)
from .comparisontools import make_collision_events


class CollisionGTComparison(GroundTruthComparison):
    """
    This class is an extention of GroundTruthComparison by focusing
    to benchmark spike in collision
    
    
    collision_lag: float 
        Collision lag in ms.
    
    """

    def __init__(self, gt_sorting, tested_sorting, collision_lag=2.0, **kwargs):
        
        # Force compute labels
        kwargs['compute_labels'] = True
        
        GroundTruthComparison.__init__(self, gt_sorting, tested_sorting, **kwargs)
        
        self.collision_lag = collision_lag
        self.detect_gt_collision()
        
    def detect_gt_collision(self):
        
        delta = int(self.collision_lag / 1000 * self.sampling_frequency)
        
        
        self.collision_events = make_collision_events(self.sorting1, delta)
        print(self.collision_events.size)
    
    def get_label_for_collision(self, gt_unit_id1, gt_unit_id2):
        if gt_unit_id1 > gt_unit_id2:
            gt_unit_id1, gt_unit_id2 = gt_unit_id2, gt_unit_id1
        
        # events
        mask = (self.collision_events['unit_id1'] == gt_unit_id1) & (self.collision_events['unit_id2'] == gt_unit_id2)
        event = self.collision_events[mask]
        
        #~ print(event['index1'])
        #~ print(self._labels_st1)
        score_label1 = self._labels_st1[gt_unit_id1][event['index1']]
        score_label2 = self._labels_st1[gt_unit_id2][event['index2']]
        delta = event['delta_frame']
        
        return score_label1, score_label2, delta
        
    
