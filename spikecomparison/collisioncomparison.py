import pandas as pd
import numpy as np
from .groundtruthcomparison import GroundTruthComparison
#~ from .comparisontools import (do_score_labels, make_possible_match,
                              #~ make_best_match, make_hungarian_match, do_confusion_matrix, do_count_score,
                              #~ compute_performence)


class CollisionGTComparison(GroundTruthComparison):
    """
    This class is an extention of GroundTruthComparison by focusing
    to benchmark spike in collision
    """

    def __init__(self, *args, **kwargs):
        GroundTruthComparison.__init__(*args, **kwargs)
        
