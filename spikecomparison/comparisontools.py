"""
Some functions internally use by SortingComparison.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment


def count_matching_events(times1, times2, delta=10):
    """
    Counts matching events.

    Parameters
    ----------
    times1: list
        List of spike train 1 frames
    times2: list
        List of spike train 2 frames
    delta: int
        Number of frames for considering matching events

    Returns
    -------
    matching_count: int
        Number of matching events
    """
    times_concat = np.concatenate((times1, times2))
    membership = np.concatenate((np.ones(times1.shape) * 1, np.ones(times2.shape) * 2))
    indices = times_concat.argsort()
    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    diffs = times_concat_sorted[1:] - times_concat_sorted[:-1]
    inds = np.where((diffs <= delta) & (membership_sorted[0:-1] != membership_sorted[1:]))[0]
    if len(inds) == 0:
        return 0
    inds2 = np.where(inds[:-1] + 1 != inds[1:])[0]
    return len(inds2) + 1


def compute_agreement_score(num_matches, num1, num2):
    """
    Computes agreement score.

    Parameters
    ----------
    num_matches: int
        Number of matches
    num1: int
        Number of events in spike train 1
    num2: int
        Number of events in spike train 2

    Returns
    -------
    score: float
        Agreement score
    """
    denom = num1 + num2 - num_matches
    if denom == 0:
        return 0
    return num_matches / denom

def do_count_event(sorting):
    """
    Count event for each units in a sorting.
    Parameters
    ----------
    sorting: SortingExtractor
        A sorting extractor
    
    Returns
    -------
    event_count: pd.Series
        Nb of spike by units.
    """
    unit_ids = sorting.get_unit_ids()
    ev_counts = np.array([len(sorting.get_unit_spike_train(u)) for u in unit_ids], dtype='int64')
    event_counts = pd.Series(ev_counts, index=unit_ids)
    return event_counts

def count_match_spikes(times1, all_times2,  delta_frames): # , event_counts1, event_counts2  unit2_ids,
    """
    Computes matching spikes between one spike train and a list of others.

    Parameters
    ----------
    times1: array
        Spike train 1 frames
    all_times2: list of array
        List of spike trains from sorting 2

    Returns
    -------
    matching_events_count: list
        List of counts of matching events
    """
    matching_event_counts = np.zeros(len(all_times2), dtype='int64')
    for i2, times2 in enumerate(all_times2):
        num_matches = count_matching_events(times1, times2, delta=delta_frames)
        matching_event_counts[i2] = num_matches
    return matching_event_counts


def make_match_count_matrix(sorting1, sorting2, delta_frames, n_jobs=1):
    """
    Make the match_event_count matrix.
    Basically it count the match event for all given pair of spike train from
    sorting1 and sorting2.
    
    Parameters
    ----------
    sorting1: SortingExtractor
        The first sorting extractor

    sorting2: SortingExtractor
        The second sorting extractor

    delta_frames: int
        Number of frames to consider spikes coincident

    n_jobs: int
        Number of jobs to run in parallel

    Returns
    -------
    
    match_event_count: array (int64)
        Matrix of match count spike
    
    """
    unit1_ids = np.array(sorting1.get_unit_ids())
    unit2_ids = np.array(sorting2.get_unit_ids())
    
    # preload all spiketrains 2 into a list
    s2_spiketrains = [sorting2.get_unit_spike_train(u2) for u2 in unit2_ids]
    
    match_event_count_lists = Parallel(n_jobs=n_jobs)(delayed(count_match_spikes)(sorting1.get_unit_spike_train(u1), 
                            s2_spiketrains,delta_frames) for i1, u1 in enumerate(unit1_ids))
    
    match_event_count = pd.DataFrame(np.array(match_event_count_lists),
                                                    index=unit1_ids, columns=unit2_ids)
    
    return match_event_count


def make_agreement_scores(sorting1, sorting2, delta_frames, n_jobs=1):
    """
    Make the agreement matrix.
    No threshold (min_accuracy) is applied at this step.
    
    Note : this computation is symetric.
    Inverting sorting1 and sorting2 give the transposed matrix.
    

    Parameters
    ----------
    sorting1: SortingExtractor
        The first sorting extractor

    sorting2: SortingExtractor
        The second sorting extractor

    delta_frames: int
        Number of frames to consider spikes coincident

    n_jobs: int
        Number of jobs to run in parallel

    Returns
    -------
    
    agreement_scores: array (float)
        The agreement score matrix.
    
    """
    unit1_ids = np.array(sorting1.get_unit_ids())
    unit2_ids = np.array(sorting2.get_unit_ids())

    ev_counts1 = np.array([len(sorting1.get_unit_spike_train(u1)) for u1 in unit1_ids], dtype='int64')
    ev_counts2 = np.array([len(sorting2.get_unit_spike_train(u2)) for u2 in unit2_ids], dtype='int64')
    event_counts1 = pd.Series(ev_counts1, index=unit1_ids)
    event_counts2 = pd.Series(ev_counts2, index=unit2_ids)

    match_event_count = make_match_count_matrix(sorting1, sorting2, delta_frames, n_jobs=n_jobs)
    
    agreement_scores = make_agreement_scores_from_count(match_event_count, event_counts1, event_counts2)
    
    return agreement_scores


def make_agreement_scores_from_count(match_event_count, event_counts1, event_counts2):
    """
    See make_agreement_scores.
    Other signature here to avoid to recompute match_event_count matrix.
    
    Parameters
    ----------
    
    match_event_count
    
    
    """

    # numpy broadcast style
    denom = event_counts1.values[:, None] + event_counts2.values[None, :] - match_event_count.values
    # little trick here when denom is 0 to avoid 0 division : lets put -1
    # it will 0 anyway
    denom[denom == 0] = -1
    
    agreement_scores = match_event_count.values / denom
    agreement_scores = pd.DataFrame(agreement_scores, 
                        index=match_event_count.index, columns=match_event_count.columns)
    return agreement_scores


def make_possible_match(sorting1, sorting2, agreement_scores, min_accuracy):
    """
    Given an agreement matrix and a min_accuracy threhold.
    Return as a dict all possible match for each spiketrain in each side.
    
    Note : this is symmetric.
    
    """
    unit1_ids = np.array(sorting1.get_unit_ids())
    unit2_ids = np.array(sorting2.get_unit_ids())
    
    # threhold the matrix
    scores = agreement_scores.copy()
    scores[agreement_scores<min_accuracy] =0
    
    possible_match_12 =  {}
    for i1, u1 in enumerate(unit1_ids):
        inds_match, = np.nonzero(scores[i1, :])
        possible_match_12[u1] = unit2_ids[inds_match]

    possible_match_21 =  {}
    for i2, u2 in enumerate(unit2_ids):
        inds_match, = np.nonzero(scores[:, i2])
        possible_match_21[u2] = unit1_ids[inds_match]
    
    return possible_match_12, possible_match_21
    
    
    

def make_best_match(sorting1, sorting2, agreement_scores, min_accuracy):
    """
    Given an agreement matrix and a min_accuracy threhold.
    return a dict a best match for each units independently of others.
    
    Note : this is symmetric.
    
    
    """
    unit1_ids = np.array(sorting1.get_unit_ids())
    unit2_ids = np.array(sorting2.get_unit_ids())

    best_match_12 = pd.Series(index=unit1_ids, dtype='int64')
    for i1, u1 in enumerate(unit1_ids):
        ind_max = np.argmax(agreement_scores[i1, :])
        if agreement_scores[i1, ind_max] >= min_accuracy:
            best_match_12[u1] = unit2_ids[ind_max]
        else:
            best_match_12[u1] = -1

    best_match_21 = pd.Series(index=unit2_ids, dtype='int64')
    for i2, u2 in enumerate(unit2_ids):
        ind_max = np.argmax(agreement_scores[:, i2])
        if agreement_scores[ind_max, i2] >= min_accuracy:
            best_match_21[u2] = unit1_ids[ind_max]
        else:
            best_match_21[u2] = -1
    
    return best_match_12, best_match_21

    
def make_hungarian_match(sorting1, sorting2, agreement_scores, min_accuracy):
    """
    Given an agreement matrix and a min_accuracy threhold.
    return the "optimal" match with the "hungarian" algo.
    This use internally the scipy.optimze.linear_sum_assignment implementation.
    
    """

    unit1_ids = sorting1.get_unit_ids()
    unit2_ids = sorting2.get_unit_ids()

    # threhold the matrix
    scores = agreement_scores.copy()
    scores[agreement_scores<min_accuracy] =0
    
    [inds1, inds2] = linear_sum_assignment(-scores)
    
    hungarian_match_12 =  pd.Series(index=unit1_ids, dtype='int64')
    for i1, u1 in enumerate(unit1_ids):
        if i1 in inds1:
            ind = np.nonzero(inds1==i1)[0][0]
            hungarian_match_12[u1] = unit2_ids[inds2[ind]]
        else:
            hungarian_match_12[u1] = -1
    
    hungarian_match_21 =  pd.Series(index=unit2_ids, dtype='int64')
    for i2, u2 in enumerate(unit2_ids):
        if i2 in inds2:
            ind = np.nonzero(inds2==i2)[0][0]
            hungarian_match_21[u2] = unit1_ids[inds1[ind]]
        else:
            hungarian_match_21[u2] = -1
    
    return hungarian_match_12, hungarian_match_21
    


#~ def do_matching(sorting1, sorting2, delta_frames, min_accuracy, n_jobs=1):
    #~ """
    #~ Computes the matching between 2 sorters.

    #~ Parameters
    #~ ----------
    #~ sorting1: SortingExtractor
        #~ The first sorting extractor

    #~ sorting2: SortingExtractor
        #~ The second sorting extractor

    #~ delta_frames: int
        #~ Number of frames to consider spikes coincident

    #~ n_jobs: int
        #~ Number of jobs to run in parallel

    #~ Returns
    #~ -------

    #~ event_counts_1: dict
        #~ Dictionary with unit as keys and number of spikes as values for sorting1
    #~ event_counts_2: dict
        #~ Dictionary with unit as keys and number of spikes as values for sorting2
    #~ matching_event_counts_12: dict
        #~ Dictionary with matching counts for each unit of sorting1 vs all units of sorting2
    #~ best_match_units_12: dict
        #~ Dictionary with best matched unit for each unit in sorting1
    #~ matching_event_counts_21: dict
        #~ Dictionary with matching counts for each unit of sorting2 vs all units of sorting1
    #~ best_match_units_21: dict
        #~ Dictionary with best matched unit of sorting2 for each unit in sorting1
    #~ unit_map12: dict
        #~ Mapped units for sorting1 to sorting2 using linear assignment. If units are not matched they are -1
    #~ unit_map21: dict
        #~ Mapped units for sorting2 to sorting1 using linear assignment. If units are not matched they are -1

    #~ """
    #~ event_counts_1 = dict()
    #~ event_counts_2 = dict()
    #~ matching_event_counts_12 = dict()
    #~ best_match_units_12 = dict()
    #~ matching_event_counts_21 = dict()
    #~ best_match_units_21 = dict()
    #~ unit_map12 = dict()
    #~ unit_map21 = dict()

    #~ unit1_ids = sorting1.get_unit_ids()
    #~ unit2_ids = sorting2.get_unit_ids()
    #~ N1 = len(unit1_ids)
    #~ N2 = len(unit2_ids)

    #~ # Compute events counts
    #~ event_counts1 = np.zeros((N1)).astype(np.int64)
    #~ for i1, u1 in enumerate(unit1_ids):
        #~ times1 = sorting1.get_unit_spike_train(u1)
        #~ event_counts1[i1] = len(times1)
        #~ event_counts_1[u1] = len(times1)
    #~ event_counts2 = np.zeros((N2)).astype(np.int64)
    #~ for i2, u2 in enumerate(unit2_ids):
        #~ times2 = sorting2.get_unit_spike_train(u2)
        #~ event_counts2[i2] = len(times2)
        #~ event_counts_2[u2] = len(times2)

    #~ # Compute matching events
    #~ s2_spiketrains = [sorting2.get_unit_spike_train(u2) for u2 in unit2_ids]
    #~ results = Parallel(n_jobs=n_jobs)(delayed(match_spikes)(sorting1.get_unit_spike_train(u1), s2_spiketrains,
                                                            #~ unit2_ids, delta_frames, event_counts1[i1],
                                                            #~ event_counts2) for i1, u1 in enumerate(unit1_ids))

    #~ matching_event_counts = np.zeros((N1, N2)).astype(np.int64)
    #~ scores = np.zeros((N1, N2))
    #~ for i1, u1 in enumerate(unit1_ids):
        #~ matching_event_counts[i1] = results[i1][0]
        #~ scores[i1] = results[i1][1]

    #~ # Find best matches for spiketrains 1
    #~ for i1, u1 in enumerate(unit1_ids):
        #~ scores0 = scores[i1, :]
        #~ matching_event_counts_12[u1] = dict()
        #~ if len(scores0) > 0:
            #~ if np.max(scores0) > 0:
                #~ inds0 = np.where(scores0 > 0)[0]
                #~ for i2 in inds0:
                    #~ matching_event_counts_12[u1][unit2_ids[i2]] = matching_event_counts[i1, i2]
                #~ i2_best = np.argmax(scores0)
                #~ best_match_units_12[u1] = unit2_ids[i2_best]
            #~ else:
                #~ best_match_units_12[u1] = -1
        #~ else:
            #~ best_match_units_12[u1] = -1

    #~ # Find best matches for spiketrains 2
    #~ for i2, u2 in enumerate(unit2_ids):
        #~ scores0 = scores[:, i2]
        #~ matching_event_counts_21[u2] = dict()
        #~ if len(scores0) > 0:
            #~ if np.max(scores0) > 0:
                #~ inds0 = np.where(scores0 > 0)[0]
                #~ for i1 in inds0:
                    #~ matching_event_counts_21[u2][unit1_ids[i1]] = matching_event_counts[i1, i2]
                #~ i1_best = np.argmax(scores0)
                #~ best_match_units_21[u2] = unit1_ids[i1_best]
            #~ else:
                #~ best_match_units_21[u2] = -1
        #~ else:
            #~ best_match_units_21[u2] = -1

    #~ # Assign best matches
    #~ [inds1, inds2] = linear_sum_assignment(-scores)
    #~ inds1 = list(inds1)
    #~ inds2 = list(inds2)
    #~ if len(unit2_ids) > 0:
        #~ k2 = np.max(unit2_ids) + 1
    #~ else:
        #~ k2 = 1
    #~ for i1, u1 in enumerate(unit1_ids):
        #~ if i1 in inds1:
            #~ aa = inds1.index(i1)
            #~ i2 = inds2[aa]
            #~ u2 = unit2_ids[i2]
            #~ # criteria on agreement_score
            #~ num_matches = matching_event_counts_12[u1].get(u2, 0)
            #~ num1 = event_counts_1[u1]
            #~ num2 = event_counts_2[u2]
            #~ agree_score = compute_agreement_score(num_matches, num1, num2)
            #~ if agree_score > min_accuracy:
                #~ unit_map12[u1] = u2
            #~ else:
                #~ unit_map12[u1] = -1
        #~ else:
            #~ # unit_map12[u1] = k2
            #~ # k2 = k2+1
            #~ unit_map12[u1] = -1
    #~ if len(unit1_ids) > 0:
        #~ k1 = np.max(unit1_ids) + 1
    #~ else:
        #~ k1 = 1
    #~ for i2, u2 in enumerate(unit2_ids):
        #~ if i2 in inds2:
            #~ aa = inds2.index(i2)
            #~ i1 = inds1[aa]
            #~ u1 = unit1_ids[i1]
            #~ # criteria on agreement_score
            #~ num_matches = matching_event_counts_12[u1].get(u2, 0)
            #~ num1 = event_counts_1[u1]
            #~ num2 = event_counts_2[u2]
            #~ agree_score = compute_agreement_score(num_matches, num1, num2)
            #~ if agree_score > min_accuracy:
                #~ unit_map21[u2] = u1
            #~ else:
                #~ unit_map21[u2] = -1
        #~ else:
            #~ # unit_map21[u2] = k1
            #~ # k1 = k1+1
            #~ unit_map21[u2] = -1

    #~ return (event_counts_1, event_counts_2,
            #~ matching_event_counts_12, best_match_units_12,
            #~ matching_event_counts_21, best_match_units_21,
            #~ unit_map12, unit_map21)


def do_score_labels(sorting1, sorting2, delta_frames, unit_map12, label_misclassification=False):
    """
    Makes the labelling at spike level for each spike train:
      * TP: true positive
      * CL: classification error
      * FN: False negative
      * FP: False positive
      * TOT:
      * TOT_ST1:
      * TOT_ST2:

    Parameters
    ----------
    sorting1: SortingExtractor instance
        The ground truth sorting
    sorting2: SortingExtractor instance
        The tested sorting
    delta_frames: int
        Number of frames to consider spikes coincident
    unit_map12: dict
        Dict of matching from sorting1 to sorting2
    label_misclassification: bool
        If True, misclassification errors are labelled

    Returns
    -------
    labels_st1: dict of np.array of str
        Contain score labels for units of sorting 1
    labels_st2: dict of np.array of str
        Contain score labels for units of sorting 2
    """
    unit1_ids = sorting1.get_unit_ids()
    unit2_ids = sorting2.get_unit_ids()
    labels_st1 = dict()
    labels_st2 = dict()
    N1 = len(unit1_ids)
    N2 = len(unit2_ids)

    # copy spike trains for faster access from extractors with memmapped data
    sts1 = {u1: sorting1.get_unit_spike_train(u1) for u1 in unit1_ids}
    sts2 = {u2: sorting2.get_unit_spike_train(u2) for u2 in unit2_ids}

    for u1 in unit1_ids:
        lab_st1 = np.array(['UNPAIRED'] * len(sts1[u1]), dtype='<U8')
        labels_st1[u1] = lab_st1
    for u2 in unit2_ids:
        lab_st2 = np.array(['UNPAIRED'] * len(sts2[u2]), dtype='<U8')
        labels_st2[u2] = lab_st2

    for u1 in unit1_ids:
        u2 = unit_map12[u1]
        if u2 != -1:
            lab_st1 = labels_st1[u1]
            lab_st2 = labels_st2[u2]
            mapped_st = sorting2.get_unit_spike_train(u2)
            # from gtst: TP, TPO, TPSO, FN, FNO, FNSO
            for sp_i, n_sp in enumerate(sts1[u1]):
                matches = (np.abs(mapped_st.astype(int) - n_sp) <= delta_frames // 2)
                if np.sum(matches) > 0:
                    if lab_st1[sp_i] != 'TP' and lab_st2[np.where(matches)[0][0]] != 'TP':
                        lab_st1[sp_i] = 'TP'
                        lab_st2[np.where(matches)[0][0]] = 'TP'
        else:
            lab_st1 = np.array(['FN'] * len(sts1[u1]))
            labels_st1[u1] = lab_st1

    if label_misclassification:
        for u1 in unit1_ids:
            lab_st1 = labels_st1[u1]
            st1 = sts1[u1]
            for l_gt, lab in enumerate(lab_st1):
                if lab == 'UNPAIRED':
                    for u2 in unit2_ids:
                        if u2 in unit_map12.values() and unit_map12[u1] != -1:
                            lab_st2 = labels_st2[u2]
                            n_sp = st1[l_gt]
                            mapped_st = sts2[u2]
                            matches = (np.abs(mapped_st.astype(int) - n_sp) <= delta_frames // 2)
                            if np.sum(matches) > 0:
                                if 'CL' not in lab_st1[l_gt] and 'CL' not in lab_st2[np.where(matches)[0][0]]:
                                    lab_st1[l_gt] = 'CL_' + str(u1) + '_' + str(u2)
                                    lab_st2[np.where(matches)[0][0]] = 'CL_' + str(u2) + '_' + str(u1)

    for u1 in unit1_ids:
        lab_st1 = labels_st1[u1]
        lab_st1[lab_st1 == 'UNPAIRED'] = 'FN'
        # for l_gt, lab in enumerate(lab_st1):
        #     if lab == 'UNPAIRED':
        #         lab_st1[l_gt] = 'FN'

    for u2 in unit2_ids:
        lab_st2 = labels_st2[u2]
        lab_st2[lab_st2 == 'UNPAIRED'] = 'FP'
        # for l_gt, lab in enumerate(lab_st2):
        #     if lab == 'UNPAIRED':
        #         lab_st2[l_gt] = 'FP'

    return labels_st1, labels_st2

def compare_spike_trains(spiketrain1, spiketrain2, delta_frames=10):
    """
    Compares 2 spike trains.

    Note:
      * The first spiketrain is supposed to be the ground truth.
      * this implementation do not count a TP when more than one spike
        is present around the same time in spiketrain2.

    Parameters
    ----------
    spiketrain1,spiketrain2: numpy.array
        Times of spikes for the 2 spike trains.

    Returns
    -------
    lab_st1, lab_st2: numpy.array
        Label of score for each spike

    """
    lab_st1 = np.array(['UNPAIRED'] * len(spiketrain1))
    lab_st2 = np.array(['UNPAIRED'] * len(spiketrain2))

    # from gtst: TP, TPO, TPSO, FN, FNO, FNSO
    for sp_i, n_sp in enumerate(spiketrain1):
        matches = (np.abs(spiketrain2.astype(int) - n_sp) <= delta_frames // 2)
        if np.sum(matches) > 0:
            if lab_st1[sp_i] != 'TP' and lab_st2[np.where(matches)[0][0]] != 'TP':
                lab_st1[sp_i] = 'TP'
                lab_st2[np.where(matches)[0][0]] = 'TP'

    for l_gt, lab in enumerate(lab_st1):
        if lab == 'UNPAIRED':
            lab_st1[l_gt] = 'FN'

    for l_gt, lab in enumerate(lab_st2):
        if lab == 'UNPAIRED':
            lab_st2[l_gt] = 'FP'

    return lab_st1, lab_st2


def do_confusion_matrix(event_counts1, event_counts2, match_12, match_event_count):
    """
    Computes the confusion matrix between two sorting.

    Parameters
    ----------
    event_counts1: pd.Series
        
    event_counts2: pd.Series
        
    match_12: pd.Series
        Series of matching from sorting1 to sorting2.
        Can be the hungarian or best match.
    match_event_count: pd.DataFrame
        The match count matrix given by make_match_count_matrix
    Returns
    ------
    confusion_matrix: pd.DataFrame
        The confusion matrix
        index are units1 reordered
        columns are units2 redordered
    """
    unit1_ids = np.array(match_event_count.index)
    unit2_ids = np.array(match_event_count.columns)
    N1 = len(unit1_ids)
    N2 = len(unit2_ids)
    
    matched_units1 = match_12[match_12 != -1].index
    matched_units2 = match_12[match_12 != -1].values
    
    unmatched_units1 = match_12[match_12 == -1].index
    unmatched_units2 = unit2_ids[~np.in1d(unit2_ids, matched_units2)]
    
    ordered_units1 = np.hstack([matched_units1, unmatched_units1])
    ordered_units2 = np.hstack([matched_units2, unmatched_units2])
    
    conf_matrix = pd.DataFrame(np.zeros((N1 + 1, N2 + 1), dtype=int),
                        index=list(ordered_units1)+['FP'],
                        columns=list(ordered_units2)+['FN'])
    
    for u1 in matched_units1:
        u2 = match_12[u1]
        num_match = match_event_count.at[u1, u2]
        conf_matrix.at[u1, u2] = num_match
        conf_matrix.at[u1, 'FN'] = event_counts1.at[u1] - num_match
        conf_matrix.at['FP', u2] = event_counts2.at[u2] - num_match
    
    for u1 in unmatched_units1:
        conf_matrix.at[u1, 'FN'] = event_counts1.at[u1]
    
    for u2 in unmatched_units2:
        conf_matrix.at['FP', u2] = event_counts2.at[u2]
    
    return conf_matrix
    
    
    mapped_units = np.array(list(unit_map12.values()))
    idxs_matched, = np.where(mapped_units != -1)
    idxs_unmatched, = np.where(mapped_units == -1)
    unit_map_matched = mapped_units[idxs_matched]

    st1_idxs = np.hstack([unit1_ids[idxs_matched], unit1_ids[idxs_unmatched]])
    st2_matched = unit_map_matched
    
    
    
    
    

    conf_matrix = np.zeros((N1 + 1, N2 + 1), dtype=int)

    mapped_units = np.array(list(unit_map12.values()))
    idxs_matched, = np.where(mapped_units != -1)
    idxs_unmatched, = np.where(mapped_units == -1)
    unit_map_matched = mapped_units[idxs_matched]

    st1_idxs = np.hstack([unit1_ids[idxs_matched], unit1_ids[idxs_unmatched]])
    st2_matched = unit_map_matched
    st2_unmatched = []

    for u_i, u1 in enumerate(unit1_ids[idxs_matched]):
        lab_st1 = labels_st1[u1]
        tp = len(np.where('TP' == lab_st1)[0])
        conf_matrix[u_i, u_i] = int(tp)
        for u_j, u2 in enumerate(unit2_ids):
            lab_st2 = labels_st2[u2]
            cl_str = str(u1) + '_' + str(u2)
            cl = len([i for i, v in enumerate(lab_st1) if 'CL' in v and cl_str in v])
            if cl != 0:
                st_p, = np.where(u2 == unit_map_matched)
                conf_matrix[u_i, st_p] = int(cl)
        fn = len(np.where('FN' == lab_st1)[0])
        conf_matrix[u_i, -1] = int(fn)

    for u_i, u1 in enumerate(unit1_ids[idxs_unmatched]):
        lab_st1 = labels_st1[u1]
        fn = len(np.where('FN' == lab_st1)[0])
        conf_matrix[u_i + len(idxs_matched), -1] = int(fn)

    for u_j, u2 in enumerate(unit2_ids):
        lab_st2 = labels_st2[u2]
        fp = len(np.where('FP' == lab_st2)[0])
        st_p, = np.where(u2 == unit_map_matched)
        if len(st_p) != 0:
            conf_matrix[-1, st_p] = int(fp)
        else:
            st2_unmatched.append(int(u2))
            conf_matrix[-1, len(idxs_matched) + len(st2_unmatched) - 1] = int(fp)

    st2_idxs = np.hstack([st2_matched, st2_unmatched]).astype('int64')

    return conf_matrix, st1_idxs, st2_idxs



