import numpy as np
import spikeextractors as se
from scipy.optimize import linear_sum_assignment
from .basecomparison import BaseComparison
from .symmetricsortingcomparison import SymmetricSortingComparison
from .comparisontools import compare_spike_trains

import networkx as nx


class MultiSortingComparison(BaseComparison):
    def __init__(self, sorting_list, name_list=None, delta_time=0.4, sampling_frequency=None,
                 match_score=0.5, chance_score=0.1, n_jobs=-1, verbose=False):

        BaseComparison.__init__(self, sorting_list, name_list=name_list,
                                delta_time=delta_time, sampling_frequency=sampling_frequency,
                                match_score=match_score, chance_score=chance_score,
                                n_jobs=n_jobs, verbose=verbose)
        self._do_matching(verbose)

    def get_sorting_list(self):
        return self.sorting_list

    def get_agreement_sorting(self, minimum_matching=0):
        sorting = AgreementSortingExtractor(self, min_agreement=minimum_matching)
        sorting.set_sampling_frequency(self.sampling_frequency)
        return sorting

    def _do_matching(self, verbose):
        # do pairwise matching
        if self._verbose:
            print('Multicomaprison step1: pairwise comparison')

        self.comparisons = []
        for i in range(len(self.sorting_list)):
            for j in range(i + 1, len(self.sorting_list)):
                if verbose:
                    print("  Comparing: ", self.name_list[i], " and ", self.name_list[j])
                comp = SymmetricSortingComparison(self.sorting_list[i], self.sorting_list[j],
                                                  sorting1_name=self.name_list[i],
                                                  sorting2_name=self.name_list[j],
                                                  delta_time=self.delta_time,
                                                  sampling_frequency=self.sampling_frequency,
                                                  match_score=self.match_score,
                                                  n_jobs=self._n_jobs,
                                                  verbose=False)
                self.comparisons.append(comp)

        if self._verbose:
            print('Multicomaprison step2: make graph')

        self.graph = nx.Graph()
        # nodes
        for i, sorting in enumerate(self.sorting_list):
            sorter_name = self.name_list[i]
            for unit in sorting.get_unit_ids():
                node_name = str(sorter_name) + '_' + str(unit)
                self.graph.add_node(node_name)
        # edges
        for comp in self.comparisons:
            for u1 in comp.sorting1.get_unit_ids():
                u2 = comp.hungarian_match_12[u1]
                if u2 != -1:
                    node1_name = str(comp.sorting1_name) + '_' + str(u1)
                    node2_name = str(comp.sorting2_name) + '_' + str(u2)
                    score = comp.agreement_scores.loc[u1, u2]
                    self.graph.add_edge(node1_name, node2_name, weight=score)

        # the graph is symmetrical
        self.graph = self.graph.to_undirected()

        # extract agrrement from graph
        if self._verbose:
            print('Multicomaprison step3: extract agreement from graph')

        self._new_units = {}
        self._spiketrains = []
        added_nodes = []
        unit_id = 0

        # Note in this graph node=one unit for one sorter
        for node in self.graph.nodes():
            edges = self.graph.edges(node, data=True)
            sorter, unit = (str(node)).split('_')
            unit = int(unit)
            if len(edges) == 0:
                avg_agr = 0
                sorting_idxs = {sorter: unit}
                self._new_units[unit_id] = {'avg_agreement': avg_agr,
                                            'sorter_unit_ids': sorting_idxs}
                unit_id += 1
                added_nodes.append(str(node))
            else:
                # check if other nodes have edges (we should also check edges of
                all_edges = list(edges)
                for e in edges:
                    # Note for alessio n1>node1 n2>node2 e>edge
                    n1, n2, d = e
                    n2_edges = self.graph.edges(n2, data=True)
                    if len(n2_edges) > 0:  # useless line if
                        for e_n in n2_edges:
                            n_n1, n_n2, d = e_n
                            # Note for alessio  why do do you sorter each elements in the all_edges ?
                            if sorted([n_n1, n_n2]) not in [sorted([u, v]) for u, v, _ in all_edges]:
                                all_edges.append(e_n)
                avg_agr = np.mean([d['weight'] for u, v, d in all_edges])
                max_edge = list(all_edges)[np.argmax([d['weight'] for u, v, d in all_edges])]

                for edge in all_edges:
                    n1, n2, d = edge
                    if n1 not in added_nodes or n2 not in added_nodes:
                        sorter1, unit1 = n1.split('_')
                        sorter2, unit2 = n2.split('_')
                        unit1 = int(unit1)
                        unit2 = int(unit2)
                        sorting_idxs = {sorter1: unit1, sorter2: unit2}
                        if unit_id not in self._new_units.keys():
                            self._new_units[unit_id] = {'avg_agreement': avg_agr,
                                                        'sorter_unit_ids': sorting_idxs}
                        else:
                            full_sorting_idxs = self._new_units[unit_id]['sorter_unit_ids']
                            for s, u in sorting_idxs.items():
                                if s not in full_sorting_idxs:
                                    full_sorting_idxs[s] = u
                            self._new_units[unit_id] = {'avg_agreement': avg_agr,
                                                        'sorter_unit_ids': full_sorting_idxs}
                        added_nodes.append(str(node))
                        if n1 not in added_nodes:
                            added_nodes.append(str(n1))
                        if n2 not in added_nodes:
                            added_nodes.append(str(n2))
                unit_id += 1

        # extract best matches true positive spike trains
        if self._verbose:
            print('multicomaprison step4 : make agreement spiketrains')

        for u, v in self._new_units.items():
            # count matched number
            matched_num = len(v['sorter_unit_ids'].keys())
            v['matched_number'] = matched_num
            self._new_units[u] = v

            if len(v['sorter_unit_ids'].keys()) == 1:
                self._spiketrains.append(self.sorting_list[self.name_list.index(
                    list(v['sorter_unit_ids'].keys())[0])].get_unit_spike_train(list(v['sorter_unit_ids'].values())[0]))
            else:
                nodes = []
                edges = []
                for sorter, unit in v['sorter_unit_ids'].items():
                    nodes.append((sorter + '_' + str(unit)))
                for n1 in nodes:
                    for n2 in nodes:
                        if n1 != n2:
                            if (n1, n2) not in edges and (n2, n1) not in edges:
                                if (n1, n2) in self.graph.edges:
                                    edges.append((n1, n2))
                                elif (n2, n1) in self.graph.edges:
                                    edges.append((n2, n1))
                max_weight = 0
                for e in edges:
                    w = self.graph.edges.get(e)['weight']
                    if w > max_weight:
                        max_weight = w
                        max_edge = (e[0], e[1], w)
                n1, n2, d = max_edge
                sorter1, unit1 = n1.split('_')
                sorter2, unit2 = n2.split('_')
                unit1 = int(unit1)
                unit2 = int(unit2)
                sp1 = self.sorting_list[self.name_list.index(sorter1)].get_unit_spike_train(unit1)
                sp2 = self.sorting_list[self.name_list.index(sorter2)].get_unit_spike_train(unit2)
                lab1, lab2 = compare_spike_trains(sp1, sp2)
                tp_idx1 = np.where(np.array(lab1) == 'TP')[0]
                tp_idx2 = np.where(np.array(lab2) == 'TP')[0]
                assert len(tp_idx1) == len(tp_idx2)
                sp_tp1 = list(np.array(sp1)[tp_idx1])
                sp_tp2 = list(np.array(sp2)[tp_idx2])
                assert np.allclose(sp_tp1, sp_tp2, atol=self.delta_frames)
                self._spiketrains.append(sp_tp1)
        self.added_nodes = added_nodes

    def _do_agreement_matrix(self, minimum_matching=0):
        sorted_name_list = sorted(self.name_list)
        sorting_agr = AgreementSortingExtractor(self, minimum_matching)
        unit_ids = sorting_agr.get_unit_ids()
        agreement_matrix = np.zeros((len(unit_ids), len(sorted_name_list)))

        for u_i, unit in enumerate(unit_ids):
            for sort_name, sorter in enumerate(sorted_name_list):
                if sorter in sorting_agr.get_unit_property(unit, 'sorter_unit_ids').keys():
                    assigned_unit = sorting_agr.get_unit_property(unit, 'sorter_unit_ids')[sorter]
                else:
                    assigned_unit = -1
                if assigned_unit == -1:
                    agreement_matrix[u_i, sort_name] = np.nan
                else:
                    agreement_matrix[u_i, sort_name] = sorting_agr.get_unit_property(unit, 'avg_agreement')
        return agreement_matrix

    def plot_agreement(self, minimum_matching=0):
        import matplotlib.pylab as plt
        sorted_name_list = sorted(self.name_list)
        sorting_agr = AgreementSortingExtractor(self, minimum_matching)
        unit_ids = sorting_agr.get_unit_ids()
        agreement_matrix = self._do_agreement_matrix(minimum_matching)

        fig, ax = plt.subplots()
        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        ax.matshow(agreement_matrix, cmap='Greens')

        # Major ticks
        ax.set_xticks(np.arange(0, len(sorted_name_list)))
        ax.set_yticks(np.arange(0, len(unit_ids)))
        ax.xaxis.tick_bottom()
        # Labels for major ticks
        ax.set_xticklabels(sorted_name_list, fontsize=12)
        ax.set_yticklabels(unit_ids, fontsize=12)

        ax.set_xlabel('Sorters', fontsize=15)
        ax.set_ylabel('Units', fontsize=20)

        return ax


class AgreementSortingExtractor(se.SortingExtractor):
    def __init__(self, multisortingcomparison, min_agreement=0):
        se.SortingExtractor.__init__(self)
        self._msc = multisortingcomparison
        if min_agreement == 0 or min_agreement == 1:
            self._unit_ids = list(self._msc._new_units.keys())
        else:
            self._unit_ids = list(u for u in self._msc._new_units.keys()
                                  if self._msc._new_units[u]['matched_number'] >= min_agreement)

        for unit in self._unit_ids:
            self.set_unit_property(unit_id=unit, property_name='matched_number',
                                   value=self._msc._new_units[unit]['matched_number'])
            self.set_unit_property(unit_id=unit, property_name='avg_agreement',
                                   value=self._msc._new_units[unit]['avg_agreement'])
            self.set_unit_property(unit_id=unit, property_name='sorter_unit_ids',
                                   value=self._msc._new_units[unit]['sorter_unit_ids'])

    def get_unit_ids(self, unit_ids=None):
        if unit_ids is None:
            return self._unit_ids
        else:
            return self._unit_ids[unit_ids]

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        if unit_id not in self.get_unit_ids():
            raise Exception("Unit id is invalid")
        return np.array(self._msc._spiketrains[list(self._msc._new_units.keys()).index(unit_id)])


def compare_multiple_sorters(sorting_list, name_list=None, delta_time=0.4, match_score=0.5, chance_score=0.1,
                             n_jobs=-1, sampling_frequency=None, verbose=False):
    '''
    Compares multiple spike sorter outputs.

    - Pair-wise comparisons are made
    - An agreement graph is built based on the agreement score

    It allows to return a consensus-based sorting extractor with the `get_agreement_sorting()` method.

    Parameters
    ----------
    sorting_list: list
        List of sorting extractor objects to be compared
    name_list: list
        List of spike sorter names. If not given, sorters are named as 'sorter0', 'sorter1', 'sorter2', etc.
    delta_time: float
        Number of ms to consider coincident spikes (default 0.4 ms)
    match_score: float
        Minimum agreement score to match units (default 0.5)
    chance_score: float
        Minimum agreement score to for a possible match (default 0.1)
    n_jobs: int
       Number of cores to use in parallel. Uses all availible if -1
    sampling_frequency: float
        Sampling frequency (used if information is not in the sorting extractors)
    verbose: bool
        if True, output is verbose

    Returns
    -------
    multi_sorting_comparison: MultiSortingComparison
        MultiSortingComparison object with the multiple sorter comparison
    '''
    return MultiSortingComparison(sorting_list=sorting_list, name_list=name_list, delta_time=delta_time,
                                  match_score=match_score, chance_score=chance_score, n_jobs=n_jobs,
                                  sampling_frequency=sampling_frequency, verbose=verbose)
