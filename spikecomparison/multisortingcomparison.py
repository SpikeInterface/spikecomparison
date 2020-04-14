import numpy as np
import spikeextractors as se
from pathlib import Path
import json
import os
from .basecomparison import BaseComparison
from .symmetricsortingcomparison import SymmetricSortingComparison
from .comparisontools import compare_spike_trains

import networkx as nx


class MultiSortingComparison(BaseComparison):
    def __init__(self, sorting_list, name_list=None, delta_time=0.4, sampling_frequency=None,
                 match_score=0.5, chance_score=0.1, n_jobs=-1, spiketrain_mode='union', verbose=False,
                 do_matching=True):
        BaseComparison.__init__(self, sorting_list, name_list=name_list,
                                delta_time=delta_time, sampling_frequency=sampling_frequency,
                                match_score=match_score, chance_score=chance_score,
                                n_jobs=n_jobs, verbose=verbose)
        self._spiketrain_mode = spiketrain_mode
        if do_matching:
            self._do_comparison()
            self._do_graph()
            self._do_agreement()

    def get_sorting_list(self):
        '''
        Returns sorting list

        Returns
        -------
        sorting_list: list
            List of SortingExtractor objects
        '''
        return self.sorting_list

    def get_agreement_sorting(self, minimum_agreement=1, minimum_agreement_only=False):
        '''
        Returns AgreementSortingExtractor with units with a 'minimum_matching' agreement.

        Parameters
        ----------
        minimum_agreement: int
            Minimum number of matches among sorters to include a unit.
        minimum_agreement_only: bool
            If True, only units with agreement == 'minimum_matching' are included.
            If False, units with an agreement >= 'minimum_matching' are included

        Returns
        -------
        agreement_sorting: AgreementSortingExtractor
            The output AgreementSortingExtractor
        '''
        assert minimum_agreement > 0, "'minimum_agreement' should be greater than 0"
        sorting = AgreementSortingExtractor(self, min_agreement=minimum_agreement,
                                            min_agreement_only=minimum_agreement_only)
        sorting.set_sampling_frequency(self.sampling_frequency)
        return sorting

    def compute_subgraphs(self):
        '''
        Computes subgraphs of connected components.

        Returns
        -------
        sg_sorter_names: list
            List of sorter names for each node in the connected component subrgaph
        sg_units: list
            List of unit ids for each node in the connected component subrgaph
        '''
        g = self.graph
        subgraphs = (g.subgraph(c).copy() for c in nx.connected_components(g))
        sg_sorter_names = []
        sg_units = []
        for i, sg in enumerate(subgraphs):
            sorter_names = []
            sorter_units = []
            for node in sg.nodes:
                sorter_names.append(node.split('_')[0])
                sorter_units.append(int(node.split('_')[1]))
            sg_sorter_names.append(sorter_names)
            sg_units.append(sorter_units)
        return sg_sorter_names, sg_units

    def dump(self, save_folder):
        save_folder = Path(save_folder)
        if not save_folder.is_dir():
            os.makedirs(str(save_folder))
        filename = str(save_folder / 'multicomparison.gpickle')
        nx.write_gpickle(self.graph, filename)
        kwargs = {'delta_time': self.delta_time, 'sampling_frequency': self.sampling_frequency,
                  'match_score': self.match_score, 'chance_score': self.chance_score,
                  'n_jobs': self._n_jobs, 'verbose': self._verbose}
        with (save_folder / 'kwargs.json').open('w') as f:
            json.dump(kwargs, f)
        sortings = {}
        for (name, sort) in zip(self.name_list, self.sorting_list):
            if sort.check_if_dumpable():
                sortings[name] = sort.make_serialized_dict()
            else:
                print(f'Skipping {name} because it is not dumpable')
        with (save_folder / 'sortings.json').open('w') as f:
            json.dump(sortings, f)

    @staticmethod
    def load_multicomparison(folder_path):
        folder_path = Path(folder_path)
        with (folder_path / 'kwargs.json').open() as f:
            kwargs = json.load(f)
        with (folder_path / 'sortings.json').open() as f:
            sortings = json.load(f)
        name_list = sortings.keys()
        sorting_list = [se.load_extractor_from_dict(v) for v in sortings.values()]
        mcmp = MultiSortingComparison(sorting_list=sorting_list, name_list=list(name_list), do_matching=False, **kwargs)
        mcmp.graph = nx.read_gpickle(str(folder_path / 'multicomparison.gpickle'))
        mcmp._do_agreement()
        return mcmp

    def _do_comparison(self,):
        # do pairwise matching
        if self._verbose:
            print('Multicomaprison step 1: pairwise comparison')

        self.comparisons = []
        for i in range(len(self.sorting_list)):
            for j in range(i + 1, len(self.sorting_list)):
                if self._verbose:
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

    def _do_graph(self):
        if self._verbose:
            print('Multicomaprison step 2: make graph')

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
        self._remove_duplicate_edges()

    def _do_agreement(self):
        # extract agrrement from graph
        if self._verbose:
            print('Multicomaprison step 3: extract agreement from graph')

        self._new_units = {}
        self._spiketrains = []
        unit_id = 0

        subgraphs = (self.graph.subgraph(c).copy() for c in nx.connected_components(self.graph))
        for i, sg in enumerate(subgraphs):
            edges = list(sg.edges(data=True))
            if len(edges) > 0:
                avg_agr = np.mean([d['weight'] for u, v, d in edges])
            else:
                avg_agr = 0
            sorter_unit_ids = {}
            for node in sg.nodes:
                sorter_name = node.split('_')[0]
                sorter_unit = int(node.split('_')[1])
                sorter_unit_ids[sorter_name] = sorter_unit
            self._new_units[unit_id] = {'avg_agreement': avg_agr, 'sorter_unit_ids': sorter_unit_ids,
                                        'agreement_number': len(sg.nodes)}
            # Append correct spike train
            if len(sorter_unit_ids.keys()) == 1:
                self._spiketrains.append(self.sorting_list[self.name_list.index(
                    list(sorter_unit_ids.keys())[0])].get_unit_spike_train(list(sorter_unit_ids.values())[0]))
            else:
                max_edge = edges[int(np.argmax([d['weight'] for u, v, d in edges]))]
                node1, node2, weight = max_edge
                sorter1, unit1 = node1.split('_')
                sorter2, unit2 = node2.split('_')
                unit1 = int(unit1)
                unit2 = int(unit2)
                sp1 = self.sorting_list[self.name_list.index(sorter1)].get_unit_spike_train(unit1)
                sp2 = self.sorting_list[self.name_list.index(sorter2)].get_unit_spike_train(unit2)

                if self._spiketrain_mode == 'union':
                    lab1, lab2 = compare_spike_trains(sp1, sp2)
                    # add FP to spike train 1 (FP are the only spikes outside the union)
                    fp_idx2 = np.where(np.array(lab2) == 'FP')[0]
                    sp_union = np.sort(np.concatenate((sp1, sp2[fp_idx2])))
                    self._spiketrains.append(list(sp_union))
                elif self._spiketrain_mode == 'intersection':
                    lab1, lab2 = compare_spike_trains(sp1, sp2)
                    # TP are the spikes in the intersection
                    tp_idx1 = np.where(np.array(lab1) == 'TP')[0]
                    sp_tp1 = list(np.array(sp1)[tp_idx1])
                    self._spiketrains.append(sp_tp1)
            unit_id += 1


        # for node in self.graph.nodes():
        #     edges = self.graph.edges(node, data=True)
        #     sorter, unit = (str(node)).split('_')
        #     unit = int(unit)
        #
        #     if len(edges) == 0:
        #         avg_agr = 0
        #         sorting_idxs = {sorter: unit}
        #         self._new_units[unit_id] = {'avg_agreement': avg_agr,
        #                                     'sorter_unit_ids': sorting_idxs}
        #         unit_id += 1
        #         added_nodes.append(str(node))
        #     else:
        #         # Add edges from the second node
        #         all_edges = list(edges)
        #         for edge in edges:
        #             node1, node2, _ = edge
        #             edges_node2 = self.graph.edges(node2, data=True)
        #             if len(edges_node2) > 0:  # useless line if
        #                 for edge_n2 in edges_node2:
        #                     n_node1, n_node2, _ = edge_n2
        #                     # Sort node names to make sure the name is not reversed
        #                     if sorted([n_node1, n_node2]) not in [sorted([u, v]) for u, v, _ in all_edges]:
        #                         all_edges.append(edge_n2)
        #         avg_agr = np.mean([d['weight'] for u, v, d in all_edges])
        #
        #         for edge in all_edges:
        #             node1, node2, data = edge
        #             if node1 not in added_nodes or node2 not in added_nodes:
        #                 sorter1, unit1 = node1.split('_')
        #                 sorter2, unit2 = node2.split('_')
        #                 unit1 = int(unit1)
        #                 unit2 = int(unit2)
        #                 sorting_idxs = {sorter1: unit1, sorter2: unit2}
        #                 if unit_id not in self._new_units.keys():
        #                     self._new_units[unit_id] = {'avg_agreement': avg_agr,
        #                                                 'sorter_unit_ids': sorting_idxs}
        #                 else:
        #                     full_sorting_idxs = self._new_units[unit_id]['sorter_unit_ids']
        #                     for s, u in sorting_idxs.items():
        #                         if s not in full_sorting_idxs:
        #                             full_sorting_idxs[s] = u
        #                     self._new_units[unit_id] = {'avg_agreement': avg_agr,
        #                                                 'sorter_unit_ids': full_sorting_idxs}
        #                 if node not in added_nodes:
        #                     added_nodes.append(str(node))
        #                 if node1 not in added_nodes:
        #                     added_nodes.append(str(node1))
        #                 if node2 not in added_nodes:
        #                     added_nodes.append(str(node2))
        #         unit_id += 1
        #
        # for u, v in self._new_units.items():
        #     # count matched number
        #     matched_num = len(v['sorter_unit_ids'].keys())
        #     v['agreement_number'] = matched_num
        #     self._new_units[u] = v
        #
        #     if len(v['sorter_unit_ids'].keys()) == 1:
        #         self._spiketrains.append(self.sorting_list[self.name_list.index(
        #             list(v['sorter_unit_ids'].keys())[0])].get_unit_spike_train(list(v['sorter_unit_ids'].values())[0]))
        #     else:
        #         nodes = []
        #         edges = []
        #         for sorter, unit in v['sorter_unit_ids'].items():
        #             nodes.append((sorter + '_' + str(unit)))
        #         for node1 in nodes:
        #             for node2 in nodes:
        #                 if node1 != node2:
        #                     if (node1, node2) not in edges and (node2, node1) not in edges:
        #                         if (node1, node2) in self.graph.edges:
        #                             edges.append((node1, node2))
        #                         elif (node2, node1) in self.graph.edges:
        #                             edges.append((node2, node1))
        #         max_weight = 0
        #         for e in edges:
        #             w = self.graph.edges.get(e)['weight']
        #             if w > max_weight:
        #                 max_weight = w
        #                 max_edge = (e[0], e[1], w)
        #         node1, node2, d = max_edge
        #         sorter1, unit1 = node1.split('_')
        #         sorter2, unit2 = node2.split('_')
        #         unit1 = int(unit1)
        #         unit2 = int(unit2)
        #         sp1 = self.sorting_list[self.name_list.index(sorter1)].get_unit_spike_train(unit1)
        #         sp2 = self.sorting_list[self.name_list.index(sorter2)].get_unit_spike_train(unit2)
        #         lab1, lab2 = compare_spike_trains(sp1, sp2)
        #         tp_idx1 = np.where(np.array(lab1) == 'TP')[0]
        #         tp_idx2 = np.where(np.array(lab2) == 'TP')[0]
        #         assert len(tp_idx1) == len(tp_idx2)
        #         sp_tp1 = list(np.array(sp1)[tp_idx1])
        #         sp_tp2 = list(np.array(sp2)[tp_idx2])
        #         assert np.allclose(sp_tp1, sp_tp2, atol=self.delta_frames)
        #         self._spiketrains.append(sp_tp1)

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

    def _remove_duplicate_edges(self):
        g = self.graph
        subgraphs = (g.subgraph(c).copy() for c in nx.connected_components(g))
        removed_nodes = 0
        for i, sg in enumerate(subgraphs):
            sorter_names = []
            for node in sg.nodes:
                sorter_names.append(node.split('_')[0])
            sorters, counts = np.unique(sorter_names, return_counts=True)

            if np.any(counts > 1):
                for sorter in sorters[counts > 1]:
                    nodes_duplicate = [n for n in sg.nodes if sorter in n]
                    # get edges
                    edges_duplicates = []
                    weights_duplicates = []
                    for n in nodes_duplicate:
                        edges = sg.edges(n, data=True)
                        for e in edges:
                            edges_duplicates.append(e)
                            weights_duplicates.append(e[2]['weight'])
                    # remove edges
                    edges_to_remove = len(nodes_duplicate) - 1
                    remove_idxs = np.argsort(weights_duplicates)[:edges_to_remove]
                    for idx in remove_idxs:
                        if self._verbose:
                            print('Removed edge', edges_duplicates[idx])
                        self.graph.remove_edge(edges_duplicates[idx][0], edges_duplicates[idx][1])
                        sg.remove_edge(edges_duplicates[idx][0], edges_duplicates[idx][1])
                        if edges_duplicates[idx][0] in nodes_duplicate:
                            sg.remove_node(edges_duplicates[idx][0])
                        else:
                            sg.remove_node(edges_duplicates[idx][1])
                        removed_nodes += 1
        if self._verbose:
            print(f'Removed {removed_nodes} duplicate nodes')


class AgreementSortingExtractor(se.SortingExtractor):
    def __init__(self, multisortingcomparison, min_agreement=1, min_agreement_only=False):
        se.SortingExtractor.__init__(self)
        self._msc = multisortingcomparison

        if min_agreement_only:
            self._unit_ids = list(u for u in self._msc._new_units.keys()
                                  if self._msc._new_units[u]['agreement_number'] == min_agreement)
        else:
            self._unit_ids = list(u for u in self._msc._new_units.keys()
                                  if self._msc._new_units[u]['agreement_number'] >= min_agreement)

        for unit in self._unit_ids:
            self.set_unit_property(unit_id=unit, property_name='agreement_number',
                                   value=self._msc._new_units[unit]['agreement_number'])
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
                             n_jobs=-1, spiketrain_mode='union', sampling_frequency=None, verbose=False):
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
    spiketrain_mode: str
        Mode to extract agreement spike trains:
            - 'union': spike trains are the union between the spike trains of the best matching two sorters
            - 'intersection': spike trains are the intersection between the spike trains of the
               best matching two sorters
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
                                  spiketrain_mode=spiketrain_mode, sampling_frequency=sampling_frequency,
                                  verbose=verbose)

