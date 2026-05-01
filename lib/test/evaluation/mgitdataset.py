import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import json


class MGITDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.mgit_path
        self.sequence_list = self._get_sequence_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = None

        anno_path = '{}/attribute/groundtruth/{}.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        occlusion_label_path = None

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = None

        nlp_path = '{}/attribute/description/{}.json'.format(self.base_path, sequence_name)
        with open(nlp_path, 'r') as file:
            data = json.load(file)
        nlp_rect = data['action']['action_1']['description']

        frames_path = '{}/data/test/{}/frame_{}'.format(self.base_path, sequence_name, sequence_name)
        frames_list = [f for f in os.listdir(frames_path)]
        frames_list = sorted(frames_list)
        frames_list = ['{}/{}'.format(frames_path, frame_i) for frame_i in frames_list]
        return Sequence(sequence_name, frames_list, 'mgit', ground_truth_rect.reshape(-1, 4),
                        object_class=None, target_visible=None, language_query=nlp_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = sequence_list=['001', '006', '007', '012', '022', '038', '045', '061', '074', '079', '087', '089', '093', '107', '111', '114', '117',
 '148', '181', '230', '255', '275', '277', '286', '311', '366', '418', '449', '469', '498']
        return sequence_list


# ['001', '006', '007', '012', '022', '038', '045', '061', '074', '079', '087', '089', '093', '107', '111', '114', '117',
#  '148', '181', '230', '255', '275', '277', '286', '311', '366', '418', '449', '469', '498']
# ['001','007','012','022','038','061','074','087','089','107',
#                                        '111','114','117','230','255','275','277','311','418',
#                                        '449','469','498']
# ['006', '045', '079', '093', '148', '181', '286', '366']

