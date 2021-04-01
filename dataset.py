import os
import numpy as np
from utils import logger


def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line


def _read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in _read_triplets(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l


def get_idx2text(entity_dict, entity2text_file):
    idx2text = {}
    pairs = _read_triplets(entity2text_file)
    for entity, text in pairs:
        idx2text[entity_dict[entity]] = text
    return idx2text


class Dataset(object):
    def __init__(self, data_path, target_only):
        self.dir = data_path
        entity_path = os.path.join(self.dir, 'entities.txt')
        relation_path = os.path.join(self.dir, 'relations.txt')
        entity2text_path = os.path.join(self.dir, 'entity2text.txt')

        # Load entity dict & relation dict
        entity_dict, relation_dict = {}, {}
        with open(entity_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                entity_id = line.strip()
                if entity_id not in entity_dict:
                    entity_dict[entity_id] = len(entity_dict)
        with open(relation_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                relation_dict[line.strip()] = idx

        # Load index to text dict
        self.idx2text = get_idx2text(entity_dict, entity2text_path)

        # Load relation data
        train_path = os.path.join(self.dir, 'train.tsv')
        valid_path = os.path.join(self.dir, 'dev.tsv')
        if target_only:
            test_path = os.path.join(self.dir, 'target', 'test.tsv')
        else:
            test_path = os.path.join(self.dir, 'test.tsv')
        self.train = np.asarray(_read_triplets_as_list(
            train_path, entity_dict, relation_dict))
        self.valid = np.asarray(_read_triplets_as_list(
            valid_path, entity_dict, relation_dict))
        self.test = np.asarray(_read_triplets_as_list(
            test_path, entity_dict, relation_dict))
        self.num_nodes = len(entity_dict)
        logger.info("# entities: {}".format(self.num_nodes))
        self.num_relations = len(relation_dict)
        logger.info("# relations: {}".format(self.num_relations))
        logger.info("# edges: {}".format(len(self.train)))

        # Filter target entities for corrupting head / tail
        self.entity_filters = [None, None]
        if target_only:
            # Only used when corrupting tail entity
            filtered_entity_path = os.path.join(
                self.dir, 'target', 'filtered_entities.txt')
            with open(filtered_entity_path, 'r') as f:
                filtered = f.read().split('\n')
            self.entity_filters[1] = {entity_dict[ent]
                                      for ent in filtered if ent in entity_dict}
