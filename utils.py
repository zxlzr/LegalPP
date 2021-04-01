"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""

import os
import numpy as np
import torch
import dgl
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('Framework')
logger.setLevel(logging.DEBUG)


#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i, triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees


def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    """Sample edges by neighborhool expansion.

    This guarantees that the sampled edges form a connected graph, which
    may help deeper GNNs that require information from more than one hop.
    """
    edges = np.zeros((sample_size,), dtype=np.int32)

    # initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = weights / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges


def sample_edge_uniform(_, __, n_triplets, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)


def generate_sampled_graph_and_labels(triplets, sample_size, split_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate, sampler="uniform"):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    if sampler == "uniform":
        edges = sample_edge_uniform(
            adj_list, degrees, len(triplets), sample_size)
    elif sampler == "neighbor":
        edges = sample_edge_neighborhood(
            adj_list, degrees, len(triplets), sample_size)
    else:
        raise ValueError(
            "Sampler type must be either 'uniform' or 'neighbor'.")

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
    logger.info("# sampled nodes: {}".format(len(uniq_v)))
    logger.info("# sampled edges: {}".format(len(src) * 2))
    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                             (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels


def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm


def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    return g.edata['norm']


def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    logger.info("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    return g, rel.astype('int64'), norm  # .astype('int64') # This is buggy


def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    logger.info("Test graph:")
    return build_graph_from_triplets(num_nodes, num_rels, (src, rel, dst))


def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels


#######################################################################
#
# Utility functions for evaluations (filtered)
#
#######################################################################

def filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities,
             entity_filter=None):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_o = []
    # Do not filter out the test triplet, since we want to predict on it
    if (target_s, target_r, target_o) in triplets_to_filter:
        triplets_to_filter.remove((target_s, target_r, target_o))
    # Do not consider an object if it is part of a triplet to filter
    for o in range(num_entities):
        if (target_s, target_r, o) not in triplets_to_filter:
            filtered_o.append(o)
    # Do further filtering
    if entity_filter:
        filtered_o = list(filter(lambda idx: idx in entity_filter, filtered_o))
    return torch.LongTensor(filtered_o)


def filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities,
             entity_filter=None):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_s = []
    # Do not filter out the test triplet, since we want to predict on it
    if (target_s, target_r, target_o) in triplets_to_filter:
        triplets_to_filter.remove((target_s, target_r, target_o))
    # Do not consider a subject if it is part of a triplet to filter
    for s in range(num_entities):
        if (s, target_r, target_o) not in triplets_to_filter:
            filtered_s.append(s)
    # Do further filtering
    if entity_filter:
        filtered_s = list(filter(lambda idx: idx in entity_filter, filtered_s))
    return torch.LongTensor(filtered_s)


def perturb_o_and_get_filtered_rank(embedding, decoder, target_triples,
                                    triplets_to_filter, entity_filter=None):
    """ Perturb object in the triplets
    """
    num_entities = embedding.shape[0]
    ranks = []
    s, r, o = target_triples[:, 0], target_triples[:, 1], target_triples[:, 2]
    for idx in tqdm(range(len(target_triples)), desc='perturbing object...'):
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_o = filter_o(triplets_to_filter, target_s,
                              target_r, target_o, num_entities,
                              entity_filter)
        target_o_idx = int((filtered_o == target_o).nonzero())
        scores = decoder.inference(embedding, target_s, target_r, filtered_o)
        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_o_idx).nonzero())
        ranks.append(rank)
    ranks = torch.LongTensor(ranks) + 1
    return ranks


def perturb_s_and_get_filtered_rank(embedding, decoder, target_triples,
                                    triplets_to_filter, entity_filter=None):
    """ Perturb subject in the triplets
    """
    num_entities = embedding.shape[0]
    ranks = []
    s, r, o = target_triples[:, 0], target_triples[:, 1], target_triples[:, 2]
    for idx in tqdm(range(len(target_triples)), desc='perturbing subject...'):
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_s = filter_s(triplets_to_filter, target_s,
                              target_r, target_o, num_entities,
                              entity_filter)
        target_s_idx = int((filtered_s == target_s).nonzero())
        scores = decoder.inference(embedding, filtered_s, target_r, target_o)
        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_s_idx).nonzero())
        ranks.append(rank)
    ranks = torch.LongTensor(ranks) + 1
    return ranks


def eval_filtered(embedding, decoder, train_triplets, valid_triplets, test_triplets,
                  hits=[], eval_type="valid", entity_filters=(None, None)):
    target_triples = None
    if eval_type == 'test':
        target_triples = test_triplets
    elif eval_type == 'valid':
        target_triples = valid_triplets

    triplets_to_filter = torch.cat(
        [train_triplets, valid_triplets, test_triplets]).tolist()
    triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}

    logger.info('Perturbing subject...')
    ranks_s = perturb_s_and_get_filtered_rank(embedding, decoder, target_triples,
                                              triplets_to_filter,
                                              entity_filters[0])
    logger.info('Perturbing object...')
    ranks_o = perturb_o_and_get_filtered_rank(embedding, decoder, target_triples,
                                              triplets_to_filter,
                                              entity_filters[1])

    ranks = torch.cat([ranks_s, ranks_o])
    result = {'head': {}, 'tail': {}, 'all': {}}

    for rank_name, rank_data in {'head': ranks_s, 'tail': ranks_o, 'all': ranks}.items():
        logger.info('Eval results for ' + rank_name)
        mr = torch.mean(rank_data.float()).item()
        mrr = torch.mean(1.0 / rank_data.float()).item()
        logger.info("MR: {:.6f}".format(mr))
        logger.info("MRR: {:.6f}".format(mrr))

        result[rank_name] = {'mr': mr, 'mrr': mrr}

        for hit in hits:
            avg_count = torch.mean((rank_data <= hit).float()).item()
            logger.info(
                "Hits @ {}: {:.6f}".format(hit, avg_count))
            result[rank_name]['hit_{}'.format(hit)] = avg_count

    return result


#######################################################################
#
# Model util function
#
#######################################################################

def load_model(model_state_file, graph_encoder, relation_decoder):
    if os.path.exists(model_state_file):
        try:
            model_dict = torch.load(model_state_file)
            graph_encoder.load_state_dict(model_dict['graph'])
            relation_decoder.load_state_dict(model_dict['relation'])
            epoch = model_dict['epoch']
            return epoch
        except Exception as e:
            logger.warning('Fails to load model because of ' + str(e))
            return 0
    else:
        return 0


def save_model(model_state_file, graph_encoder, relation_decoder, epoch):
    model_folder = model_state_file[:model_state_file.rfind('/')]
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save({
        'graph': graph_encoder.state_dict(),
        'relation': relation_decoder.state_dict(),
        'epoch': epoch
    }, model_state_file)
