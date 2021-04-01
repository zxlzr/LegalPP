import os
import json
import time
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
from argparse import ArgumentParser
from tensorboardX import SummaryWriter

import utils
from utils import logger
from dataset import Dataset
from module import load_graph_encoder, load_relation_decoder


def main(params):
    # Load data
    dataset = Dataset(params['data_path'], params['test']['test_target'])
    num_nodes = dataset.num_nodes
    train_data = dataset.train
    valid_data = dataset.valid
    test_data = dataset.test
    num_relations = dataset.num_relations

    # validation and testing triplets
    valid_data = torch.tensor(valid_data, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.long)
    train_data_tensor = torch.tensor(train_data, dtype=torch.long)

    # GPU settings
    use_cuda = params['use_cuda'] and torch.cuda.is_available()
    cpu_device = torch.device('cpu')
    device1 = torch.device(params['graph_encoder']['device'])
    device2 = torch.device(params['relation_decoder']['device'])

    # Load module
    embed = None
    if params['load_embed']['do']:
        embed = torch.from_numpy(
            np.load(params['load_embed']['embed_path'])['embed'])
        logger.info('Loaded pretrained embedding weight from {}'.format(
            params['load_embed']['embed_path']))
    graph_encoder = load_graph_encoder(params, dataset, embed)
    relation_decoder = load_relation_decoder(params, dataset)

    learning_rate = params['train']['lr']
    weight_decay = params['train']['weight_decay']
    optimizer = Adam(list(graph_encoder.parameters()) + list(relation_decoder.parameters()),
                     lr=learning_rate, weight_decay=weight_decay)

    # If model exists, start training with it
    model_state_file = params['model_path']
    epoch = utils.load_model(
        model_state_file, graph_encoder, relation_decoder)
    if epoch:
        logger.info('Restore model from: {}, '
                    'using best epoch: {}'.format(model_state_file,
                                                  epoch))

    # build test graph
    test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_relations, train_data)
    # test_deg = test_graph.in_degrees(
    #     range(test_graph.number_of_nodes())).float().view(-1, 1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = utils.node_norm_to_edge_norm(
        test_graph, torch.from_numpy(test_norm).view(-1, 1))

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

    # Training
    if params['train']['do']:
        logger.info('Start training...')

        # Use tensorboard to record scalars
        writer = SummaryWriter(params['train']['log_file'])

        epoch = 0
        best_mrr = 0

        while True:
            epoch += 1

            # Prepare model
            for model in [graph_encoder, relation_decoder]:
                model.train()
            optimizer.zero_grad()

            # perform edge neighborhood sampling to generate training graph and data
            batch_size = params['train']['train_batch_size']
            split_size = params['train']['graph_split_size']
            negative_sample = params['train']['negative_sample']
            edge_sampler = params['train']['edge_sampler']
            g, node_id, rel, node_norm, data, labels = \
                utils.generate_sampled_graph_and_labels(
                    train_data, batch_size, split_size,
                    num_relations, adj_list, degrees, negative_sample,
                    edge_sampler)
            logger.info('Done edge sampling')

            # set node / edge feature
            node_id = torch.from_numpy(node_id).view(-1, 1).long()
            rel = torch.from_numpy(rel)
            edge_norm = utils.node_norm_to_edge_norm(
                g, torch.from_numpy(node_norm).view(-1, 1))
            data, labels = torch.from_numpy(data), torch.from_numpy(labels)

            # Forward pass
            t0 = time.time()
            emb_entity = graph_encoder(g, node_id, rel, edge_norm)
            score = relation_decoder(emb_entity, data)

            # Calculate loss on same device
            if labels.device != score.device:
                labels = labels.to(score.device)
            loss = F.binary_cross_entropy_with_logits(score, labels)
            regularization = params['train']['regularization']
            loss += regularization * \
                (torch.pow(emb_entity, 2).mean() +
                 relation_decoder.reglurization())
            t1 = time.time()

            # Record loss
            writer.add_scalar('loss', loss.item(), epoch)

            # clip gradients
            loss.backward()
            for model in [graph_encoder, relation_decoder]:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params['train']['grad_norm'])

            # Optimize
            optimizer.step()
            t2 = time.time()

            forward_time = t1 - t0
            backward_time = t2 - t1
            logger.info('Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | '
                        'Forward {:.4f}s | Backward {:.4f}s'.
                        format(epoch, loss.item(), best_mrr, forward_time, backward_time))

            # validation
            if epoch % params['train']['eval_every'] == 0:
                # perform validation on CPU because full graph is too large
                if use_cuda:
                    graph_encoder.set_device(cpu_device)
                    relation_decoder.set_device(cpu_device)
                logger.info('start eval')
                with torch.no_grad():
                    emb_entity = graph_encoder(
                        test_graph, test_node_id, test_rel, test_norm)
                    logger.info(
                        'graph encoded embedding of each node calculated')
                    res = utils.eval_filtered(emb_entity, relation_decoder,
                                              train_data_tensor, valid_data, test_data,
                                              hits=[1, 3, 10], eval_type='test')
                    mrr = res['all']['mrr']

                    # Record evaluation results
                    for rank_name, eval_result in res.items():
                        for k, v in eval_result.items():
                            writer.add_scalar(
                                '{}_{}'.format(rank_name, k), v, epoch)

                # save best model
                if mrr < best_mrr:
                    if epoch >= params['train']['n_epochs']:
                        break
                else:
                    best_mrr = mrr
                    utils.save_model(model_state_file, graph_encoder,
                                     relation_decoder, epoch)

                # Recover device
                graph_encoder.set_device(device1)
                relation_decoder.set_device(device2)

        logger.info('training done')

    if params['test']['do']:
        logger.info('start testing:')
        # use best model checkpoint
        epoch = utils.load_model(
            model_state_file, graph_encoder, relation_decoder)
        if epoch:
            logger.info('Restore model from: {}, '
                        'using best epoch: {}'.format(model_state_file,
                                                      epoch))

        # perform validation on CPU because full graph is too large
        if use_cuda:
            graph_encoder.set_device(cpu_device)
            relation_decoder.set_device(cpu_device)

        with torch.no_grad():
            emb_entity = graph_encoder(
                test_graph, test_node_id, test_rel, test_norm)
            utils.eval_filtered(emb_entity, relation_decoder,
                                train_data_tensor, valid_data,
                                test_data, hits=[1, 3, 10],
                                eval_type='test', entity_filters=dataset.entity_filters)
        logger.info('testing done')

    if params['export_embed']['do']:
        logger.info('exporting embedding from graph model...')

        # use best model checkpoint
        epoch = utils.load_model(
            model_state_file, graph_encoder, relation_decoder)
        if epoch:
            logger.info('Restore model from: {}, '
                        'using best epoch: {}'.format(model_state_file,
                                                      epoch))

        # Save embedding weights
        save_file = params['export_embed']['embed_path']
        emb_weights = graph_encoder.emb_node.weight.detach().cpu().numpy()
        np.savez_compressed(save_file, embed=emb_weights)
        logger.info('Saved embedding weights to ' + save_file)


if __name__ == '__main__':
    parser = ArgumentParser('Link prediction framework')
    parser.add_argument('-c', '--config', type=str, default='config/config.json',
                        help='path to configuration file containing a dict of parameters')
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        config = json.load(f)
    main(config)
