import torch
from torch import nn
from module import ModuleWithDevice


class DistMultDecoder(ModuleWithDevice):
    def __init__(self, num_relations, num_dim):
        super(DistMultDecoder, self).__init__()
        self.emb_relation = nn.Parameter(torch.zeros(num_relations, num_dim),
                                         requires_grad=True)
        nn.init.xavier_uniform_(self.emb_relation,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, emb_entity, triples):
        # Make sure data is on right device
        emb_entity, triples = self.assure_device(emb_entity, triples)

        s = emb_entity[triples[:, 0]]
        r = self.emb_relation[triples[:, 1]]
        o = emb_entity[triples[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def inference(self, emb_entity, s, r, o):
        # Calculate scores of triples including corrupted s / o
        emb_s = emb_entity[s]
        emb_o = emb_entity[o]
        emb_r = self.emb_relation[r]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
        return scores

    def reglurization(self):
        return torch.pow(self.emb_relation, 2).mean()


class SimplEDecoder(ModuleWithDevice):
    def __init__(self, num_relations, num_dim):
        super(SimplEDecoder, self).__init__()
        self.emb_relation = nn.Parameter(torch.zeros(num_relations, num_dim),
                                         requires_grad=True)
        self.inv_emb_relation = nn.Parameter(torch.zeros(num_relations, num_dim),
                                             requires_grad=True)
        nn.init.xavier_uniform_(
            self.emb_relation, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(
            self.inv_emb_relation, gain=nn.init.calculate_gain('relu'))

    def forward(self, emb_entity, triples):
        # Make sure data is on right device
        emb_entity, triples = self.assure_device(emb_entity, triples)

        s = emb_entity[triples[:, 0]]
        r = self.emb_relation[triples[:, 1]]
        r_inv = self.inv_emb_relation[triples[:, 1]]
        o = emb_entity[triples[:, 2]]

        return (torch.sum(s * r * o, -1) + torch.sum(s * r_inv * o, -1)) / 2

    def inference(self, emb_entity, s, r, o):
        # Calculate scores of triples including corrupted s / o
        emb_s = emb_entity[s]
        emb_o = emb_entity[o]
        emb_r = self.emb_relation[r]
        emb_r_inv = self.inv_emb_relation[r]
        emb_triplet = emb_s * emb_r * emb_o
        emb_triplet_inv = emb_s * emb_r_inv * emb_o
        scores = (torch.sum(emb_triplet, -1) +
                  torch.sum(emb_triplet_inv, -1)) / 2
        return scores

    def reglurization(self):
        return torch.pow(self.emb_relation, 2).mean() + torch.pow(self.inv_emb_relation, 2).mean()


# class TransEDecoder(ModuleWithDevice):
#     def __init__(self, num_relations, num_dim):
#         super(TransEDecoder, self).__init__()
#         self.emb_relation = nn.Parameter(torch.zeros(num_relations, num_dim),
#                                          requires_grad=True)
#         nn.init.xavier_uniform_(
#             self.emb_relation, gain=nn.init.calculate_gain('relu'))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, emb_entity, triples):
#         # Make sure data is on right device
#         emb_entity, triples = self.assure_device(emb_entity, triples)

#         s = emb_entity[triples[:, 0]]
#         r = self.emb_relation[triples[:, 1]]
#         o = emb_entity[triples[:, 2]]
#         score = 1 - self.sigmoid(s + r - o).mean(dim=1)

#         return score


def load_relation_decoder(params, dataset):
    # Load relation decoder
    relation_decoder_name = params['relation_decoder']['name']
    device = params['relation_decoder']['device']

    n_hidden = params['relation_decoder']['n_hidden']

    if relation_decoder_name == 'distmult':
        relation_decoder = DistMultDecoder(dataset.num_relations, n_hidden)
    # elif relation_decoder_name == 'transe':
    #     relation_decoder = TransEDecoder(dataset.num_relations, n_hidden)
    elif relation_decoder_name == 'simple':
        relation_decoder = SimplEDecoder(dataset.num_relations, n_hidden)
    else:
        raise NotImplementedError()
    relation_decoder.set_device(device)

    return relation_decoder
