import torch
import torch.nn as nn
from torch.nn import functional as F
from dgl.nn.pytorch import RelGraphConv, GATConv
from sklearn import decomposition

from module import ModuleWithDevice


class RelGraphConvolutionEncoder(ModuleWithDevice):
    def __init__(self, num_nodes, num_relations, num_hidden, num_bases,
                 num_hidden_layers=2, dropout=0.0, embed=None):
        super(RelGraphConvolutionEncoder, self).__init__()
        num_bases = None if num_bases < 0 else num_bases

        # Embedding layer
        if embed is not None:
            if embed.shape[1] > num_hidden:
                svd = decomposition.TruncatedSVD(n_components=num_hidden)
                embed = svd.fit_transform(embed)
                embed = torch.tensor(embed, dtype=torch.float)
            self.emb_node = nn.Embedding.from_pretrained(embed)
        else:
            self.emb_node = nn.Embedding(num_nodes, num_hidden)

        # Register layers
        layers = []
        for i in range(num_hidden_layers):
            act = F.relu if i < num_hidden_layers - 1 else None
            layers.append(RelGraphConv(num_hidden, num_hidden, num_relations, "bdd",
                                       num_bases, activation=act, self_loop=True,
                                       dropout=dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, g, h, r, norm):
        # Make sure data is on right device
        g, h, r, norm = self.assure_device(g, h, r, norm)

        h = self.emb_node(h.squeeze())

        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h


class GraphAttentionEncoder(ModuleWithDevice):
    def __init__(self, num_nodes, num_layers, in_dim, num_hidden,
                 heads, feat_drop, attn_drop, negative_slope, residual, embed=None):
        super(GraphAttentionEncoder, self).__init__()

        # Embedding layer
        if embed is not None:
            if embed.shape[1] > num_hidden:
                svd = decomposition.TruncatedSVD(n_components=num_hidden)
                embed = svd.fit_transform(embed)
                embed = torch.tensor(embed, dtype=torch.float)
            self.emb_node = nn.Embedding.from_pretrained(embed)
        else:
            self.emb_node = nn.Embedding(num_nodes, num_hidden)

        # Hidden layers
        layers = []
        for idx in range(num_layers):
            activation = None if idx == num_layers - 1 else F.elu
            if idx == 0:
                layer = GATConv(in_dim, num_hidden, heads[idx], feat_drop, attn_drop,
                                negative_slope, False, activation)
            else:
                layer = GATConv(num_hidden * heads[idx - 1], num_hidden, heads[idx], feat_drop,
                                attn_drop,
                                negative_slope, residual, activation)
            layers.append(layer)
        self.gat_layers = nn.ModuleList(layers)

    def forward(self, g, h, _, __):
        # Make sure data is on right device
        g, h = self.assure_device(g, h)

        h = self.emb_node(h.squeeze())

        for layer in self.gat_layers:
            h = layer(g, h).flatten(1)
        return h


def load_graph_encoder(params, dataset, embed=None):
    # Load graph encoder
    graph_encoder_name = params['graph_encoder']['name']
    n_hidden = params['graph_encoder']['n_hidden']
    device = params['graph_encoder']['device']

    if graph_encoder_name == 'rgcn':
        model_details = params['graph_encoder']['details']['rgcn']
        n_layers = model_details['n_layers']
        n_bases = model_details['n_bases']
        dropout = model_details['dropout']
        # If use concat method, the gcn should provide an embedding
        graph_encoder = RelGraphConvolutionEncoder(dataset.num_nodes, dataset.num_relations * 2,
                                                   n_hidden, n_bases, n_layers, dropout, embed)
    elif graph_encoder_name == 'gat':
        model_details = params['graph_encoder']['details']['gat']
        n_layers = model_details['n_layers']
        negative_slope = model_details['negative_slope']
        residual = model_details['residual']
        attn_drop = model_details['attn_drop']
        in_drop = model_details['in_drop']
        num_heads = model_details['n_heads']
        # Always return 1 head
        heads = [num_heads] * (n_layers - 1) + [1]
        graph_encoder = GraphAttentionEncoder(dataset.num_nodes, n_layers, n_hidden, n_hidden, heads,
                                              in_drop, attn_drop, negative_slope, residual, embed)
    else:
        raise NotImplementedError()

    graph_encoder.set_device(device)

    return graph_encoder
