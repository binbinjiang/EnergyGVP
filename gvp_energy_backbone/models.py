import torch
import torch.nn as nn
from . import GVP, GVPConvLayer, LayerNorm
from .get_gvp_repr import get_node_repr

class CPDModel(torch.nn.Module):
    '''
    GVP-GNN for structure-conditioned autoregressive 
    protein design as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a categorical distribution
    over 20 amino acids at each position in a `torch.Tensor` of 
    shape [n_nodes, 20].
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    The standard forward pass requires sequence information as input
    and should be used for training or evaluating likelihood.
    For sampling or design, use `self.sample`.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :param num_layers: number of GVP-GNN layers in each of the encoder
                       and decoder modules
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 en_num_layers=3, de_num_layers=3, drop_rate=0.1):
    
        super(CPDModel, self).__init__()
        
        self.W_v = nn.Sequential(
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
            LayerNorm(node_h_dim)
        )
        self.W_e = nn.Sequential(
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
            LayerNorm(edge_h_dim)
        )
        
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(en_num_layers))
        
        self.W_s = nn.Embedding(20, 20)
        edge_h_dim = (edge_h_dim[0] + 20, edge_h_dim[1])
      
        self.decoder_layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, 
                             drop_rate=drop_rate, autoregressive=True) 
            for _ in range(de_num_layers))
        
        self.W_out = GVP(node_h_dim, (20, 0), activations=(None, None))

        node_hidden_dim_scalar = 100
        node_hidden_dim_vector = 16
        gvp_out_dim = node_hidden_dim_scalar + (3 *node_hidden_dim_vector)
        self.logit_linear_output = nn.Linear(gvp_out_dim, 20)


    def forward(self, pos, h_V, edge_index, h_E, seq, energy=False):
        '''
        Forward pass to be used at train-time, or evaluating likelihood.
        
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: int `torch.Tensor` of shape [num_nodes]
        '''
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        
        # encoder_embeddings = h_V
        perturbed_gvp_out_scalars, perturbed_gvp_out_vectors  = h_V
        encoder_embeddings = get_node_repr(pos, perturbed_gvp_out_scalars, perturbed_gvp_out_vectors)

        if energy:
            # s, v  = h_V
            # print("encoder_embeddings",s.shape, v.shape)
            return encoder_embeddings

        logits = self.logit_linear_output(encoder_embeddings)

        # h_S = self.W_s(seq)
        # h_S = h_S[edge_index[0]]
        # h_S[edge_index[0] >= edge_index[1]] = 0
        # h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])

        # for layer in self.decoder_layers:
        #     h_V = layer(h_V, edge_index, h_E, autoregressive_x = encoder_embeddings)
        
        # logits = self.W_out(h_V)
        
        return encoder_embeddings, logits
    