from torch_scatter import scatter_add,  scatter_mean
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
import math


loss_func = {
        "L1" : nn.L1Loss(reduction='none'),
        "L2" : nn.MSELoss(reduction='none'),
        "Cosine" : nn.CosineSimilarity(dim=-1, eps=1e-08),
        "CrossEntropy" : nn.CrossEntropyLoss(reduction='none')
    }


class StrucEncoder(nn.Module):
    def __init__(
        self,
        struc_encoder,
        align_dim: int = 512,
        node_hidden_dim_scalar = 100,
        node_hidden_dim_vector = 16,
        noise_type = 'riemann',
        pred_mode ='energy',
    ):
        super().__init__()
        self.struc_encoder = struc_encoder

        sigma_begin = 10
        sigma_end = 0.01
        num_noise_level = 50
        self.noise_type = noise_type
        self.pred_mode = pred_mode

        sigmas = torch.tensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_noise_level)), dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False) # (num_noise_level)

        self.noise_pred = nn.Sequential(nn.Linear(align_dim * 2, align_dim),
                                       nn.SiLU(),
                                       nn.Linear(align_dim, num_noise_level))

        gvp_out_dim = node_hidden_dim_scalar + (3 *node_hidden_dim_vector)
        self.embed_gvp_output = nn.Linear(gvp_out_dim, align_dim)
        self.embed_gvp_output2 = nn.Linear(3 *node_hidden_dim_vector, align_dim)

        # for energy head
        self.k_layer_1 = nn.Linear(align_dim, align_dim)
        self.v_layer_1 = nn.Linear(align_dim, align_dim)
        self.q_layer_1 = nn.Linear(align_dim, align_dim)

        self.energy_head_softmax = nn.Softmax(dim=-1)

        self.graph_dec = nn.Sequential(nn.Linear(align_dim, align_dim),
                                       nn.SiLU(),
                                       nn.Linear(align_dim, 1))

    @torch.no_grad()
    def get_force_target(self, perturbed_pos, pos, node2graph):
        # s = - (pos_p @ (pos_p.T @ pos_p) - pos @ (pos.T @ pos_p)) / (torch.norm(pos_p.T @ pos_p) + torch.norm(pos.T @ pos_p))
        if self.noise_type == 'riemann':
            N, v = pos.shape[-2], pos.shape[-1]
            center = scatter_mean(pos, node2graph, dim = -3) # num_graph *N* 3
            perturbed_center = scatter_mean(perturbed_pos, node2graph, dim = -3) # num_graph * N*3
            pos_c = pos - center[node2graph] # B * N*3
            perturbed_pos_c = perturbed_pos - perturbed_center[node2graph] # B * N*3
            perturbed_pos_c_left = perturbed_pos_c.repeat_interleave(v,dim=-1) # B * N*9
            perturbed_pos_c_right = perturbed_pos_c.repeat([1,1,v])# B * N*9
            pos_c_left = pos_c.repeat_interleave(v,dim=-1) # B * N*9
            ptp = scatter_add(perturbed_pos_c_left * perturbed_pos_c_right, node2graph, dim = -3).reshape(-1,N,v,v) # num_graph *N* 3 * 3     
            otp = scatter_add(pos_c_left * perturbed_pos_c_right, node2graph, dim = -3).reshape(-1,N,v,v) # num_graph *N* 3 * 3     
            # print(ptp.shape, otp.shape)
            # print(pos_c_left.shape, perturbed_pos_c_right.shape)
            ptp = ptp[node2graph] # B *N* 3 * 3  
            otp = otp[node2graph] # B *N* 3 * 3  
            # print(ptp.shape, otp.shape)
            tar_force = - 2 * (perturbed_pos_c.unsqueeze(2) @ ptp - pos_c.unsqueeze(2) @ otp).squeeze(2) / (torch.norm(ptp,dim=(2,3)) + torch.norm(otp,dim=(2,3))).unsqueeze(-1).repeat([1,1,v])
            # print(tar_force.shape)
            return tar_force
        else:
            return pos - perturbed_pos

    @torch.no_grad()
    def get_force_target_raw(self, perturbed_pos, pos, node2graph):
        # s = - (pos_p @ (pos_p.T @ pos_p) - pos @ (pos.T @ pos_p)) / (torch.norm(pos_p.T @ pos_p) + torch.norm(pos.T @ pos_p))
        if self.noise_type == 'riemann':
            v = pos.shape[-1]
            center = scatter_mean(pos, node2graph, dim = -2) # num_graph* 3
            perturbed_center = scatter_mean(perturbed_pos, node2graph, dim = -2) # num_graph*3
            pos_c = pos - center[node2graph] # B*3
            perturbed_pos_c = perturbed_pos - perturbed_center[node2graph] # B*3
            perturbed_pos_c_left = perturbed_pos_c.repeat_interleave(v,dim=-1) # B*9
            perturbed_pos_c_right = perturbed_pos_c.repeat([1,v])# B*9
            pos_c_left = pos_c.repeat_interleave(v,dim=-1) # B*9
            ptp = scatter_add(perturbed_pos_c_left * perturbed_pos_c_right, node2graph, dim=-2).reshape(-1,v,v) # num_graph* 3 * 3     
            otp = scatter_add(pos_c_left * perturbed_pos_c_right, node2graph, dim=-2).reshape(-1,v,v) # num_graph* 3 * 3     
            # print(ptp.shape, otp.shape)
            # print(pos_c_left.shape, perturbed_pos_c_right.shape)
            ptp = ptp[node2graph] # B * 3 * 3  
            otp = otp[node2graph] # B * 3 * 3  
            # print(ptp.shape, otp.shape)
            tar_force = - 2 * (perturbed_pos_c.unsqueeze(1) @ ptp - pos_c.unsqueeze(1) @ otp).squeeze(1) / (torch.norm(ptp,dim=(1,2)) + torch.norm(otp,dim=(1,2))).unsqueeze(-1).repeat([1,v])
            # print(tar_force.shape)
            return tar_force
        else:
            return pos - perturbed_pos
    
    @torch.no_grad()
    def fit_pos(self, perturbed_pos, pos, node2graph):
        N, v = pos.shape[-2], pos.shape[-1]
        center = scatter_mean(pos, node2graph, dim = -3) # num_graph *N* 3
        perturbed_center = scatter_mean(perturbed_pos, node2graph, dim = -3) # num_graph *N* 3
        pos_c = pos - center[node2graph] # B * N*3
        perturbed_pos_c = perturbed_pos - perturbed_center[node2graph] # B * N*3
        pos_c = pos_c.repeat([1,1,v])
        perturbed_pos_c = perturbed_pos_c.repeat_interleave(v,dim=-1)
        H = scatter_add(pos_c * perturbed_pos_c, node2graph, dim = -3).reshape(-1,v,v) # (num_graph*N)* 3 * 3
        U, S, V = torch.svd(H)

        # print(U.shape, H.shape, V.shape)
        # Rotation matrix
        R = V @ U.transpose(2,1)
        # print(perturbed_center.shape)
        # print((perturbed_center.reshape(-1,v).unsqueeze(1) @ R.transpose(2,1)).squeeze(1).reshape(-1,N,v).shape)
        # print(center.shape)
        t = center - (perturbed_center.reshape(-1,v).unsqueeze(1) @ R.transpose(2,1)).squeeze(1).reshape(-1,N,v)
        R = R[node2graph]
        t = t[node2graph]
        p_aligned = (perturbed_pos.unsqueeze(1) @ R.transpose(2,1)).squeeze(1) + t
        return p_aligned

    @torch.no_grad()
    def perturb(self, pos, node2graph, used_sigmas, steps=1):
        if self.noise_type == 'riemann':
            pos_p = pos
            for t in range(1, steps + 1):
                alpha = 1 / (2 ** t)
                s = self.get_force_target(pos_p, pos, node2graph)
                pos_p = pos_p + alpha * s + torch.randn_like(pos) * math.sqrt(2 * alpha) * used_sigmas
            return pos_p
        elif self.noise_type == 'kabsch':
            pos_p = pos + torch.randn_like(pos) * used_sigmas
            pos_p = self.fit_pos(pos_p, pos, node2graph)
            return pos_p
        elif self.noise_type == 'gaussian':
            pos_p = pos + torch.randn_like(pos) * used_sigmas
            return pos_p

    @torch.no_grad()
    def perturb_raw(self, pos, node2graph, used_sigmas, steps=1):
        if self.noise_type == 'riemann':
            pos_p = pos
            for t in range(1, steps + 1):
                alpha = 1 / (2 ** t)
                s = self.get_force_target_raw(pos_p, pos, node2graph)
                pos_p = pos_p + alpha * s + torch.randn_like(pos) * math.sqrt(2 * alpha) * used_sigmas
            return pos_p
        elif self.noise_type == 'kabsch':
            pos_p = pos + torch.randn_like(pos) * used_sigmas
            pos_p = self.fit_pos(pos_p, pos, node2graph)
            return pos_p
        elif self.noise_type == 'gaussian':
            pos_p = pos + torch.randn_like(pos) * used_sigmas
            return pos_p

    def energy_head_new(self, struc_repr, node2graph):
        # print(struc_repr.shape)
        struc_repr = scatter_add(struc_repr, node2graph, dim=-2) #B*512 => num_graph*512

        # k = self.k_layer_1(struc_repr) # num_graph*512
        # v = self.v_layer_1(struc_repr) # num_graph*512
        # q = self.q_layer_1(struc_repr) # num_graph*512
        # context = torch.stack((k,v,q), dim=1)#num_graph*3*512 
        # print(context.shape)

        # output = self.graph_dec(context)# num_graph*3*1

        output = self.graph_dec(struc_repr)# num_graph*1

        return output.squeeze(-1)

    def Coords2Cb_torch(self, a, b, c, L=1.522, A=1.927, D=-2.143, device=None):
        """
        a: N; b: Ca; C: O
        input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
        output: 4th coord
        """
        L, A, D = torch.tensor(L), torch.tensor(A), torch.tensor(D)
        L, A, D = L.to(device), A.to(device), D.to(device)
        def normalize(x):
            return x / torch.linalg.norm(x, ord=2, axis=-1, keepdims=True)

        bc = normalize(b - c)
        n = normalize(torch.cross(b - a, bc))
        m = [bc, torch.cross(n, bc), n]
        d = [L * torch.cos(A), L * torch.sin(A) * torch.cos(D), -L * torch.sin(A) * torch.sin(D)]
        return c + sum([m * d for m, d in zip(m, d)])

    def forward(
        self,
        pos, node2graph,device,
        newProteinGraphDataset,
        h_V, edge_index, h_E, seq=None,
    ):
        B,N,v = pos.shape #  num_nodes,3,3

        pos_flat = pos.reshape(B,N*v)
        # mask = torch.isfinite(pos.sum(dim=(1,2)))
        mask = torch.isfinite(pos_flat.sum(dim=(1)))
        # pos[~mask] = np.inf
        pos_flat[~mask] = 0

        noise_level = torch.randint(0, self.sigmas.size(0), (B,)).to(device) # (B)
        used_sigmas = self.sigmas[noise_level] # (B)
        used_sigmas_raw = used_sigmas.unsqueeze(-1) #  B*1
        used_sigmas = used_sigmas.unsqueeze(-1).unsqueeze(-1) #  B*1*1
        
        perturbed_pos = self.perturb_raw(pos_flat, node2graph, used_sigmas_raw, steps=1) # B*N*3
        target = self.get_force_target_raw(perturbed_pos, pos_flat, node2graph) / used_sigmas_raw

        input_pos = perturbed_pos.clone()
        input_pos.requires_grad_(True)
        input_pos = input_pos.reshape(B,N, v)

        perturbed_h_V, perturbed_edge_index, perturbed_h_E = newProteinGraphDataset.new_featurize_as_graph(input_pos)

        perturbed_node_repr = self.struc_encoder(input_pos, perturbed_h_V, perturbed_edge_index, perturbed_h_E, seq=seq.clone(), energy=True)
        perturbed_struc_repr = self.embed_gvp_output(perturbed_node_repr)
        
        energy = self.energy_head_new(perturbed_struc_repr, node2graph) # B*d=> num_graph*N
        # print(energy.shape)
        # energy = energy.unsqueeze(1).repeat(1,L,1) # B*N=> B*L*N

        if self.pred_mode =='energy':
            grad_outputs = [torch.ones_like(energy)]
            dy = grad(
                    [energy],
                    [input_pos],
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                )[0]
            pred_noise = (-dy).view(-1,N*v)
            # print(perturbed_struc_repr.shape, energy.shape, dy.shape, pred_noise.shape, target.shape)
            # torch.Size([4786, 512]) torch.Size([19]) torch.Size([4786, 4, 3]) torch.Size([19144, 3]) torch.Size([4786, 3])

        loss_denoise = loss_func['L2'](pred_noise, target)
        loss_denoise = torch.sum(loss_denoise, dim = -1)

        # original input
        node_repr = self.struc_encoder(pos, h_V, edge_index, h_E, seq=seq, energy=True)
        struc_repr = self.embed_gvp_output(node_repr)

        graph_rep = torch.cat([struc_repr, perturbed_struc_repr], dim=-1)
        pred_scale = self.noise_pred(graph_rep)
        # print(pred_scale.shape, noise_level.shape)
        loss_pred_noise = loss_func['CrossEntropy'](pred_scale, noise_level)
        pred_scale_ = pred_scale.argmax(dim=-1)
        # print(loss_denoise.mean())
        # print(loss_pred_noise.mean())

        return loss_denoise.mean(), loss_pred_noise.mean()