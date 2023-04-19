import argparse
import torch
from gvp_energy_backbone.models import CPDModel
from gvp_energy_backbone.data import CATHDataset, BatchSampler, ProteinGraphDataset
import tqdm, os
import torch_geometric
from functools import partial   
print = partial(print, flush=True)
from gvp_energy_backbone.get_new_graph_feat import NewProteinGraphDataset

def main(model, device, args):
    print("Loading CATH dataset")
    cath = CATHDataset(path=args.cath_data,splits_path=args.cath_splits)        

    dataloader = lambda x: torch_geometric.data.DataLoader(x, num_workers=args.num_workers,
                            batch_sampler=BatchSampler(x.node_counts, max_nodes=args.max_nodes))
    trainset, valset, testset = map(ProteinGraphDataset, (cath.train, cath.val, cath.test))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_log = os.path.join(args.models_dir, "train.log")

    newProteinGraphDataset = NewProteinGraphDataset(device=device)

    for epoch in range(args.restore_epoch, args.epochs):
        model.train()
        train_loader, val_loader, test_loader = map(dataloader, (trainset, valset, testset))
        avg_loss, avg_loss_denoise, avg_loss_pred_noise = loop(model, train_loader, newProteinGraphDataset, epoch, optimizer=optimizer, device=device)

        path = f"{all_model_dir}/ep_{epoch}.pt"
        torch.save(model.state_dict(), path)
        
        w_lines_base = f"ep={epoch}\tavg_denoise={avg_loss_denoise:.4f}\tavg_pred_noise={avg_loss_pred_noise:.4f}\tavg_loss={avg_loss:.4f}"
        # print(w_lines_base)
        with open(train_log,"a") as f0:
            w_lines = f"{w_lines_base}\n"
            f0.writelines(w_lines)


def loop(model, dataloader, newProteinGraphDataset, epoch, optimizer=None, device=None):
    t = tqdm.tqdm(dataloader)
    sum_loss_denoise, sum_loss_pred_noise,sum_loss = 0,0,0

    t = tqdm.tqdm(dataloader)
    total_count = 0
    
    for batch in t:
        if optimizer: 
            optimizer.zero_grad()

        batch = batch.to(device)

        # print(batch.coords.shape) # N,4,3
        # print(batch.nodes_num) # N

        # transform the lengths to node2graph
        nodes_arr = batch.nodes_num
        node2graph= torch.zeros(batch.coords.shape[0])
        start, end = 0, 0
        for indx,l_node in enumerate(nodes_arr):
            end += l_node
            # print(indx, start,end)
            node2graph[start:end] = indx
            start += l_node
        node2graph = node2graph.to(device=device, dtype=torch.long)

        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)

        loss_denoise, loss_pred_noise = model(batch.coords, node2graph, device, newProteinGraphDataset, h_V, batch.edge_index, h_E, seq=batch.seq)

        # compute loss
        loss_denoise = loss_denoise*0.01
        loss_pred_noise = loss_pred_noise*0.1
        loss = loss_denoise + loss_pred_noise

        backward_loss = loss_denoise + loss_pred_noise
        # backward_loss = loss_value

        if optimizer:
            optimizer.zero_grad()
            backward_loss.backward()
            optimizer.step()

        num_nodes = int(batch.mask.sum())
        total_count += num_nodes

        sum_loss += float(loss) * num_nodes
        avg_loss = sum_loss / total_count

        sum_loss_denoise += float(loss_denoise) * num_nodes
        avg_loss_denoise = sum_loss_denoise/total_count

        sum_loss_pred_noise += float(loss_pred_noise) * num_nodes
        avg_loss_pred_noise = sum_loss_pred_noise/total_count

        t.set_description("epoch={%d}, loss=%.5f,denoise=%.5f,noise=%.5f" % (epoch, float(avg_loss),float(avg_loss_denoise),float(avg_loss_pred_noise)))

        torch.cuda.empty_cache()

        break

    return avg_loss, avg_loss_denoise, avg_loss_pred_noise


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', metavar='PATH', default='./output/',
                        help='directory to save trained models, default=./output/')
    parser.add_argument('--num-workers', metavar='N', type=int, default=4,
                    help='number of threads for loading data, default=4')
    parser.add_argument('--max-nodes', metavar='N', type=int, default=3000,
                        help='max number of nodes per batch, default=3000')
    parser.add_argument('--epochs', metavar='N', type=int, default=100,
                        help='training epochs, default=100')
    parser.add_argument('--cath-data', metavar='PATH', default='/root/StrucSeqProj/data/chain_set.jsonl',
                        help='location of CATH dataset, default=/root/StrucSeqProj/data/chain_set.jsonl')
    parser.add_argument('--cath-splits', metavar='PATH', default='/root/StrucSeqProj/data/chain_set_splits.json',
                        help='location of CATH split file, default=/root/StrucSeqProj/data/chain_set_splits.json')
    parser.add_argument('--ts50', metavar='PATH', default='/root/StrucSeqProj/data/ts50.json',
                        help='location of TS50 dataset, default=/root/StrucSeqProj/data/ts50.json')
    parser.add_argument('--n-samples', metavar='N', default=100,
                        help='number of sequences to sample (if testing recovery), default=100')

    parser.add_argument('--restore_epoch', metavar='N', type=int, default=0,
                        help='the starting epoch for restored model, default=0')
    parser.add_argument('--restore_model', metavar='PATH', default=None,
                        help='path of restored model')
    
    parser.add_argument('--en_num_layers', type=int, default=3, help='number of gvp encoder number')
    parser.add_argument('--de_num_layers', type=int, default=3, help='number of gvp decoder number')

    args = parser.parse_args()

    node_hidden_dim_scalar = 100
    node_hidden_dim_vector = 16
    node_dim = (node_hidden_dim_scalar, node_hidden_dim_vector)
    edge_dim = (32, 1)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    if not os.path.exists(args.models_dir):
        os.makedirs(args.models_dir)
    all_model_dir = os.path.join(args.models_dir, "checkpoint")
    if not os.path.exists(all_model_dir):
        os.makedirs(all_model_dir)
    
    # model = CPDModel((6, 3), node_dim, (32, 1), edge_dim).to(device)
    gvp_model = CPDModel((6, 3), node_dim, (32, 1), edge_dim, en_num_layers=args.en_num_layers, de_num_layers=args.de_num_layers)
    gvp_model = gvp_model.to(device)
    # struc_encoder = StrucEncoder(model, align_dim=512, noise_type='kabsch')

    from struc_encoder import StrucEncoder
    # struc_encoder = StrucEncoder(gvp_model, align_dim=512, node_hidden_dim_scalar=node_hidden_dim_scalar, node_hidden_dim_vector=node_hidden_dim_vector, noise_type='gaussian')
    struc_encoder = StrucEncoder(gvp_model, align_dim=512, node_hidden_dim_scalar=node_hidden_dim_scalar, node_hidden_dim_vector=node_hidden_dim_vector)
    # torch.Size([345, 4, 3])
    # tensor([345])
    # pos= torch.randn(345, 4, 3).to(device=device)

    if args.restore_model:
        struc_encoder.load_state_dict(torch.load(args.restore_model))

    # transform the lengths to node2graph
    # nodes_arr = torch.tensor([3,2,338,1,1]).to(device=device)
    # node2graph= torch.zeros(345).to(device=device)

    # start, end = 0, 0
    # for indx,l_node in enumerate(nodes_arr):
    #     end += l_node
    #     print(indx, start,end)
    #     node2graph[start:end] = indx
    #     start += l_node
    # node2graph = node2graph.to(torch.long)
    # print(node2graph.shape)
    # exit(0)

    # struc_encoder(pos, node2graph)
    # exit(0)

    struc_encoder = struc_encoder.train()
    struc_encoder = struc_encoder.to(device)
    
    main(struc_encoder, device=device, args=args)