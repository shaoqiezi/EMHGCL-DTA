def train(model, predictor, device, train_loader, drug_graphs_DataLoader, lr, epoch,
          batch_size, affinity_graph, prot):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    predictor.train()
    LOG_INTERVAL = 10
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, chain(model.parameters(), predictor.parameters())), lr=lr, weight_decay=0)
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        ssl_loss, drug_embedding, target_embedding = model(affinity_graph.to(device), drug_graph_batchs, prot)
        output = predictor(data.to(device), drug_embedding, target_embedding)

        loss = loss_fn(output, data.y.view(-1, 1).float().to(device)) + ssl_loss
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))


def test(model, predictor, device, loader, drug_graphs_DataLoader, affinity_graph, prot):
    model.eval()
    predictor.eval()

    total_preds = torch.Tensor()
    total_labels = torch.Tensor()


    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    with torch.no_grad():
        for data in loader:
            _, drug_embedding, target_embedding = model(affinity_graph.to(device), drug_graph_batchs, prot)
            output = predictor(data.to(device), drug_embedding, target_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def train_predict():
    print("Data preparation in progress for the {} dataset...".format(args.dataset))
    affinity_mat = load_data(args.dataset)
    train_data, test_data, affinity_graph = process_data(affinity_mat, args.dataset)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    drug_graphs_dict = get_drug_molecule_graph(
        json.load(open(f'data/{args.dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug")
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate,
                                                         batch_size=affinity_graph.num_drug)

    targets = json.load(open(f'data/{args.dataset}/targets.txt'), object_pairs_hook=OrderedDict)
    seq = [(key, targets[key]) for key in targets.keys()]

    esmmodel = ESMBlock()
    prot_feature = esmmodel(seq)
    print("Model preparation... ")
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    model = EMHGCLDTA(tau=args.tau,
                    lam=args.lam,
                    ns_dims=[affinity_graph.num_drug + affinity_graph.num_target, 512, 256],
                    d_ms_dims=[78, 78, 78 * 2, 256],
                    t_ms_dims=[54, 54, 54 * 2, 256],
                    embedding_dim=128,
                    dropout_rate=args.edge_dropout_rate)
    predictor = PredictModule()
    model.to(device)
    predictor.to(device)

    print("Start training...")
    for epoch in range(args.epochs):
        train(model, predictor, device, train_loader, drug_graphs_DataLoader, args.lr, epoch+1,
              args.batch_size, affinity_graph, prot_feature)

    print('\npredicting for test data')
    G, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, affinity_graph,
                prot_feature)

    result = model_evaluate(G, P)
    print("result:", result)

if __name__ == '__main__':
    import os
    import argparse
    import torch
    import json
    import warnings
    from collections import OrderedDict
    from torch import nn
    from itertools import chain
    from data_process import load_data, process_data, get_drug_molecule_graph
    from utils import GraphDataset, collate, model_evaluate
    from models import EMHGCLDTA, PredictModule, ESMBlock

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--edge_dropout_rate', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--lam', type=float, default=0.5)
    args, _ = parser.parse_known_args()

    train_predict()




