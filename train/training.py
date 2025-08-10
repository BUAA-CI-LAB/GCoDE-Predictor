import argparse
import sys
from util import IOStream, cal_loss, mape, rmse
import torch
import torch.optim as optim
sys.path.append("..")
import os.path as osp
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append('../../../')

from torch_geometric.loader import DataLoader
import os
# 只训练gin模型
from tqdm import tqdm

from model.predictor import GIN0

def get_data(data_name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../','dataset', data_name,'data.pt')
    dataset = torch.load(path)
    num_samples = len(dataset)
    print(f'Number of samples: {num_samples}')
    return dataset
def _init_():
    print("enter init!")
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/' + args.exp_name):
        os.makedirs('outputs/' + args.exp_name)
    if not os.path.exists('outputs/' + args.exp_name + '/' + 'models'):
        os.makedirs('outputs/' + args.exp_name + '/' + 'models')




def train(args, io, train_loader,test_loader,model,device):
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    print("Use AdamW")
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=30, threshold=0.01,
                                                     verbose=True)
    if args.loss == "mape":
        criterion = cal_loss
    else:
        criterion = rmse

    best_loss = 1000
    best_model = ''
    for epoch in tqdm(range(args.epochs)):
        ####################
        # Train
        ####################
        train_loss = 0
        mape_metric = 0.0
        model.train()
        count = 0.0
        for data in train_loader:
            data = data.to(device)
            batch_size = len(data.y)
            opt.zero_grad()
            outs = model(data)
            tag = torch.reshape(data.y, (outs.shape[0], -1))[:, args.device].reshape(outs.shape)
            loss = criterion(outs, tag)
            loss.backward()
            opt.step()
            train_loss += loss.item() * batch_size
            mape_metric += mape(outs.cpu().squeeze(), tag.cpu().squeeze()) * batch_size
            count += batch_size

        outstr = 'Train %d, loss: %.6f, mape %d: %.6f%%' % (
            epoch, train_loss * 1.0 / count, args.device, mape_metric / count)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        mape_metric = 0.0
        count = 0.0
        model.eval()
        for data in test_loader:
            data = data.to(device)
            batch_size = len(data.y)
            outs = model(data)
            tag = torch.reshape(data.y, (outs.shape[0], -1))[:, args.device].reshape(outs.shape)
            loss = criterion(outs, tag)
            count += batch_size
            test_loss += loss.item() * batch_size
            mape_metric += mape(outs.cpu().squeeze(), tag.cpu().squeeze()) * batch_size

        if test_loss * 1.0 / count < best_loss:
            best_loss = test_loss * 1.0 / count
            torch.save(model.state_dict(), f'outputs/{args.exp_name}/models/model.t7')
            best_model=model.state_dict()
        outstr = 'Test %d, loss: %.6f, mape %d: %.6f%%, best loss: %.6f' % (
            epoch, test_loss * 1.0 / count, args.device, mape_metric / count, best_loss)
        io.cprint(outstr)

        if epoch > 20:
            scheduler.step(test_loss)

    return best_model, best_loss


def z_score_normalization(data_list):
    """
    对提供的样本列表进行z-score标准化。

    参数:
    - data_list (list of torch_geometric.data.Data): 数据列表。

    返回:
    - data_list (list of torch_geometric.data.Data): 标准化后的数据列表。
    """

    # 合并所有样本的延迟特征值（最后一个维度）
    all_values = [data.x[:, -1] for data in data_list]  # 获取每个样本中的延迟特征
    all_values = torch.cat(all_values, dim=0)

    # 计算均值和标准差
    mean = all_values.mean()
    std = all_values.std()
    io.cprint(f"mean: {mean}")
    io.cprint(f"std: {std}")
    # 标准化每个样本的延迟特征
    for data in data_list:
        data.x[:, -1] = (data.x[:, -1] - mean) / (std + 1e-7)

    return data_list,mean,std



def all_connect_global(data_list):
    processed_list = []

    for data in data_list:
        # 获取所有指向0的边
        to_zero_edges = torch.nonzero(data.edge_index[1] == 0).squeeze()

        # 如果没有指向0的边，则直接添加原数据到结果列表
        if to_zero_edges.numel() == 0:
            processed_list.append(data)
            continue

        new_edges = data.edge_index[:, to_zero_edges]

        # 翻转这些边的方向（即0指向其他节点），但排除0到0的边
        new_edges = torch.flip(new_edges, [0])
        new_edges = new_edges[:, ~(new_edges[1] == 0)]

        # 将新的边添加到原始的edge_index中，并确保没有重复的边
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1).unique(dim=1)
        processed_list.append(data)

    return processed_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN Predictor')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset_name', type=str, default="mn_tx2_1060_40",
                        help='dataset')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--loss', type=str, default="mape",
                        help='model_path')
    parser.add_argument('--device', type=int, default=0,
                        help='device id')
    parser.add_argument('--lr', type=float, default=0.0008,
                        help='device id')
    parser.add_argument('--agg', type=str, default="mean")
    parser.add_argument('--pool', type=str, default="sum")
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--outdim', type=int, default=10)
    parser.add_argument('--hid1', type=int, default=1024)
    parser.add_argument('--hid2', type=int, default=1024)
    parser.add_argument('--hid3', type=int, default=1024)
    parser.add_argument('--norm', type=bool, default=True)
    parser.add_argument('--connect', type=bool, default=False)
    parser.add_argument('--split', type=str, default="1016_new_11/models/split.t7")

    args = parser.parse_args()

    _init_()
    sp = osp.join(osp.dirname(osp.realpath(__file__)), 'outputs', args.split)
    split = torch.load(sp)
    train_indices = split[0]
    test_indices = split[1]


    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data_name = args.dataset_name
    dataset =get_data(data_name)
    # train_indices, test_indices = regression_k_fold(dataset)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GIN0(num_features=args.outdim,num_layers=args.layers,hidden1=args.hid1,hidden2=args.hid2,hidden3=args.hid3,aggr=args.agg,pool=args.pool).to(device)

    model.to(device).reset_parameters()
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]


    if args.norm:
        train_dataset, train_mean, train_std = z_score_normalization(train_dataset)
        test_dataset, test_mean, test_std = z_score_normalization(test_dataset)
        norm_info = {
            "train_mean": train_mean,
            "train_std": train_std,
            "test_mean": test_mean,
            "test_std": test_std,
        }
        torch.save(norm_info, f'outputs/{args.exp_name}/models/norm.t7')
        print("using norm")
    if args.connect:
        dataset=all_connect_global(dataset)
        print("using double direction global")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             num_workers=2, drop_last=False)

    model_dict, best_loss = train(args, io, train_loader,test_loader,model,device)
    io.cprint(f"best loss: {best_loss}")
