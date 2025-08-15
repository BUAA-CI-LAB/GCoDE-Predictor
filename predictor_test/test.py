import argparse
import sys
from util import IOStream, cal_loss, mape, rmse
import torch
import os.path as osp
sys.path.append(".")
sys.path.append("..")

from torch_geometric.loader import DataLoader
import os

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




def z_score_fit(data_list):
    all_values = torch.cat([d.x[:, -1] for d in data_list], dim=0)
    mean, std = all_values.mean(), all_values.std()
    return mean, std


def z_score_apply(data_list, mean, std):
    for d in data_list:
        d.x[:, -1] = (d.x[:, -1] - mean) / (std + 1e-7)
    return data_list

def test(args, io,test_loader,model,device):

    if args.loss == "mape":
        criterion = cal_loss
    else:
        criterion = rmse
    # criterion = rmse
    test_loss = 0.0
    count = 0.0
    model = model.to(device)
    model = model.eval()
    total_time = 0
    cnt = 0
    pre=[]
    gold = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            batch_size = data.num_graphs

            outs = model(data)
            tag = torch.reshape(data.y, (outs.shape[0], -1))[:, args.device].reshape(outs.shape)

            loss = criterion(outs, tag)
            count += batch_size
            test_loss += loss.item() * batch_size

            cnt += 1

            if args.error_bound:
                io.cprint(" ".join([str(x) for x in (outs.squeeze().cpu().detach().numpy().tolist())]))
                io.cprint(" ".join([str(x) for x in (tag.squeeze().cpu().detach().numpy().tolist())]))

    outstr = 'Test :: test loss: %.6f' % (test_loss / count)

    # loss = criterion(torch.tensor(pre).squeeze(), torch.tensor(gold).squeeze())
    # print(loss)

    print(outstr)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN Predictor')
    parser.add_argument('--exp_name', type=str, default='pi_i7_mn40_40Mbps', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset_name', type=str, default="mn_pi_i7_40Mbps",
                        help='dataset')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model', type=str, default="pi_i7_mn40_40Mbps",
                        help='model_path')
    parser.add_argument('--loss', type=str, default="mape",
                        help='model_path')
    parser.add_argument('--device', type=int, default=-1,
                        help='device id')
    parser.add_argument('--agg', type=str, default="mean")
    parser.add_argument('--pool', type=str, default="sum")
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--outdim', type=int, default=10)
    parser.add_argument('--hid1', type=int, default=1024)
    parser.add_argument('--hid2', type=int, default=1024)
    parser.add_argument('--hid3', type=int, default=1024)
    parser.add_argument('--error_bound', action='store_true')
    parser.add_argument('--no-error_bound', dest='error_bound', action='store_false')
    parser.set_defaults(error_bound=True)
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--no-norm', dest='norm', action='store_false')
    parser.set_defaults(norm=True)
    args = parser.parse_args()


    _init_()
    sp = osp.join(osp.dirname(osp.realpath(__file__)), 'outputs', args.model, "models", "split.t7")
    model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'outputs', args.model,"models","model.t7")
    split = torch.load(sp)
    train_indices = split[0]
    test_indices = split[1]


    io = IOStream('outputs/' + args.model + '/eval.log')
    if args.error_bound == False:
        print(str(args))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data_name = args.dataset_name
    dataset =get_data(data_name)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GIN0(num_features=args.outdim,num_layers=args.layers,hidden1=args.hid1,hidden2=args.hid2,hidden3=args.hid3,aggr=args.agg,pool=args.pool).to(device)


    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')),strict=True)

    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]


    if args.norm:
        train_mean, train_std = z_score_fit(train_dataset)
        train_dataset = z_score_apply(train_dataset, train_mean, train_std)
        test_dataset = z_score_apply(test_dataset, train_mean, train_std)


    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             num_workers=2, drop_last=False)

    test(args, io,test_loader,model,device)






