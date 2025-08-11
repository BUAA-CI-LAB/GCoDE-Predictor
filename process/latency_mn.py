import os
import os.path as osp
import numpy as np
import torch
import sys

sys.path.append('.')
sys.path.append('../../')

from util import judge_exec_pose,compute_communicate,transfer_data_DE,transfer_data_DED_worker,\
    transfer_data_DED_master,transfer_data_ED_master,mode_judge
from util import aggregate_index, _index_to_subop, _index_to_op
from torch_geometric.data import Data, Batch
import scipy.sparse as sp
from torch import Tensor
from torch_geometric.typing import PairOptTensor, PairTensor

try:
    from torch_cluster import knn
except ImportError:
    knn = None
def compute_arch_channel_mn(choice,function):
    in_channel = 3
    out_channel =3
    channel = []
    for i in range(len(choice)):
        x=choice[i]
        if x == 0:
            out_channel = out_channel
        elif x == 1:
            out_channel = _index_to_subop["channel"][function[i]]
        elif x == 2:
            fe, aggr = aggregate_index[function[i]]
            if fe == "extractor_full":
                out_channel = out_channel * 3 + 1
            elif fe == "extractor_234":
                out_channel = out_channel * 2 + 1
            elif fe == "extractor_134":
                out_channel = out_channel * 2 + 1
            elif fe == "extractor_34":
                out_channel = out_channel + 1
            elif fe == "extractor_2":
                out_channel = out_channel
            elif fe == "extractor_13":
                out_channel = out_channel * 2
            elif fe == "extractor_3":
                out_channel = out_channel
            elif fe == "extractor_4":
                out_channel = 1
            else:
                print('wrong feature extractor!')
                sys.exit()
        elif x == 3:
            out_channel = out_channel
        elif x == 4:
            if "||" in _index_to_subop["pool"][function[i]]:
                out_channel = out_channel * 2
            else:
                out_channel = out_channel
        elif x == 5:
            out_channel = out_channel
        else:
            print("wrong choice")
            sys.exit()
        channel.append((in_channel,out_channel))
        in_channel = out_channel
    return channel

def communicate_data(channel,choice,mode):


    index = [i for i, element in enumerate(choice) if element == 5]

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data','mn40_10batch_tuple.pt')

    dataset = torch.load(path)
    x,edge_index,batch = dataset[0]

    if isinstance(x, Tensor):
        x: PairTensor = (x, x)

    if x[0].dim() != 2:
        raise ValueError("Static graphs not supported in DynamicEdgeConv")

    b: PairOptTensor = (None, None)
    if isinstance(batch, Tensor):
        b = (batch, batch)
    elif isinstance(batch, tuple):
        assert batch is not None
        b = (batch[0], batch[1])

    edge_index = knn(x[0], x[1], 20, b[0], b[1]).flip([0])

    if mode == "DE":
        pool_flag = False
        for i in range(0,index[0]):
            if choice[i] == 4:
                pool_flag = True
                break
        dim = channel[index[0]]
        if pool_flag:
            x_new = torch.randn(1, dim[-1])
        else:
            x_new = torch.randn(1024, dim[-1])
        x_new2 = torch.randn(1, 40)
        data1 = transfer_data_DE(x_new, edge_index, batch, choice)
        data2 = (x_new2,None,None)
    elif mode == "DED":
        dim1, dim2 = channel[index[0]],channel[index[1]]
        pool_flag = False
        pool_flag2 = False
        for i in range(0,index[0]):
            if choice[i] == 4:
                pool_flag = True
                break
        if not pool_flag:
            for i in range(index[0], index[1]):
                if choice[i] == 4:
                    pool_flag2 = True
                    break
        else:
            pool_flag2 = True
        if pool_flag:
            x_new = torch.randn(1, dim1[-1])
            x_new2 = torch.randn(1, dim2[-1])
        else:
            x_new = torch.randn(1024, dim1[-1])
            if pool_flag2:
                x_new2 = torch.randn(1, dim2[-1])
            else:
                x_new2 = torch.randn(1024, dim2[-1])
        data1 = transfer_data_DED_worker(x_new,edge_index,batch,choice)
        data2 = transfer_data_DED_master(x_new2,edge_index,batch,choice,"mn40")

    elif mode == "ED":
        dim1, dim2 = channel[index[0]], channel[index[1]]

        pool_flag2 = False

        for i in range(index[0], index[1]):
            if choice[i] == 4:
                pool_flag2 = True
                break

        x_new1 = torch.randn(1024, 3)
        if pool_flag2:
            x_new2 = torch.randn(1, dim2[-1])
        else:
            x_new2 = torch.randn(1024, dim2[-1])
        data1 = (x_new1,edge_index,batch)
        data2 = transfer_data_ED_master(x_new2, edge_index, batch, choice,"mn40")
    elif mode == "E":
        x_new1 = torch.randn(1024, 3)
        x_new2 = torch.randn(1, 40)
        data1 = (x_new1, None, batch)
        data2 = (x_new2,None,None)
    else:
        print("wrong mode")
        sys.exit()
    return data1,data2
def generate_adj(gc="ori"):

    if gc=="ori":
        A = np.zeros(shape=[18, 18])
        for i in range(0, 18):
            A[i][i] = 1
            A[i][0] = 1
        A[1][2] = 1
        for i in range(0, 15):
            A[i + 2][i + 3] = 1
    elif gc=='io_g':
        A = np.zeros(shape=[18, 18])
        for i in range(0, 18):
            A[i][i] = 1
        A[1][2] = 1
        for i in range(0, 15):
            A[i + 2][i + 3] = 1
        A[1][0] = 1
        A[-1][0] = 1
    else:
        print("wrong graph")
        sys.exit()
    return A


def generate_feature(device,edge,choice,function,bandwidth,feature_method='cat',gc='ori'):

    ms=choice+choice
    mode = mode_judge(ms)
    channel = compute_arch_channel_mn(choice, function)
    device_lut = torch.load(os.path.join(osp.dirname(osp.realpath(__file__)),'..', 'lut_lat',device ,'LUT.pkl'))
    edge_lut = torch.load(os.path.join(osp.dirname(osp.realpath(__file__)), '..', 'lut_lat',edge, 'LUT.pkl'))
    exc_pos = judge_exec_pose(choice)
    if feature_method=='lat':
        initial_features_tensor = torch.empty(18, dtype=torch.float64)
        initial_features_tensor[0] = 0  # global node
        initial_features_tensor[1] = 0  # input node
        if mode=="DE":
            data1, data2 = communicate_data(channel, choice, mode)
            lat = compute_communicate(data2,bandwidth)
            initial_features_tensor[-1] = lat  # output node
        else:
            initial_features_tensor[-1] = 0  # output node
        for i,op in enumerate(choice):
            op_type = _index_to_op[op]
            in_dim,out_dim = channel[i]
            if op_type == "graph_process":
                op_type = 'knn'
            if op_type == 'connect':
                lat = 0
            elif op_type == "combine":
                pool_flag=False
                for j in range(i):
                    if _index_to_op[choice[j]]=='pool':
                        pool_flag =True
                if pool_flag:
                    key_to_search = (op_type, "afterpool",int(in_dim), int(out_dim))
                else:
                    key_to_search = (op_type, "nopool", int(in_dim), int(out_dim))
                if key_to_search not in device_lut:
                    print(f"The key {key_to_search} does not exist in the data.")
                    sys.exit()
                if exc_pos[i]==0:
                    lat = device_lut[key_to_search]
                elif exc_pos[i]==1:
                    lat = edge_lut[key_to_search]
                else:
                    print("error")
                    sys.exit()
            elif op_type == "communicate":
                data1,data2 = communicate_data(channel,choice,mode)
                indices_of_element = [i for i, element in enumerate(choice) if element == 5]

                if i == indices_of_element[0]:
                    lat =compute_communicate(data1,bandwidth)
                else:
                    if len(indices_of_element)>1:
                        lat = compute_communicate(data2, bandwidth)
                    else:
                        print("worng index")
                        sys.exit()
            else:
                key_to_search = (op_type, in_dim, out_dim)
                if key_to_search not in device_lut:
                    print(f"The key {key_to_search} does not exist in the data.")
                    sys.exit()
                if exc_pos[i]==0:
                    lat = device_lut[key_to_search]
                elif exc_pos[i]==1:
                    lat = edge_lut[key_to_search]
                else:
                    print("error")
                    sys.exit()
            initial_features_tensor[i+2] = lat

        pool_flag = False
        for j in range(12):
            if _index_to_op[choice[j]] == 'pool':
                pool_flag = True
        if pool_flag:
            key_to_search1 = ("combine", "afterpool", int(channel[-1][-1]), 512)
            key_to_search2 = ("combine", "afterpool",512, 256)
            key_to_search3 = ("combine", "afterpool",256, 40)
        else:
            key_to_search1 = ("combine", "nopool", int(channel[-1][-1]), 512)
            key_to_search2 = ("combine", "nopool",512, 256)
            key_to_search3 = ("combine", "nopool",256, 40)

        if key_to_search1 not in device_lut:
            print(f"The key {key_to_search1} does not exist in the data.")
            sys.exit()
        else:
            if mode != "DE":
                lat1 = device_lut[key_to_search1]
                lat2 = device_lut[key_to_search2]
                lat3 = device_lut[key_to_search3]

            else:
                lat1 = edge_lut[key_to_search1]
                lat2 = edge_lut[key_to_search2]
                lat3 = edge_lut[key_to_search3]
        initial_features_tensor[14] = lat1
        initial_features_tensor[15] = lat2
        initial_features_tensor[16] = lat3
        initial_features_tensor = initial_features_tensor.view(-1, 1)
    elif feature_method == 'cat':
        x = np.zeros((18, 10))
        x[0][6] = 1  # global node
        x[1][7] = 1  # input node
        x[-1][8] = 1  # output node
        # lat fuzhi
        x[0][-1] = 0
        x[1][-1] = 0

        if mode=="DE":
            data1, data2 = communicate_data(channel, choice, mode)
            lat = compute_communicate(data2,bandwidth)
            x[-1][-1] = lat  # output node
        else:
            x[-1][-1] = 0  # output node

        exc_pos = judge_exec_pose(choice)
        for i,op in enumerate(choice):
            op_type = _index_to_op[op]
            in_dim,out_dim = channel[i]
            if op_type == "graph_process":
                op_type = 'knn'
            if op_type == 'connect':
                lat = 0
            elif op_type == "combine":
                pool_flag=False
                for j in range(i):
                    if _index_to_op[choice[j]]=='pool':
                        pool_flag =True
                if pool_flag:
                    key_to_search = (op_type, "afterpool",int(in_dim), int(out_dim))
                else:
                    key_to_search = (op_type, "nopool", int(in_dim), int(out_dim))
                if key_to_search not in device_lut:
                    print(f"The key {key_to_search} does not exist in the data.")
                    sys.exit()
                if exc_pos[i]==0:
                    lat = device_lut[key_to_search]
                elif exc_pos[i]==1:
                    lat = edge_lut[key_to_search]
                else:
                    print("error")
                    sys.exit()
            elif op_type == "communicate":
                data1,data2 = communicate_data(channel,choice,mode)
                indices_of_element = [i for i, element in enumerate(choice) if element == 5]

                if i == indices_of_element[0]:
                    lat =compute_communicate(data1,bandwidth)
                else:
                    if len(indices_of_element)>1:
                        lat = compute_communicate(data2, bandwidth)
                    else:
                        print("worng index")
                        sys.exit()
            else:
                key_to_search = (op_type, in_dim, out_dim)
                if key_to_search not in device_lut:
                    print(f"The key {key_to_search} does not exist in the data.")
                    sys.exit()
                if exc_pos[i]==0:
                    lat = device_lut[key_to_search]
                elif exc_pos[i]==1:
                    lat = edge_lut[key_to_search]
                else:
                    print("error")
                    sys.exit()
            x[i + 2][op] = 1
            x[i+2][-1] = lat

        pool_flag = False
        for j in range(12):
            if _index_to_op[choice[j]] == 'pool':
                pool_flag = True
        if pool_flag:
            key_to_search1 = ("combine", "afterpool", int(channel[-1][-1]), 512)
            key_to_search2 = ("combine", "afterpool",512, 256)
            key_to_search3 = ("combine", "afterpool",256, 40)
        else:
            key_to_search1 = ("combine", "nopool", int(channel[-1][-1]), 512)
            key_to_search2 = ("combine", "nopool",512, 256)
            key_to_search3 = ("combine", "nopool",256, 40)

        if key_to_search1 not in device_lut:
            print(f"The key {key_to_search1} does not exist in the data.")
            sys.exit()
        else:
            if mode != "DE":
                lat1 = device_lut[key_to_search1]
                lat2 = device_lut[key_to_search2]
                lat3 = device_lut[key_to_search3]

            else:
                lat1 = edge_lut[key_to_search1]
                lat2 = edge_lut[key_to_search2]
                lat3 = edge_lut[key_to_search3]

        x[-4][1]=1
        x[-4][-1] = lat1
        x[-3][1]=1
        x[-3][-1] = lat2
        x[-2][1]=1
        x[-2][-1] = lat3

        initial_features_tensor = torch.from_numpy(x.astype(np.float64))
    elif feature_method == 'onehot':
        x = np.zeros((18, 10))
        x[0][6] = 1  # global node
        x[1][7] = 1  # input node
        x[-1][8] = 1  # output node
        x[0][-1] = 0
        x[1][-1] = 0
        for i in range(0, 12):
            x[i + 2][int(choice[i])] = 1
            x[i + 2][-1] = exc_pos[i]
        if mode != "DE":
            x[-4][1] = 1
            x[-4][-1] = 0
            x[-3][1] = 1
            x[-3][-1] = 0
            x[-2][1] = 1
            x[-2][-1] = 0
            x[-1][-1] = 0

        else:
            x[-4][1] = 1
            x[-4][-1] = 1
            x[-3][1] = 1
            x[-3][-1] = 1
            x[-2][1] = 1
            x[-2][-1] = 1
            x[-1][-1] = 1
        initial_features_tensor = torch.from_numpy(x.astype(np.float64))

    elif feature_method == 'hgnas_new':
        x = np.zeros((18, 21))
        x[0][6] = 1  # global node
        x[1][7] = 1  # input node
        x[-1][8] = 1  # output node
        x[0][-1] = 0
        x[1][-1] = 0
        for i in range(0, 12):
            x[i + 2][int(choice[i])] = 1
            x[i + 2][int(function[i]) + 9] = 1
            x[i + 2][-1] = exc_pos[i]
        if mode != "DE":
            x[-4][1] = 1
            x[-4][10] = 1
            x[-4][-1] = 0
            x[-3][1] = 1
            x[-3][10] = 1
            x[-3][-1] = 0
            x[-2][1] = 1
            x[-2][-1] = 0
            x[-2][10] = 1
            x[-1][-1] = 0

        else:
            x[-4][1] = 1
            x[-4][-1] = 1
            x[-3][1] = 1
            x[-3][-1] = 1
            x[-2][1] = 1
            x[-2][-1] = 1
            x[-1][-1] = 1

            x[-2][10] = 1
            x[-3][10] = 1
            x[-4][10] = 1
        initial_features_tensor = torch.from_numpy(x.astype(np.float64))

    else:
        print("wrong feature methos")
        sys.exit()

    return initial_features_tensor







def predictor_infor_data(choice ,function,device_name,edge_name,bandwidth,mean,std,gc="ori",feature_method='cat'):
    data_l = []
    G = generate_adj(gc)

    x = generate_feature(device_name, edge_name, choice, function, bandwidth, feature_method=feature_method, gc=gc)
    last_x = x[:, -1]
    LUT_lat = last_x.sum()

    exc_pos = judge_exec_pose(choice)
    a = choice + function
    mode = mode_judge(a)
    exc_pos_tensor = torch.tensor(exc_pos)
    choice_tensor = torch.tensor(choice)

    mask = (exc_pos_tensor == 0) & (choice_tensor != 5)

    selected_elements = x[2:14][mask]
    last_elements = selected_elements[:, -1]
    if mode == "DE":
        sum_result = last_elements.sum()
    else:
        additional_elements = x[-4:-1]
        selected_elements = torch.cat((selected_elements,additional_elements))
        last_elements = selected_elements[:, -1]
        sum_result = last_elements.sum()
    mask_5 = choice_tensor == 5
    selected_elements_5 = x[2:14][mask_5]
    last_elements = selected_elements_5[:, -1]
    commu_time = last_elements.sum()

    y=None
    edge_tmp = sp.coo_matrix(G)
    indices = np.vstack((edge_tmp.row, edge_tmp.col))
    edge = torch.LongTensor(indices)
    # print(edge)
    data = Data(x=x, edge_index=edge)

    data.x[:, -1] = (data.x[:, -1] - mean) / (std + 1e-7)

    data_l.append(data)
    return Batch.from_data_list(data_list=data_l),sum_result,commu_time,LUT_lat


def load_data_LUT(path,device_name,edge_name,norm=None,bandwidth = 40,gc="ori",feature_method='cat'):
    data_list = []
    with open(os.path.join(path)) as f:
        f_content = f.readlines()
        for row in f_content:
            data = row.strip().split(',')
            choice = [int(x) for x in data[0].split()]
            function = [int(x) for x in data[1].split()]
            total_time = np.float64(data[2])
            # 12 layers，op input output global,

            G = generate_adj(gc)

            x = generate_feature(device_name,edge_name,choice,function,bandwidth,feature_method=feature_method,gc=gc)

            y = torch.from_numpy(np.array([total_time]).astype(np.float64))

            edge_tmp = sp.coo_matrix(G)
            indices = np.vstack((edge_tmp.row, edge_tmp.col))
            edge = torch.LongTensor(indices)

            data = Data(x=x, y=y, edge_index=edge)

            data_list.append(data)

        print(len(data_list))

    return data_list




if __name__ == '__main__':
    dataset_name = "mn_tx2_1060_40_new"
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..','dataset', 'mn_tx2_1060_40mbps.txt')
    graph_cons = 'ori'
    f = 'cat'
    processed_path = os.path.join(osp.dirname(osp.realpath(__file__)), '..', 'dataset', dataset_name, 'data.pt')
    # ld = load_data_LUT(path, device_name='tx2', edge_name='1060', norm=None,bandwidth=40,gc=graph_cons,feature_method=f)
    # torch.save(ld, processed_path)
    a = torch.load(processed_path)
    print(a)
