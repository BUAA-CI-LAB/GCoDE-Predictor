import ast
import pickle
import sys
import zlib

import numpy as np
import torch

_op_to_index = {
    'connect': 0,
    'combine': 1,
    'aggregate': 2,
    'graph_process': 3,
    'pool': 4,
    'communicate': 5
}
_index_to_op = {
    0 : 'connect',
    1 : 'combine',
    2 : 'aggregate',
    3 : 'graph_process',
    4 : 'pool',
    5 : 'communicate'
}

_op_to_subop = {
    'connect': ['connect'],
    'combine': ['channel'],
    'aggregate': ['extractor', 'aggregator'],
    'graph_process': ['gp'],
    'pool': ['pool'],
    'communicate': ['communicate'],
}

aggregate_index = {
    0: ['extractor_full','max'],
    1: ['extractor_full', 'mean'],
    2: ['extractor_full', 'add'],
    3: ['extractor_2','max'],
    4: ['extractor_2', 'mean'],
    5: ['extractor_2', 'add'],
    6: ['extractor_13', 'max'],
    7: ['extractor_13', 'mean'],
    8: ['extractor_13', 'add'],
    9: ['extractor_3', 'max'],
    10: ['extractor_3', 'mean'],
    11: ['extractor_3', 'add'],
}

index_aggregate = {
    ('extractor_full', 'max'): 0,
    ('extractor_full', 'mean'): 1,
    ('extractor_full', 'add'): 2,
    ('extractor_2', 'max'): 3,
    ('extractor_2', 'mean'): 4,
    ('extractor_2', 'add'): 5,
    ('extractor_13', 'max'): 6,
    ('extractor_13', 'mean'): 7,
    ('extractor_13', 'add'): 8,
    ('extractor_3', 'max'): 9,
    ('extractor_3', 'mean'): 10,
    ('extractor_3', 'add'): 11,
}



_subop_to_index = {
    'connect': {
        # 'skip_connect': 0,
        'Identity': 0,
    },
    'channel': {
        8 : 0,
        16: 1,
        32: 2,
        64: 3,
        128: 4,
        256: 5,
        512: 6,
    },
    'extractor': {
        'extractor_full': 0,
        'extractor_2': 1,
        'extractor_13': 2,
        'extractor_3': 3,
    },
    'aggregator': {
        'max': 0,
        'mean': 1,
        'add': 2,
    },
    'gp': {
        'knn': 0,
        # 'random': 1,
    },
    'pool': {
        "max": 0,
        "mean": 1,
        'add': 2,
        "max||mean": 3,
        "max||add": 4,
        'mean||add': 5,
    },
    'communicate': {
        'communicate': 0,
    },
}


_index_to_subop = {
    'connect': {
        0: 'Identity'
    },
    'channel': {
        0: 8,
        1: 16,
        2: 32,
        3: 64,
        4: 128,
        5: 256,
        6: 512,
    },
    'extractor': {
        0: 'extractor_full',
        1: 'extractor_2',
        2: 'extractor_13',
        3: 'extractor_3',
    },
    'aggregator': {
        0: 'max',
        1: 'mean',
        2: 'add',
    },
    'gp': {
        0: 'knn',
    },
    'pool': {
        0: "max",
        1: "mean",
        2: 'add',
        3: "max||mean",
        4: "max||add",
        5: 'mean||add',
    },
    'communicate': {
        0: 'communicate',
    }
}




# 用于测量时间时排除异常值，返回的是列表，使用的时候要用个平均，还需要判断，如果列表为空，则这次测量不行，重新测
def time_mean(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    d = [x for x in data if lower_bound <= x <= upper_bound]
    average = np.mean(d)

    return average


# 计算通信模块的延迟
def compute_communicate(data, bandwidth_mbps):

    data = compress_and_convert(data)

    data_size = len(data)  # 数据大小，单位是字节

    # 计算传输时间，单位是秒
    transfer_time_seconds = (data_size * 8) / (bandwidth_mbps * 10 ** 6)

    # 转换为毫秒
    transfer_time_ms = transfer_time_seconds * 1000

    return transfer_time_ms



# model_str用list把
def model_str_revert(model_str, lyaers=12):
    if isinstance(model_str, list):
        # 提取原始数组
        # 根据每个原始数组的长度，将大列表分割成三个子列表
        op = model_str[:lyaers]
        function = model_str[lyaers:]

    # 用于发送的模型信息
    elif isinstance(model_str, str):
        actual_list = ast.literal_eval(model_str)
        op = actual_list[:lyaers]
        function = actual_list[lyaers:]
    return op, function


def judge_exec_pose(choice):
    exec_pos = []
    d=0
    for i in range(len(choice)):
        exec_pos.append(d)
        if choice[i] == 5: # communicate op
            d=int(not d)

    return exec_pos



def transfer_data_DED_worker(x, edge_index, batch, op):
    first_5_found = False
    batch_t = False
    for i, num in enumerate(op):
        if not first_5_found and num == 5:
            first_5_found = True
            continue
        if first_5_found:
            if num == 5:  # 如果找到第二个5，退出循环
                break
            if num == 3:
                batch_t = True
                edge_index = None
                break
            elif num == 2:
                break
            elif num == 4:
                batch_t = True
    if not batch_t:
        batch = None
    data_type = (x,edge_index,batch)
    return data_type

def transfer_data_DED_master(x, edge_index, batch,  op, dataset):

    if dataset=="mr":
        return (x,None,None)
    else:
        if op[-1] == 5:  # 如果第二个5是列表的最后一个元素
            edge_index = None
            batch = None
        else:
            first_5_found = False
            second_5_found = False
            batch_t = False
            for i, num in enumerate(op):
                if not first_5_found and num == 5:
                    first_5_found = True
                    continue
                if first_5_found and not second_5_found:
                    if num == 5:  # 找到第二个5，开始判断
                        second_5_found = True
                        continue
                if second_5_found:
                    if num == 3:
                        batch_t = True
                        edge_index = None
                        break
                    elif num == 2:
                        break
                    elif num == 4:
                        batch_t = True
            if not batch_t:
                batch = None

        data_type = (x,edge_index,batch)
        return data_type

def transfer_data_ED_master(x, edge_index, batch, op,dataset):
    if dataset=="mr":
        return (x,None,None)
    else:
        if op[-1] == 5:  # 如果第二个5是列表的最后一个元素
            edge_index = None
            batch = None
        else:
            first_5_found = False
            second_5_found = False
            batch_t = False
            for i, num in enumerate(op):
                if not first_5_found and num == 5:
                    first_5_found = True
                    continue
                if first_5_found and not second_5_found:
                    if num == 5:  # 找到第二个5，开始判断
                        second_5_found = True
                        continue
                if second_5_found:
                    if num == 3:
                        batch_t = True
                        edge_index = None
                        break
                    elif num == 2:
                        break
                    elif num == 4:
                        batch_t = True
            if not batch_t:
                batch = None
        data = (x, edge_index, batch)
        return data

def decompress_and_convert(data):
    # 解压缩
    decompressed_data = zlib.decompress(data)

    # 使用pickle加载数据
    A = pickle.loads(decompressed_data)
    x, edge_index, batch = A

    # 如果edge_index为int16，则转换为int64
    if edge_index is not None:
        if edge_index.dtype == torch.int16:
            edge_index = edge_index.to(dtype=torch.int64)

    # 如果batch为int8，则转换为int64
    if batch is not None:
        if batch.dtype == torch.int8:
            batch = batch.to(dtype=torch.int64)

    return x, edge_index, batch

def compress_and_convert(A):
    x, edge_index, batch = A

    # 如果edge_index不为None，则转换为int16
    if edge_index is not None and edge_index.dtype == torch.int64:
        edge_index = edge_index.to(dtype=torch.int16)

    # 如果batch不为None，则转换为int8
    if batch is not None and batch.dtype == torch.int64:
        batch = batch.to(dtype=torch.int8)

    # 使用pickle进行序列化
    serialized_data = pickle.dumps((x, edge_index, batch))

    # 压缩
    compressed_data = zlib.compress(serialized_data)

    return compressed_data

def transfer_data_DE(x, edge_index, batch, op):
    if op[-1] != 5:
        first_5_found = False
        batch_t = False
        for i, num in enumerate(op):
            if not first_5_found and num == 5:
                first_5_found = True
                continue
            if first_5_found:
                if num == 5:  # 如果找到第二个5，退出循环
                    break
                if num == 3:
                    batch_t = True
                    edge_index = None
                    break
                elif num == 2:
                    break
                elif num == 4:
                    batch_t = True
        if not batch_t:
            batch = None
    else:
        batch = None
        edge_index = None
    data = (x,edge_index,batch)
    return data








def mode_judge(model_str, lyaers=12):
    mode=""
    op, fuction = model_str_revert(model_str, lyaers=lyaers)
    pose = judge_exec_pose(op)
    dev = pose
    x = []
    if op[0] == 5:
        dev = dev[1:]
    for i in dev:
        if i not in x:
            x.append(int(i))
        if i != x[-1]:
            x.append(int(i))
    if op[-1]==5:
        if x[-1]==1:
            x.append(0)
        else:
            x.append(1)
    l = len(x)
    if 0 not in x:
        mode = "E"
    elif 1 not in x:
        mode = "D"
    elif l==2 and (x[0]==0 and x[1]==1):
        mode = "DE"
    elif l == 2 and (x[1] == 0 and x[0] == 1):
        mode = "ED"
    elif l == 3 and (x[0] == 0 and x[1] == 1 and x[2] == 0):
        mode = "DED"
    return mode

# msg 必须是str
def msg_struct(msg, msg_type):
    if not isinstance(msg,str):
        msg = str(msg)
    if msg_type=="model":
        model_len = str(len(msg))
        if len(model_len) > 8:
            print("error model_len len")
            sys.exit()
        while len(model_len) < 8:
            model_len = "0" + model_len
        struct_msg=model_len+msg
    elif msg_type=="length":
        if len(msg) > 8:
            print("error msg len")
            sys.exit()
        while len(msg) < 8:
            msg = "0" + msg
        struct_msg = msg
    else:
        print("wrong msg")
        sys.exit()
    return struct_msg

import sys
class Logger(object):
    def __init__(self, file_path: str = "./Default.log"):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def cal_loss(pred, gold, smoothing=True):
    loss = torch.abs((gold - pred) / gold) * 100
    # loss = torch.mean(torch.sum(loss, dim=1))
    loss = torch.mean(loss)
    return loss


def mape(pred, actual):
    actual, pred = actual.detach().numpy(), pred.detach().numpy()
    return np.mean(np.abs(actual - pred) / actual, axis=0) * 100


def rmse(pred, gold):
    a= torch.mean(torch.square(gold - pred))
    loss = torch.sqrt(a)
    return loss

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()