import argparse
import torch

def rmse(pred, gold):
    x = gold - pred
    y= torch.square(x)
    a= torch.mean(y)
    loss = torch.sqrt(a)
    return loss

def mape(pred, gold):
    loss = torch.abs((gold - pred) / gold) * 100
    # loss = torch.mean(torch.sum(loss, dim=1))
    loss = torch.mean(loss)
    return loss

# B为真实值
def relative_size_accuracy(A, B):
    assert len(A) == len(B), "Both lists must have the same length."

    correct_count = 0
    total_pairs = 0

    for i in range(len(A)):
        for j in range(i + 1, len(A)):  # 避免重复对和自对比
            if (A[i] > A[j] and B[i] > B[j]) or (A[i] < A[j] and B[i] < B[j]):
                correct_count += 1
            total_pairs += 1

    return correct_count / total_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN Predictor')
    parser.add_argument('--exp_name', type=str, default='pi_i7_mn40_40Mbps',
                        help='Name of the experiment')
    args = parser.parse_args()
    pre_total = []
    gt_total = []
    with open(f'outputs/{args.exp_name}/eval.log', 'r') as f:
        data = f.readlines()

        for i in range(int(len(data) / 2)):
            pre = data[2 * i].split()
            gt = data[2 * i + 1].split()
            pre = [float(x) for x in pre]
            gt = [float(x) for x in gt]
            pre_total.extend(pre)
            gt_total.extend(gt)
    accuracy = relative_size_accuracy(pre_total, gt_total)

    print(f"Relative prediction accuracy： {accuracy}")
    pct1 = 0
    pct5 = 0
    pct10 = 0
    pct20 = 0
    for i in range(len(pre_total)):
        low = gt_total[i]
        high = gt_total[i]
        low_ = 0.99 * low
        high_ = 1.01 * high
        if pre_total[i] < high_ and pre_total[i] > low_:
            pct1 += 1
        low_ = 0.95 * low
        high_ = 1.05 * high
        if pre_total[i] < high_ and pre_total[i] > low_:
            pct5 += 1
        low_ = 0.9 * low
        high_ = 1.1 * high
        if pre_total[i] < high_ and pre_total[i] > low_:
            pct10 += 1
        low_ = 0.8 * low
        high_ = 1.2 * high
        if pre_total[i] < high_ and pre_total[i] > low_:
            pct20 += 1

    print('error bound (1%): ', pct1 / len(pre_total))
    print('error bound (5%): ',pct5 / len(pre_total))
    print('error bound (10%): ',pct10 / len(pre_total))
    print('error bound (20%): ', pct20 / len(pre_total))


    pred = torch.tensor(pre_total)
    gold = torch.tensor(gt_total)
    ab =torch.abs(pred - gold)
    max_error = torch.max(ab)

    rs = rmse(pred,gold)
    mp = mape(pred,gold)

    print('rmse (ms): ', rs)
    print('MAPE (%): ', mp)

    print('max_error: ', max_error)
