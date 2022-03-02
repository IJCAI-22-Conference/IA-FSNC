import random
import argparse
from preprocess import *
from train import *
import numpy as np
import time

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def arg_parse():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='FewGraph arguments.')
    parser.add_argument('--spt_num', type=int, help='supprt set number')
    parser.add_argument('--qry_num', type=int, help='query set number')
    parser.add_argument('--selftrain_num', type=int, help='Pseudo-label node number')
    parser.add_argument('--augepoch', type=int, help='epoch')
    parser.add_argument('--epoch', type=int, help='epoch')
    parser.add_argument('--feature_dim', type=int, help='feature')
    parser.add_argument('--class_num', type=int, help='class num')
    parser.add_argument('--opti1_lr', type=float, help='conv1 + conv2 + classifier')
    parser.add_argument('--opti2_lr', type=float, help='gener + classifier')
    parser.add_argument('--device', type=str, help='cuda or cpu')

    parser.set_defaults(spt_num = 1,
                        qry_num = 12,
                        selftrain_num = 30,
                        augepoch = 50,
                        epoch = 100,
                        class_num = 8,
                        opti1_lr = 0.01,
                        opti2_lr = 0.01,
                        device = device
                       )
    return parser.parse_args()

if __name__ == '__main__':
    start = time.time()

    arg = arg_parse()
    all_classes = list(range(arg.class_num))
    results = []

    for i in range(100, 200):
        print("meta train:")
        print("{}-th division: ".format(i+1))
        random.seed(i)
        test_classes = random.sample(all_classes, 2)
        train_classes = [a1 for a1 in all_classes if a1 not in test_classes]
        train_classes = random.sample(train_classes, 4)

        # meta_train
        conv1_weight, gener_weight = metatrain_train(arg, train_classes)

        # 同一个数据划分下的10次随机种子
        print("meta test:")
        for seed in range(100,110):
            # meta_test
            setup_seed(seed)
            result = metatest_train(arg, test_classes, conv1_weight, gener_weight)
            results.append(result)

    print(np.mean(results))
    print(np.var(results))
    # print(np.std(results))



    end = time.time()
    print("该程序运行的总时间:",round(end-start,2),"秒")
    print("该程序运行的总时间:",round((end-start)/60,2),"分")
