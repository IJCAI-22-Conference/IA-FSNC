from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull
import random
import torch

dataset = Amazon(root='./data', name='Photo')

def metatrain_split(arg, train_classes):
    data = dataset[0].to(arg.device)
    node_num = len(data.y)
    labels = (data.y).clone().detach()

    ## 取出meta-learning 中测试集的类别
    class1_idx = []
    class2_idx = []
    class3_idx = []
    class4_idx = []
    # class5_idx = []
    # class6_idx = []
    # class7_idx = []
    # class8_idx = []
    # class9_idx = []
    # class10_idx = []

    # 取出类别的index
    for i in range(len(labels)):
        if (labels[i] == train_classes[0]):
            class1_idx.append(i)
            labels[i] = 0
        elif (labels[i] == train_classes[1]):
            class2_idx.append(i)
            labels[i] = 1
        elif (labels[i] == train_classes[2]):
            class3_idx.append(i)
            labels[i] = 2
        elif (labels[i] == train_classes[3]):
            class4_idx.append(i)
            labels[i] = 3
        # elif (labels[i] == train_classes[4]):
        #     class5_idx.append(i)
        #     labels[i] = 4
        # elif (labels[i] == train_classes[5]):
        #     class6_idx.append(i)
        #     labels[i] = 5
        # elif (labels[i] == train_classes[6]):
        #     class7_idx.append(i)
        #     labels[i] = 6
        # elif (labels[i] == train_classes[7]):
        #     class8_idx.append(i)
        #     labels[i] = 7
        # elif (labels[i] == train_classes[8]):
        #     class9_idx.append(i)
        #     labels[i] = 8
        # elif (labels[i] == train_classes[9]):
        #     class10_idx.append(i)
        #     labels[i] = 9

    # 随机抽取 support set 和 query set
    class1_spt = random.sample(class1_idx, arg.spt_num)
    class2_spt = random.sample(class2_idx, arg.spt_num)
    class3_spt = random.sample(class3_idx, arg.spt_num)
    class4_spt = random.sample(class4_idx, arg.spt_num)
    # class5_spt = random.sample(class5_idx, arg.spt_num)
    # class6_spt = random.sample(class6_idx, arg.spt_num)
    # class7_spt = random.sample(class7_idx, arg.spt_num)
    # class8_spt = random.sample(class8_idx, arg.spt_num)
    # class9_spt = random.sample(class9_idx, arg.spt_num)
    # class10_spt = random.sample(class10_idx, arg.spt_num)

    class1_qry = [n1 for n1 in class1_idx if n1 not in class1_spt]
    class2_qry = [n2 for n2 in class2_idx if n2 not in class2_spt]
    class3_qry = [n3 for n3 in class3_idx if n3 not in class3_spt]
    class4_qry = [n4 for n4 in class4_idx if n4 not in class4_spt]
    # class5_qry = [n5 for n5 in class5_idx if n5 not in class5_spt]
    # class6_qry = [n6 for n6 in class6_idx if n6 not in class6_spt]
    # class7_qry = [n7 for n7 in class7_idx if n7 not in class7_spt]
    # class8_qry = [n8 for n8 in class8_idx if n8 not in class8_spt]
    # class9_qry = [n9 for n9 in class9_idx if n9 not in class9_spt]
    # class10_qry = [n10 for n10 in class10_idx if n10 not in class10_spt]
    class1_qry = random.sample(class1_qry, arg.qry_num)
    class2_qry = random.sample(class2_qry, arg.qry_num)
    class3_qry = random.sample(class3_qry, arg.qry_num)
    class4_qry = random.sample(class4_qry, arg.qry_num)
    # class5_qry = random.sample(class5_qry, arg.qry_num)
    # class6_qry = random.sample(class6_qry, arg.qry_num)
    # class7_qry = random.sample(class7_qry, arg.qry_num)
    # class8_qry = random.sample(class8_qry, arg.qry_num)
    # class9_qry = random.sample(class9_qry, arg.qry_num)
    # class10_qry = random.sample(class10_qry, arg.qry_num)

    spt_idx = class1_spt + class2_spt + class3_spt + class4_spt
    qry_idx = class1_qry + class2_qry + class3_qry + class4_qry
    random.shuffle(spt_idx)
    random.shuffle(qry_idx)

    # mask of train and test
    train_mask = torch.zeros(node_num, dtype=torch.bool)
    for i in range(len(spt_idx)):
        train_mask[spt_idx[i]] = 1

    test_mask = torch.zeros(node_num, dtype=torch.bool)
    for i in range(len(qry_idx)):
        test_mask[qry_idx[i]] = 1

    return data, class1_idx, class2_idx, class3_idx, class4_idx, spt_idx, qry_idx, train_mask, test_mask, labels








def metatest_split(arg, test_classes):
    data = dataset[0].to(arg.device)

    node_num = len(data.y)
    labels = (data.y).clone().detach()

    ## 取出meta-learning 中测试集的类别
    class1_idx = []
    class2_idx = []
    # class3_idx = []
    # class4_idx = []
    # class5_idx = []

    # 取出类别的index
    for i in range(len(labels)):
        if (labels[i] == test_classes[0]):
            class1_idx.append(i)
            labels[i] = 0
        elif (labels[i] == test_classes[1]):
            class2_idx.append(i)
            labels[i] = 1
        # elif (labels[i] == test_classes[2]):
        #     class3_idx.append(i)
        #     labels[i] = 2
        # elif (labels[i] == test_classes[3]):
        #     class4_idx.append(i)
        #     labels[i] = 3
        # elif (labels[i] == test_classes[4]):
        #     class5_idx.append(i)
        #     labels[i] = 4

    # 随机抽取 support set 和 query set
    class1_spt = random.sample(class1_idx, arg.spt_num)
    class2_spt = random.sample(class2_idx, arg.spt_num)
    # class3_spt = random.sample(class3_idx, arg.spt_num)
    # class4_spt = random.sample(class4_idx, arg.spt_num)
    # class5_spt = random.sample(class5_idx, arg.spt_num)
    class1_qry = [n1 for n1 in class1_idx if n1 not in class1_spt]
    class2_qry = [n2 for n2 in class2_idx if n2 not in class2_spt]
    # class3_qry = [n3 for n3 in class3_idx if n3 not in class3_spt]
    # class4_qry = [n4 for n4 in class4_idx if n4 not in class4_spt]
    # class5_qry = [n5 for n5 in class5_idx if n5 not in class5_spt]
    class1_qry = random.sample(class1_qry, arg.qry_num)
    class2_qry = random.sample(class2_qry, arg.qry_num)
    # class3_qry = random.sample(class3_qry, arg.qry_num)
    # class4_qry = random.sample(class4_qry, arg.qry_num)
    # class5_qry = random.sample(class5_qry, arg.qry_num)
    spt_idx = class1_spt + class2_spt
    qry_idx = class1_qry + class2_qry
    random.shuffle(spt_idx)
    random.shuffle(qry_idx)


    # mask of train and test
    train_mask = torch.zeros(node_num, dtype=torch.bool)
    for i in range(len(spt_idx)):
        train_mask[spt_idx[i]] = 1

    test_mask = torch.zeros(node_num, dtype=torch.bool)
    for i in range(len(qry_idx)):
        test_mask[qry_idx[i]] = 1

    return data, class1_idx, class2_idx, spt_idx, qry_idx, train_mask, test_mask, labels

