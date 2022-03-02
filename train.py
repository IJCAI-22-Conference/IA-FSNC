from network import *
from preprocess import *

def metatrain_train(arg, train_classes):
    data, class1_idx, class2_idx, class3_idx, class4_idx, spt_idx, qry_idx, train_mask, test_mask, labels = metatrain_split(arg, train_classes)
    model = Net().to(arg.device)
    train_num = 4

    optimizer1 = torch.optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.classifier.parameters()}], lr = arg.opti1_lr, weight_decay=5e-4)

    # 生成器和classifier
    optimizer2 = torch.optim.Adam([
        {'params': model.gener.parameters()},
        {'params': model.classifier.parameters()}], lr = arg.opti2_lr, weight_decay=5e-4)

    # self-traing, 生成node embeding
    model.train()  # 开始训练
    for epoch in range(arg.epoch):
        optimizer1.zero_grad()
        out = model.forward(data)
        loss = F.nll_loss(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer1.step()

    # ---------------------- self-traing-----------------------
    all_class = class1_idx + class2_idx + class3_idx + class4_idx
    all_idx = spt_idx + qry_idx  # 伪标签节点不能是spt+qry
    new_x = model.confidence(data)
    new_x = new_x[all_class, :]

    labels_mx = []  # 存放每个label的排序后的样本矩阵
    labels_dis = []  # 最终提取出的各个label的样本
    pselabel_num = 0  # 伪标签总数，因为如果固定10个，而训练出判断为该类的不足10个的话，生成噪声会报错

    ## 取出每个label信息熵低的Top-k个
    for i in range(train_num):
        mx = new_x[new_x[:, -1] == i, :]  # 提取
        labels_mx.append(mx[mx[:, -2].argsort()])  # 排序

    for i in range(train_num):
        label_num = []  # 伪标签节点对应的index，不能是 spt和que里面的值
        for j in range(labels_mx[i].shape[0]):
            if int(labels_mx[i][j, 0]) not in all_idx:
                label_num.append(j)
            if len(label_num) == arg.selftrain_num:
                break

        pselabel_num += len(label_num)
        # number, pseudo-label
        labels_dis.append(labels_mx[i][label_num, :][:, [0, 2]])

    ## 给置信度高的 node 打上伪标签
    for i in range(train_num):
        for j in range(labels_dis[i].shape[0]):
            # 在label上打上伪标签
            labels[int(labels_dis[i][j, 0])] = int(labels_dis[i][j, 1])

            # 在训练集 mask 打上伪标签的index
            train_mask[int(labels_dis[i][j, 0])] = 1
            spt_idx.append(int(labels_dis[i][j, 0]))

    # 加上了伪标签的训练
    for epoch in range(arg.epoch):
        optimizer1.zero_grad()
        out = model.forward(data)
        loss = F.nll_loss(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer1.step()

    # ----------------------------------------------------------

    # 提取全部节点的 node embeding
    node_embeding = model.embeding(data)
    node_embeding = node_embeding.detach()  # 去掉梯度

    # support set 的 node embeding
    spt_embeding = []
    for i in range(len(spt_idx)):
        spt_embeding.append(node_embeding[spt_idx[i],])
    spt_embeding = torch.stack(spt_embeding)  # list 转为 torch

    # 加上noise
    spt_embeding = spt_embeding + torch.tensor(np.random.uniform(-0.1, 0.1, [pselabel_num + arg.spt_num*train_num, 60]),
                                               dtype=torch.float32).to(arg.device)  # noise直接往矩阵上面加

    # gener label
    spt_labels = []
    for i in range(len(spt_idx)):
        spt_labels.append(labels[spt_idx[i]])
    spt_labels = torch.tensor(spt_labels).to(arg.device)

    # train gener + classifer
    for epoch in range(arg.augepoch):
        optimizer2.zero_grad()
        out = model.class1gener(spt_embeding)  # class1gener
        loss = F.nll_loss(out, spt_labels)
        loss.backward()
        optimizer2.step()

    # test
    model.eval()
    _, pred = model.classify(node_embeding).max(dim=1)
    correct = int(pred[test_mask].eq(labels[test_mask]).sum().item())
    acc = correct / int(test_mask.sum())
    print('Accuracy: {:.4f}'.format(acc))

    conv1_weight = model.conv1.lin.weight
    gener_weight = model.gener[0].weight

    return conv1_weight, gener_weight







def metatest_train(arg, test_classes, conv1_weight, gener_weight):
    data, class1_idx, class2_idx, spt_idx, qry_idx, train_mask, test_mask, labels = metatest_split(arg, test_classes)
    model = Net2().to(arg.device)
    test_num = 2

    model.conv1.lin.weight, model.gener[0].weight = conv1_weight, gener_weight    # 初始化先验知识的参数
    optimizer1 = torch.optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.classifier.parameters()}], lr = arg.opti1_lr, weight_decay=5e-4)

    # 生成器和classifier
    optimizer2 = torch.optim.Adam([
        {'params': model.gener.parameters()},
        {'params': model.classifier.parameters()}], lr = arg.opti2_lr, weight_decay=5e-4)

    # self-traing, 生成node embeding
    model.train()  # 开始训练
    for epoch in range(arg.epoch):
        optimizer1.zero_grad()
        out = model.forward(data)
        loss = F.nll_loss(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer1.step()

    # ---------------------- self-traing-----------------------
    all_class = class1_idx + class2_idx
    all_idx = spt_idx + qry_idx   # 伪标签节点不能是spt+qry
    new_x = model.confidence(data)
    new_x = new_x[all_class, :]


    labels_mx = []   # 存放每个label的排序后的样本矩阵
    labels_dis = []  # 最终提取出的各个label的样本
    pselabel_num = 0   # 伪标签总数，因为如果固定10个，而训练出判断为该类的不足10个的话，生成噪声会报错

    ## 取出每个label信息熵低的Top-k个
    for i in range(test_num):
        mx = new_x[new_x[:, -1] == i, :]   # 提取
        labels_mx.append(mx[mx[:, -2].argsort()])   # 排序

    for i in range(test_num):
        label_num = []   # 伪标签节点对应的index，不能是 spt和que里面的值
        for j in range(labels_mx[i].shape[0]):
            if int(labels_mx[i][j, 0]) not in all_idx:
                label_num.append(j)
            if len(label_num) == arg.selftrain_num:
                break

        pselabel_num += len(label_num)
        # number, pseudo-label
        labels_dis.append(labels_mx[i][label_num, :][:, [0,2]])

    ## 给置信度高的 node 打上伪标签
    for i in range(test_num):
        for j in range(labels_dis[i].shape[0]):
            # 在label上打上伪标签
            labels[int(labels_dis[i][j, 0])] = int(labels_dis[i][j, 1])

            # 在训练集 mask 打上伪标签的index
            train_mask[int(labels_dis[i][j, 0])] = 1
            spt_idx.append(int(labels_dis[i][j, 0]))

    # 加上了伪标签的训练
    for epoch in range(arg.epoch):
        optimizer1.zero_grad()
        out = model.forward(data)
        loss = F.nll_loss(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer1.step()

    #----------------------------------------------------------

    # 提取全部节点的 node embeding
    node_embeding = model.embeding(data)
    node_embeding = node_embeding.detach()   # 去掉梯度

    # support set 的 node embeding
    spt_embeding = []
    for i in range(len(spt_idx)):
        spt_embeding.append(node_embeding[spt_idx[i], ])
    spt_embeding = torch.stack(spt_embeding)   # list 转为 torch

    # 加上noise
    spt_embeding = spt_embeding + torch.tensor(np.random.uniform(-0.1, 0.1,[pselabel_num + arg.spt_num*test_num, 60]),dtype=torch.float32).to(arg.device)  # noise直接往矩阵上面加

    # gener label
    spt_labels = []
    for i in range(len(spt_idx)):
        spt_labels.append(labels[spt_idx[i]])
    spt_labels = torch.tensor(spt_labels).to(arg.device)

    # train gener + classifer
    for epoch in range(arg.augepoch):
        optimizer2.zero_grad()
        out = model.class1gener(spt_embeding)    # class1gener
        loss = F.nll_loss(out, spt_labels)
        loss.backward()
        optimizer2.step()

    # test
    model.eval()
    _, pred = model.classify(node_embeding).max(dim=1)
    correct = int(pred[test_mask].eq(labels[test_mask]).sum().item())
    acc = correct / int(test_mask.sum())
    print('Accuracy: {:.4f}'.format(acc))

    return acc