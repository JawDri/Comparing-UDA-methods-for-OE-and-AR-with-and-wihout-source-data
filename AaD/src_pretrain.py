import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader, Dataset
from data_list import ImageList
import random, pdb, math, copy
from loss import CrossEntropyLabelSmooth
from sklearn.metrics import confusion_matrix


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

class RandomIntDataset(Dataset):
  def __init__(self, data, labels):
    # we randomly generate an array of ints that will act as data
    self.data = torch.tensor(data)
    # we randomly generate a vector of ints that act as labels
    self.labels = torch.tensor(labels)

  def __len__(self):
    # the size of the set is equal to the length of the vector
    return len(self.labels)

  def __str__(self):
    # we combine both data structures to present them in the form of a single table
    return str(torch.cat((self.data, self.labels.unsqueeze(1)), 1))

  def __getitem__(self, i):
  # the method returns a pair: given - label for the index number i
    return self.data[i], self.labels[i]

class RandomIntDataset_indx(Dataset):
  def __init__(self, data, labels, indx):
    # we randomly generate an array of ints that will act as data
    self.data = torch.tensor(data)
    # we randomly generate a vector of ints that act as labels
    self.labels = torch.tensor(labels)
    self.indx = torch.tensor(indx)

  def __len__(self):
    # the size of the set is equal to the length of the vector
    return len(self.labels)

  def __str__(self):
    # we combine both data structures to present them in the form of a single table
    return str(torch.cat((self.data, self.labels.unsqueeze(1),self.indx.unsqueeze(1)), 1))

  def __getitem__(self, i):
  # the method returns a pair: given - label for the index number i
    return self.data[i], self.labels[i], self.indx[i]



def data_load(args): 
    ## prepare data
    train_bs = args.batch_size
    if args.da == 'uda':
        Source_test = pd.read_csv("./data/Source_test.csv")
        Source_train = pd.read_csv("./data/Source_train.csv")
        Target_train = pd.read_csv("./data/Target_train.csv")
        Target_test = pd.read_csv("./data/Target_test.csv")

        Source_train_data = Source_train.drop(['labels'], axis= 1).values
        Source_train_labels = Source_train.labels.values
        train_source = RandomIntDataset(Source_train_data, Source_train_labels)

        Source_test_data = Source_test.drop(['labels'], axis= 1).values
        Source_test_labels = Source_test.labels.values
        test_source = RandomIntDataset(Source_test_data, Source_test_labels)

        Target_train_data = Target_train.drop(['labels'], axis= 1).values
        Target_train_labels = Target_train.labels.values
        indx = Target_train.index.values
        train_target = RandomIntDataset_indx(Target_train_data, Target_train_labels, indx)


        Target_test_data = Target_test.drop(['labels'], axis= 1).values
        Target_test_labels = Target_test.labels.values
        test_target = RandomIntDataset(Target_test_data, Target_test_labels)

  

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source, batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["source_te"] = DataLoader(test_source, batch_size=train_bs*2, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=train_bs*2, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent


def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.DTNBase().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    smax=100
    #while iter_num < max_iter:
    for epoch in range(args.max_epoch):
        iter_source = iter(dset_loaders["source_tr"])
        for batch_idx, (inputs_source,
                        labels_source) in enumerate(iter_source):

            if inputs_source.size(0) == 1:
                continue

            iter_num += 1
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            feature_src = netB(netF(inputs_source))

            outputs_source = netC(feature_src)
            classifier_loss = CrossEntropyLabelSmooth(
                num_classes=args.class_num, epsilon=args.smooth)(
                    outputs_source, labels_source)

            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()

        #if iter_num % interval_iter == 0 or iter_num == max_iter:
        netF.eval()
        netB.eval()
        netC.eval()
        if args.dset=='visda-2017':
            acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netB, netC, flag=True)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str+'\n')

        if acc_s_te >= acc_init:
            acc_init = acc_s_te
            best_netF = netF.state_dict()
            best_netB = netB.state_dict()
            best_netC = netC.state_dict()

        netF.train()
        netB.train()
        netC.train()

    netF.eval()
    netB.eval()
    netC.eval()
    acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC,flag= True)

    log_str = 'Task: {}; Accuracy on target = {:.2f}%'.format(args.name_src, acc_s_te) + '\n' + acc_list
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netB, netC

def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.DTNBase().cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, acc_list = cal_acc(dset_loaders['test'],
                                netF,
                                netB,
                                netC,
                                flag=True)
    log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neighbors')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=10, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument(
        '--dset',
        type=str,
        default='visda-2017')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='weight/source/')
    parser.add_argument('--da', type=str, default='uda')
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    args = parser.parse_args()

    if args.dset == 'visda-2017':
        names = ['train', 'validation']
        args.class_num = 2


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'


    args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_source(args)
    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    test_target(args)
