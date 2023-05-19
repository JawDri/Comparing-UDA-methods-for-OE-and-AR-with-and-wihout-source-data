import argparse
import os
import os.path as osp
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader, Dataset
#from data_list import ImageList_train,ImageList_test
import random
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from memory import MemoryBank
import torch.nn.functional as F
from autoaugment import ImageNetPolicy
#os.environ['CUDA_VISIBLE_DEVICES'] ='0'
def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def eval_initial(memory, loader, netF, netB, netC):
    """Initialize the memory bank after one epoch warm up"""
    netF.eval()
    netB.eval()
    # netC.eval()

    features = torch.zeros(memory.num_samples, memory.num_features)
    labels = torch.zeros(memory.num_samples).long()
    outputs = torch.zeros(memory.num_samples, args.class_num)
    with torch.no_grad():
        for i, (imgs, _, idxs) in enumerate(loader):
            imgs = imgs
            feature = netB(netF(imgs))
            output = netC(feature)
            features[idxs] = feature
            labels[idxs] = (args.class_num + idxs).long()
            outputs[idxs] = torch.softmax(output,dim=-1)
            
        for i in range(args.class_num):
            rank_out = outputs[:,i]
            _,r_idx = torch.topk(rank_out,args.K)
            labels[r_idx] = i

        memory.features = F.normalize(features, dim=1)
        memory.labels = labels
    del features,labels,outputs


def lr_scheduler(optimizer, iter_num, max_iter,flag=False, gamma=10, power=0.75):
    """
    flag=True, the learning rate decays 0.2
    gamma, control the speed of learning rate decay
    """
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    if flag: decay*=0.2
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

    train_size = len(train_target)
    #print(train_target.indx)
    dset_loaders = {}
    
    dset_loaders["target"] = DataLoader(train_target, batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["eval"] = DataLoader(train_target, batch_size=train_bs, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=train_bs*2, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    return dset_loaders,train_size


def cal_acc(loader, netF, netB, netC, memory, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    Fscore = f1_score(all_label, torch.squeeze(predict).float(), average='weighted')
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    
    ## Record the memory information
    m_mask = (memory.labels.cpu()<args.class_num).float()
    m_acc = torch.sum((memory.labels.cpu()==all_label).float() * m_mask.cpu()).item()/float(m_mask.cpu().sum())
    m_num = m_mask.cpu().sum().item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc , m_acc*100, m_num, Fscore*100
    else:
        return accuracy*100, mean_ent, m_acc*100, m_num,Fscore*100

def train_target(args):
    dset_loaders, train_size = data_load(args)
    
    #print(train_size)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.DTNBase()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net)

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck)
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck)

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))

    
    netF = nn.DataParallel(netF)
    netB = nn.DataParallel(netB)
    netC = nn.DataParallel(netC)


    #create memory bank:
    memory = MemoryBank(args.bottleneck, train_size, args,
                            temp=args.temp, momentum=args.momentum)
    

    best_acc = 0
    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netC.named_parameters():
        if args.lr_decay3 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay3}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_s1, _, tar_idx = next(iter_test)
            inputs_w = inputs_s1
            inputs_s = inputs_s1
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_s1, _, tar_idx = next(iter_test)
            inputs_w = inputs_s1
            inputs_s = inputs_s1

        if inputs_w.size(0) == 1:
            continue
        # the SHOT method to generate pseudo labels
        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            netC.eval()
            mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label)
            memory.pred_labels = mem_label
            netF.train()
            netB.train()
            if args.lr_decay3 > 0:
                netC.train()
        
        # training
        inputs_w, inputs_s, inputs_s1 = inputs_w, inputs_s, inputs_s1

        iter_num += 1
        if iter_num>= args.lr_change*interval_iter:
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter,flag=True,gamma=args.gamma)
        else: lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter,flag=False,gamma=args.gamma)

        netF.train()
        netB.train()
        if args.lr_decay3 > 0:
            netC.train()
        features_w = netB(netF(inputs_w))
        outputs_w = netC(features_w)
        
        features_s = netB(netF(inputs_s))
        outputs_s = netC(features_s)

        features_s1 = netB(netF(inputs_s1))
        outputs_s1 = netC(features_s1)

        # update the source-like samples and target-specific samples
        with torch.no_grad():
            p_l = (torch.softmax(outputs_w,dim=-1))
            max_prob, tmp_label = torch.max(p_l,dim=-1)
            mask = max_prob.ge(args.p_threshold).float()
            #print(tar_idx,memory.labels)
            origin_label = memory.labels[tar_idx]
            memory.labels[tar_idx] = (tmp_label.cpu()*mask.cpu() + (1-mask.cpu())*origin_label.cpu()).long()
            
        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_w.cpu(), pred.cpu())
            classifier_loss += nn.CrossEntropyLoss()(outputs_s.cpu(), pred.cpu())
            classifier_loss += nn.CrossEntropyLoss()(outputs_s1.cpu(), pred)
            
            classifier_loss *= args.cls_par*0.65
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0)

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_w)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss.cpu()
            

        if iter_num > interval_iter:
            loss1,loss2= memory(F.normalize(features_w, dim=1),F.normalize(features_s, dim=1),F.normalize(features_s1, dim=1),tar_idx, args.k_nbor)
            classifier_loss += args.lamda_m * loss1
            classifier_loss += args.lamda_ad * loss2

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num==interval_iter:
            # Initialize target-domain instance features and labels
            eval_initial(memory, dset_loaders["eval"], netF, netB, netC)
        
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list, m_acc, m_num, Fscore = cal_acc(dset_loaders['test'], netF, netB, netC, memory, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%. Fscore = {:.2f}%.  \n The memory bank labeled numbers:{} with acc {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te,Fscore, m_num, m_acc) + '\n' + acc_list
                if best_acc < acc_s_te:
                    best_acc = acc_s_te
                    # save the best model
                    print('Save the model with acc:', best_acc)
                    torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
                    torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
                    torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
                    
            else:
                acc_s_te, _ , m_acc, m_num, Fscore = cal_acc(dset_loaders['test'], netF, netB, netC, memory, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%.  Fscore = {:.2f}%. \n The memory bank labeled numbers:{} with acc {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te,Fscore, m_num, m_acc)
                if best_acc < acc_s_te:
                    best_acc = acc_s_te

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()
            if args.lr_decay3 > 0:
                netC.train()

    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()


    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    # print(np.shapelabelset.size())
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'By SHOT psuedo labeling, Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return pred_label.astype('int')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=60, help="max training epochs")
    parser.add_argument('--k_nbor', type=int, default=5, help="select k neiborhoods")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--lr_change', type=int, default=30, help="change the lr epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--gamma', type=int, default=15, help="learning rate scheduler")
    parser.add_argument('--net', type=str, default='res', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0, help="threshold for generate psuedo labels")
    parser.add_argument('--p_threshold', type=float, default=0.97, help="confident threshold for source-like samples")
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--lr_decay3', type=float, default=0, help="fix the classifier layer")
    parser.add_argument('--lamda_m', type=float, default=0.5)
    parser.add_argument('--lamda_ad', type=float, default=0.2)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the memory bank")
    parser.add_argument('--K', type=int, default=300, 
                        help="initial number of each class")
    parser.add_argument('--T', type=str, default='1',
                        help="Time")
    parser.add_argument('--eval_aug', type=str, default='weak',
                        help="types of augmented features in memory bank (this is not important in this version)")
    parser.add_argument('--ad_method', type=str, default='EMMD',
                        help="adaptation metric: EMMD or LMMD(with linear kernel)")

    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 2

    
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    
    np.random.seed(SEED)
    random.seed(SEED)
    
    
    args.interval = args.max_epoch
    print('The gpu device',args.gpu_id)

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)+'lam_m_'+str(args.lamda_m)+ '_' + 'da_' + str(args.lamda_ad) + 'entropy_' + str(args.ent_par) + 'lr_' + str(args.lr) + args.T
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_target(args)

