import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets
from model_compat import DSN
import torchvision.utils as vutils
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
Source_train = pd.read_csv("/content/drive/MyDrive/DSN/data/Source_train.csv")
Source_test = pd.read_csv("/content/drive/MyDrive/DSN/data/Source_test.csv")
Target_train = pd.read_csv("/content/drive/MyDrive/DSN/data/Target_train.csv")
Target_test = pd.read_csv("/content/drive/MyDrive/DSN/data/Target_test.csv")

FEATURES = list(i for i in Source_train.columns if i!= 'labels')
TARGET = "labels"

from sklearn.preprocessing import StandardScaler
Normarizescaler = StandardScaler()
Normarizescaler.fit(np.array(Source_train[FEATURES]))

class PytorchDataSet(Dataset):
    
    def __init__(self, df):
        
        # for test data, In test data, it's easier to fill it with something on purpose.
        
        if "labels" not in df.columns:
            df["labels"] = 9999
        
        self.df = df
        
        self.train_X = np.array(self.df[FEATURES])
        self.train_Y = np.array(self.df[TARGET])
        self.train_X = Normarizescaler.transform(self.train_X)
        
        
        self.train_X = torch.from_numpy(self.train_X).float()
        self.train_Y = torch.from_numpy(self.train_Y).long()
    
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, idx):
        
        return {"X":self.train_X[idx], "Y":self.train_Y[idx]}

Source_test = PytorchDataSet(Source_test)
Target_test = PytorchDataSet(Target_test)












def test(epoch, name):

    ###################
    # params          #
    ###################
    cuda = True
    cudnn.benchmark = True
    batch_size = 32
    image_size = 9###########9

    ###################
    # load data       #
    ###################


    

   

    model_root = 'model'
    if name == 'mnist':
        mode = 'source'
        image_root = os.path.join('dataset', 'mnist')
        
        dataloader = torch.utils.data.DataLoader(
        dataset=Source_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
)

    elif name == 'mnist_m':
        mode = 'target'
        image_root = os.path.join('dataset', 'mnist_m', 'mnist_m_test')
        test_list = os.path.join('dataset', 'mnist_m', 'mnist_m_test_labels.txt')

        dataloader = torch.utils.data.DataLoader(
        dataset=Target_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    else:
        print('error dataset name')

    ####################
    # load model       #
    ####################

    my_net = DSN()
    checkpoint = torch.load(os.path.join(model_root, 'dsn_mnist_mnistm_epoch_' + str(epoch) + '.pth'))
    my_net.load_state_dict(checkpoint)
    my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    ####################
    # transform image  #
    ####################


    def tr_image(img):

        img_new = (img + 1) / 2

        return img_new


    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    fscr = 0

    while i < len_dataloader:

        data_input = next(data_iter)
        img = data_input['X']
        label = data_input['Y']
        #img, label = data_input

        batch_size = len(label)

        input_img = torch.FloatTensor(batch_size,image_size,1)
        class_label = torch.LongTensor(batch_size,1)

        if cuda:
            img = img.cuda()
            label = label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
        
        
        #print(img.shape,input_img.shape)

        input_img.resize_as_(img).copy_(img)
        input_img = input_img.view(-1, 9,1)###########9
        class_label.resize_as_(label).copy_(label)
        class_label = class_label.view(-1,1)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)

        result = my_net(input_data=inputv_img, mode='source', rec_scheme='share')
        pred = result[3].data.max(1, keepdim=True)[1]

        result = my_net(input_data=inputv_img, mode=mode, rec_scheme='all')
        rec_img_all = tr_image(result[-1].data)

        result = my_net(input_data=inputv_img, mode=mode, rec_scheme='share')
        rec_img_share = tr_image(result[-1].data)

        result = my_net(input_data=inputv_img, mode=mode, rec_scheme='private')
        rec_img_private = tr_image(result[-1].data)
        '''
        if i == len_dataloader - 2:
            vutils.save_image(rec_img_all, name + '_rec_image_all.png', nrow=8)
            vutils.save_image(rec_img_share, name + '_rec_image_share.png', nrow=8)
            vutils.save_image(rec_img_private, name + '_rec_image_private.png', nrow=8)
'''
        n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size
        fscr += f1_score(pred.cpu(), classv_label.data.view_as(pred).cpu(), average='weighted')
        i += 1

    accu = n_correct * 1.0 / n_total
    F1score = fscr/(n_total/batch_size)

    print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, name, accu))
    print('epoch: %d, F1-score of the %s dataset: %f' % (epoch, name, F1score))
