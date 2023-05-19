import numpy
import os
import sys
import util
import pandas as pd
#from urlparse import urljoin
from urllib.parse import urljoin
import gzip
import struct
import operator
import numpy as np
#from preprocessing import preprocessing
def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]
class USPS:
        base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'

        data_files = {
        'train': 'zip.train.gz',
        'test': 'zip.test.gz'
        }
        def __init__(self,path=None,shuffle=True,output_size=[1,9],output_channel=1,split='train',select=[], unbalance=1.):######
                self.image_shape=(1,9)######32
                self.label_shape=()     
                self.path=path
                self.shuffle=shuffle
                self.output_size=output_size
                self.output_channel=output_channel
                self.split=split
                self.select=select
                self.unbalance=unbalance
                self.num_classes = 2#######2
                self.download()
                self.pointer=0
                self.load_dataset()
        def download(self):
                data_dir = self.path
                if not os.path.exists(data_dir):
                        os.mkdir(data_dir)
                for filename in self.data_files.values():
                        path = self.path+'/'+filename
                        if not os.path.exists(path):
                                url = urljoin(self.base_url, filename)
                                util.maybe_download(url, path)
        def shuffle_data(self):
                images = self.images[:]
                labels = self.labels[:]
                self.images = []
                self.labels = []

                idx = np.random.permutation(len(labels))
                for i in idx:
                        self.images.append(images[i])
                        self.labels.append(labels[i])
        

        def load_dataset(self):
                
                if self.split=='train':
                        Source_train = pd.read_csv("./data/Source_train.csv")
                        labels = Source_train.labels
                        images = Source_train.drop(['labels'], axis= 1)
                        labels = np.array(labels, dtype=np.int32)
                        images = np.array(images, dtype=np.float32).reshape(-1, 1, 9)#######32
                        self.images = images
                        self.labels = labels
                elif self.split=='test':
                        Source_test = pd.read_csv("./data/Source_test.csv")
                        labels = Source_test.labels
                        images = Source_test.drop(['labels'], axis= 1)
                        labels = np.array(labels, dtype=np.int32)
                        images = np.array(images, dtype=np.float32).reshape(-1, 1, 9)#######32
                        self.images = images
                        self.labels = labels
                


        
        def reset_pointer(self):
                self.pointer=0
                if self.shuffle:
                        self.shuffle_data()     

        def class_next_batch(self,num_per_class):
                batch_size=10*num_per_class
                classpaths=[]
                ids=[]
                for i in range(10):
                        classpaths.append([])
                for j in range(len(self.labels)):
                        label=self.labels[j]
                        classpaths[label].append(j)
                for i in range(10):
                        ids+=np.random.choice(classpaths[i],size=num_per_class,replace=False).tolist()
                selfimages=np.array(self.images)
                selflabels=np.array(self.labels)
                return np.array(selfimages[ids]),get_one_hot(selflabels[ids],self.num_classes)

        def next_batch(self,batch_size):
                if self.pointer+batch_size>len(self.labels):
                        self.reset_pointer()
                images=self.images[self.pointer:(self.pointer+batch_size)]
                labels=self.labels[self.pointer:(self.pointer+batch_size)]
                self.pointer+=batch_size
                return np.array(images),get_one_hot(labels,self.num_classes)  
        

def main():
        mnist=USPS(path='data/usps')
        print(mnist.images.max(), mnist.images.min(), mnist.images.shape)
        a,b=mnist.next_batch(1)
        

if __name__=='__main__':
        main()
