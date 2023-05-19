"""
Derived from: https://github.com/kratzert/finetune_alexnet_with_tensorflow/
"""
import tensorflow as tf
import numpy as np
import math
import sys
import tf_slim
from tf_slim.layers import layers as flc
#from tensorflow.contrib.slim import fully_connected as flc
sys.path.append('optimizers')

class LeNetModel(object):

    def __init__(self, num_classes=2, is_training=True,image_size=9,dropout_keep_prob=0.5):#######2/32
      self.num_classes = num_classes
      self.dropout_keep_prob = dropout_keep_prob
      self.default_image_size=image_size
      self.is_training=is_training
      self.num_channels=1
      self.mean=None
      self.bgr=False
      self.range=None
      self.featurelen=num_classes
      self.source_moving_centroid=tf.compat.v1.get_variable(name='source_moving_centroid',shape=[num_classes,self.featurelen],initializer=tf.zeros_initializer(),trainable=False)
      self.target_moving_centroid=tf.compat.v1.get_variable(name='target_moving_centroid',shape=[num_classes,self.featurelen],initializer=tf.zeros_initializer(),trainable=False)

      tf.compat.v1.summary.histogram('source_moving_centroid',self.source_moving_centroid)
      tf.compat.v1.summary.histogram('target_moving_centroid',self.target_moving_centroid)

    def g(self, x, training=False):
      # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
      conv1 = conv(x, 5, 5, 20, 1, 1, padding='VALID',bn=True,name='conv1', training=training)
      #pool1 = max_pool(conv1, 2, 2, 2, 2, padding='VALID',name='pool1')

      # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
      conv2 = conv(conv1, 5, 5, 50, 1, 1, padding='VALID',bn=True,name='conv2', training=training)
      #pool2 = max_pool(conv2, 2, 2, 2, 2, padding='VALID', name ='pool2')
      

      # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
      flattened = flc.flatten(conv2)
      fc1 = fc(flattened, 50, 100, bn=False, training=training, name='fc1')
      fc2 = fc(fc1, 100, self.num_classes, relu=False,bn=False, training=training, name='fc2')
      return fc2

    def f(self, fc2, training=False):
      # fc1 = fc(flattened, 800, 500, bn=False, training=training, name='fc1')
      # fc2 = fc(fc1, 500, 10, relu=False,bn=False, training=training, name='fc2')
      
      return fc2

    def inference(self, x, training=False):
      
      # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
      conv1 = conv(x, 5, 5, 20, 1, 1, padding='VALID',bn=True,name='conv1')
      #pool1 = max_pool(conv1, 2, 2, 2, 2, padding='VALID',name='pool1')

      # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
      conv2 = conv(conv1, 5, 5, 50, 1, 1, padding='VALID',bn=True,name='conv2')
      #pool2 = max_pool(conv2, 2, 2, 2, 2, padding='VALID', name ='pool2')

      
      # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
      flattened = flc.flatten(conv2)
      self.flattened=flattened
      fc1 = fc(flattened, 800, 500, bn=False,name='fc1')
      fc2 = fc(fc1, 500, self.num_classes, relu=False,bn=False,name='fc2')
      self.fc1=fc1
      self.fc2=fc2
      self.score=fc2
      self.output=tf.compat.v1.nn.softmax(self.score)
      self.feature=fc2
      
      return self.score


    def adoptimize(self,learning_rate,train_layers=[]):
        var_list=[v for v in tf.compat.v1.trainable_variables() if 'D' in v.name]
        D_weights=[v for v in var_list if 'weights' in v.name]
        D_biases=[v for v in var_list if 'biases' in v.name]
        print('=================Discriminator_weights=====================')
        print(D_weights)
        print('=================Discriminator_biases=====================')
        print(D_biases)

        self.Dregloss=5e-4*tf.compat.v1.reduce_mean([tf.compat.v1.nn.l2_loss(v) for v in var_list if 'weights' in v.name])
        D_op1 = tf.compat.v1.train.MomentumOptimizer(learning_rate,0.9).minimize(self.D_loss+self.Dregloss, var_list=D_weights)
        D_op2 = tf.compat.v1.train.MomentumOptimizer(learning_rate*2.0,0.9).minimize(self.D_loss+self.Dregloss, var_list=D_biases)
        D_op=tf.compat.v1.group(D_op1,D_op2)
        
        return D_op

    def wganloss(self,x,xt,batch_size,lam=10.0):
        with tf.compat.v1.variable_scope('reuse_inference') as scope:
          scope.reuse_variables()
          self.inference(x,training=True)
          source_fc6=self.fc6
          source_fc7=self.fc7
          source_fc8=self.fc8
          source_softmax=self.output
          source_output=outer(source_fc7,source_softmax)
          print('SOURCE_OUTPUT: ',source_output.get_shape())
          scope.reuse_variables()
          self.inference(xt,training=True)
          target_fc6=self.fc6
          target_fc7=self.fc7
          target_fc8=self.fc8
          target_softmax=self.output
          target_output=outer(target_fc7,target_softmax)
          print('TARGET_OUTPUT: ',target_output.get_shape())
        with tf.compat.v1.variable_scope('reuse') as scope:
          target_logits,_=D(target_fc8)
          scope.reuse_variables()
          source_logits,_=D(source_fc8)
          eps=tf.compat.v1.random_uniform([batch_size,1],minval=0.0,maxval=1.0)
          X_inter=eps*source_fc8+(1-eps)*target_fc8
          grad = tf.compat.v1.gradients(D(X_inter), [X_inter])[0]
          grad_norm = tf.compat.v1.sqrt(tf.compat.v1.reduce_sum((grad)**2, axis=1))
          grad_pen = lam * tf.compat.v1.reduce_mean((grad_norm - 1)**2)
          D_loss=tf.compat.v1.reduce_mean(target_logits)-tf.compat.v1.reduce_mean(source_logits)+grad_pen
          G_loss=tf.compat.v1.reduce_mean(source_logits)-tf.compat.v1.reduce_mean(target_logits)
          self.G_loss=G_loss
          self.D_loss=D_loss
          self.D_loss=0.3*self.D_loss
          self.G_loss=0.3*self.G_loss
          
          return G_loss,D_loss
    def adloss(self,x,xt,y,yt,LAMBDA,weightlamb):

      with tf.compat.v1.variable_scope('reuse_inference') as scope:
        scope.reuse_variables()
        target_feature = self.g(xt,training=True)
        self.feature = target_feature
      with tf.compat.v1.variable_scope('reuse_inference') as scope:
        scope.reuse_variables()
        target_feature_2 = self.g(xt,training=True)

      with tf.compat.v1.variable_scope('reuse_inference') as scope:
        scope.reuse_variables()
        source_feature = self.g(x,training=True)
        self.source_feature = source_feature

        scope.reuse_variables()
        target_pred = self.f(target_feature_2, training=True)
        target_pred_onehot = tf.compat.v1.one_hot(tf.compat.v1.argmax(target_pred, 1), self.num_classes)
        self.chose_rate = tf.compat.v1.constant(1.)

      with tf.compat.v1.variable_scope('reuse') as scope:
        source_logits,_=D(source_feature)
        scope.reuse_variables()
        target_logits,_=D(target_feature)
      
      graph_source = tf.compat.v1.reduce_sum(y[:, None, :] * y[None, :, :], 2)
      distance_source = tf.compat.v1.reduce_mean((source_feature[:, None, :] - source_feature[None, :, :])**2, 2)
      self.source_sntg_loss = tf.compat.v1.reduce_mean(graph_source * distance_source + (1-graph_source)*tf.nn.relu(LAMBDA- distance_source))
      
      source_result=tf.compat.v1.argmax(y,1)
      target_result=tf.compat.v1.argmax(target_pred,1)
      current_source_count=tf.compat.v1.unsorted_segment_sum(tf.compat.v1.ones_like(source_result, dtype=tf.compat.v1.float32),source_result,self.num_classes)
      current_target_count=tf.compat.v1.unsorted_segment_sum(tf.compat.v1.ones_like(target_result, dtype=tf.compat.v1.float32),target_result,self.num_classes)

      current_positive_source_count=tf.compat.v1.maximum(current_source_count,tf.compat.v1.ones_like(current_source_count))
      current_positive_target_count=tf.compat.v1.maximum(current_target_count,tf.compat.v1.ones_like(current_target_count))

      current_source_centroid=tf.compat.v1.divide(tf.compat.v1.unsorted_segment_sum(data=source_feature,segment_ids=source_result,num_segments=self.num_classes),current_positive_source_count[:, None])
      current_target_centroid=tf.compat.v1.divide(tf.compat.v1.unsorted_segment_sum(data=target_feature,segment_ids=target_result,num_segments=self.num_classes),current_positive_target_count[:, None])

      fm_mask = tf.compat.v1.to_float(tf.compat.v1.greater(current_source_count * current_target_count, 0))
      fm_mask /= tf.compat.v1.reduce_mean(fm_mask+1e-8)

      graph_target = tf.compat.v1.reduce_sum(target_pred_onehot[:, None, :] * target_pred_onehot[None, :, :], 2)
      distance_target = tf.compat.v1.reduce_mean((target_feature[:, None, :] - target_feature[None, :, :])**2, 2)
      self.target_sntg_loss = tf.compat.v1.reduce_mean(graph_target * distance_target + (1-graph_target)*tf.compat.v1.nn.relu(LAMBDA- distance_target))

      self.sntg_loss = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_mean(tf.compat.v1.square(current_source_centroid-current_target_centroid),1)*fm_mask) + self.target_sntg_loss + self.source_sntg_loss

      D_real_loss=tf.compat.v1.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=target_logits,labels=tf.compat.v1.ones_like(target_logits)))
      D_fake_loss=tf.compat.v1.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=source_logits,labels=tf.compat.v1.zeros_like(source_logits)))
      self.D_loss=D_real_loss+D_fake_loss

      self.G_loss=-self.D_loss
      tf.summary.scalar('JSD',self.G_loss/2+math.log(2))

      #------------- Domain Adversarial Loss is scaled by 0.1 following RevGrad--------------------------
      self.G_loss=0.1*self.G_loss
      self.D_loss=0.1*self.D_loss
      
      return self.G_loss,self.D_loss
    def loss(self, batch_x, batch_y=None):
      with tf.compat.v1.variable_scope('reuse_inference') as scope:
        y_predict = self.f(self.g(batch_x, training=True), training=True)
      self.loss = tf.compat.v1.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y))
      
      return y_predict,self.loss

    def optimize(self, learning_rate, train_layers,adlamb,sntglamb,unbalance,revgrad=0):
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      print(train_layers)
      var_list=[v for v in tf.compat.v1.trainable_variables() if v.name.split('/')[1] in ['conv1','conv2','fc1','fc2']]
      self.Gregloss=5e-4*tf.compat.v1.reduce_mean([tf.compat.v1.nn.l2_loss(x) for x in var_list if 'weights' in x.name])

      new_weights=[v for v in var_list if 'weights' in v.name or 'gamma' in v.name]
      new_biases=[v for v in var_list if 'biases' in v.name or 'beta' in v.name]


      print('==============new_weights=======================')
      print(new_weights)
      print('==============new_biases=======================')
      print(new_biases)

      if unbalance == 1.:
              self.F_loss=self.loss+self.Gregloss+sntglamb*self.sntg_loss
              if revgrad != 0:
                      self.F_loss += adlamb*self.G_loss
      else:
              self.F_loss=self.loss+self.Gregloss+sntglamb*self.sntg_loss
      update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
      print('+++++++++++++++ batch norm update ops +++++++++++++++++')
      print(update_ops)
      with tf.compat.v1.control_dependencies(update_ops):
          train_op3=tf.compat.v1.train.MomentumOptimizer(learning_rate*1.0,0.9).minimize(self.F_loss, var_list=new_weights)
          train_op4=tf.compat.v1.train.MomentumOptimizer(learning_rate*2.0,0.9).minimize(self.F_loss, var_list=new_biases)
      train_op=tf.compat.v1.group(train_op3,train_op4)
      return train_op

    def load_original_weights(self, session, skip_layers=[]):
        weights_dict = np.load('bvlc_alexnet.npy', encoding='bytes').item()
        
        for op_name in weights_dict:
            # if op_name in skip_layers:
            #     continue

            if op_name == 'fc8' and self.num_classes != 1000:
                continue

            with tf.compat.v1.variable_scope('reuse_inference/'+op_name, reuse=True):
                print('=============================OP_NAME  ========================================')
                for data in weights_dict[op_name]:
                    if len(data.shape) == 1:
                        var = tf.compat.v1.get_variable('biases')
                        print(op_name,var)
                        session.run(var.assign(data))
                    else:
                        var = tf.compat.v1.get_variable('weights')
                        print(op_name,var)
                        session.run(var.assign(data))


"""
Helper methods
"""
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, training, bn=False,padding='SAME', groups=1):
    input_channels = int(x.get_shape()[-1])
    #print(input_channels)
    convolve = lambda i, k: tf.compat.v1.nn.conv1d(i, k, padding=padding)

    with tf.compat.v1.variable_scope(name) as scope:
        weights = tf.compat.v1.get_variable('weights', shape=[1, int(input_channels/groups), num_filters],initializer=tf_slim.layers.initializers.xavier_initializer())
        biases = tf.compat.v1.get_variable('biases', shape=[num_filters])
        #print(biases.shape)
        if groups == 1:
            conv = convolve(x, weights)
        else:
            #print('kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
            input_groups = tf.compat.v1.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.compat.v1.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
            conv = tf.compat.v1.concat(axis=3, values=output_groups)

        bias = tf.compat.v1.reshape(tf.compat.v1.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
        if bn==True:
            bias=flc.batch_norm(bias,scale=True, is_training=training)
        relu = tf.compat.v1.nn.relu(bias, name=scope.name)
        return relu

def D(x):
    with tf.compat.v1.variable_scope('D'):
      num_units_in=int(x.get_shape()[-1])
      num_units_out=1
      n=500
      weights = tf.compat.v1.get_variable('weights',shape=[num_units_in,n],initializer=tf_slim.layers.initializers.xavier_initializer())
      biases = tf.compat.v1.get_variable('biases', shape=[n], initializer=tf.compat.v1.zeros_initializer())
      hx=(tf.compat.v1.matmul(x,weights)+biases)
      ax=tf.compat.v1.nn.relu(hx)

      weights2 = tf.compat.v1.get_variable('weights2',shape=[n,n],initializer=tf_slim.layers.initializers.xavier_initializer())
      biases2 = tf.compat.v1.get_variable('biases2', shape=[n], initializer=tf.compat.v1.zeros_initializer())
      hx2=(tf.compat.v1.matmul(ax,weights2)+biases2)
      ax2=tf.nn.relu(hx2)
      weights3 = tf.compat.v1.get_variable('weights3',shape=[n,num_units_out],initializer=tf_slim.layers.initializers.xavier_initializer())
      biases3 = tf.compat.v1.get_variable('biases3', shape=[num_units_out], initializer=tf.compat.v1.zeros_initializer())
      hx3=tf.compat.v1.matmul(ax2,weights3)+biases3
      
      return hx3,tf.nn.sigmoid(hx3)

def fc(x, num_in, num_out, name, training,relu=True,bn=False,stddev=0.001):
    with tf.compat.v1.variable_scope(name) as scope:
        weights = tf.compat.v1.get_variable('weights', shape=[num_in,num_out],initializer=tf_slim.layers.initializers.xavier_initializer())
        biases = tf.compat.v1.get_variable('biases',initializer=tf.compat.v1.constant(0.1,shape=[num_out]))
        act = tf.compat.v1.nn.xw_plus_b(x, weights, biases, name=scope.name)
        if bn==True:
            act=flc.batch_norm(act,scale=True, is_training=training)
        if relu == True:
            relu = tf.compat.v1.nn.relu(act)
            
            return relu
        else:
            
            return act
def leaky_relu(x, alpha=0.2):
    return tf.compat.v1.maximum(tf.minimum(0.0, alpha * x), x)

def outer(a,b):
        #print('YESSSSSSSSSSSSSSSSSSSSSSSSSSSS')
        a=tf.compat.v1.reshape(a,[-1,a.get_shape()[-1],1])
        b=tf.compat.v1.reshape(b,[-1,1,b.get_shape()[-1]])
        c=a*b
        #print('YESSSSSSSSSSSSSSSSSSSSSSSSSSSS')
        return flc.flatten(c)

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.compat.v1.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides = [1, stride_y, stride_x, 1],
                          padding = padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.compat.v1.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

def dropout(x, keep_prob):
    return tf.compat.v1.nn.dropout(x, keep_prob)
