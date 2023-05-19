import os, sys, time

import numpy as np
import tensorflow as tf
import datetime
from model import LeNetModel
from mnist import MNIST
from svhn import SVHN
from usps import USPS
from preprocessing import preprocessing
tf.compat.v1.disable_eager_execution()


import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

import math
#from tensorflow.contrib.tensorboard.plugins import projector

tf.compat.v1.app.flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate for adam optimizer')
tf.compat.v1.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.compat.v1.app.flags.DEFINE_float('LAMBDA', 30, 'LAMBDA') # 30 for svhn->mnist
tf.compat.v1.app.flags.DEFINE_integer('num_epochs', 10, 'Number of epochs for training')
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.compat.v1.app.flags.DEFINE_integer('seed', 1, 'Seed')
tf.compat.v1.app.flags.DEFINE_string('train_layers', 'fc8,fc7,fc6,conv5,conv4,conv3,conv2,conv1', 'Finetuning layers, seperated by commas')
tf.compat.v1.app.flags.DEFINE_string('multi_scale', '256,257', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at networs input size')
tf.compat.v1.app.flags.DEFINE_string('train_root_dir', '/content/drive/MyDrive/CAT/data', 'Root directory to put the training data')
tf.compat.v1.app.flags.DEFINE_string('source', 'svhn', 'source')
tf.compat.v1.app.flags.DEFINE_string('target', 'mnist', 'target')
tf.compat.v1.app.flags.DEFINE_string('tag', 'cat', 'tag')
tf.compat.v1.app.flags.DEFINE_integer('log_step', 10000, 'Logging period in terms of iteration')
tf.compat.v1.app.flags.DEFINE_float('unbalance', 1., 'Unbalance ratio') #set 10 for the imbalanced experiments
tf.compat.v1.app.flags.DEFINE_integer('revgrad', 0, 'Use revgrad or not')

FLAGS = tf.compat.v1.app.flags.FLAGS
MAX_STEP=10000
NUM_CLASSES = 2
data_map = {"mnist": MNIST, "svhn": SVHN, "usps": USPS}

select_source = []
select_target = []
rng = np.random.RandomState(seed=100 + FLAGS.seed)
if (FLAGS.source == "mnist" and FLAGS.target == "usps"):
        select_source = list(rng.permutation(60000)[:2000])
        select_target = list(rng.permutation(7291)[:1800])
        #print(select_source[:10], select_target[:10])
elif (FLAGS.source == "usps" and FLAGS.target == "mnist"):
        select_target = list(rng.permutation(60000)[:2000])
        select_source = list(rng.permutation(7291)[:1800])
        #print(select_source[:10], select_target[:10])

TRAIN=data_map[FLAGS.source]('data/' + FLAGS.source,split='train',shuffle=True, select=select_source, unbalance=FLAGS.unbalance)

VALID=data_map[FLAGS.target]('data/' + FLAGS.target,split='train',shuffle=True, select=select_target, unbalance=1./FLAGS.unbalance)

TEST=data_map[FLAGS.target]('data/' + FLAGS.target,split='test',shuffle=False, select=select_target, unbalance=1./FLAGS.unbalance)


def decay(start_rate,epoch,num_epochs):
        return start_rate/pow(1+0.001*epoch,0.75)

def adaptation_factor(x):
	den=1.0+math.exp(-10*x)
	lamb=2.0/den-1.0
	return min(lamb,1.0)
def scatter(data, label, dir, file_name, mus=None, mark_size=2):
    if label.ndim == 2:
        label = np.argmax(label, axis=1)

    df = pd.DataFrame(data={'x':data[:,0], 'y':data[:,1], 'class':label})
    sns_plot = sns.lmplot('x', 'y', data=df, hue='class', fit_reg=False, scatter_kws={'s':mark_size})
    sns_plot.savefig(os.path.join(dir, file_name))
    if mus is not None:
        df_mus = pd.DataFrame(data={'x':mus[:,0], 'y':mus[:,1], 'class':np.asarray(range(mus.shape[0])).astype(np.int32)})
        sns_plot_mus = sns.lmplot('x', 'y', data=df_mus, hue='class', fit_reg=False, scatter_kws={'s':mark_size*20})
        sns_plot_mus.savefig(os.path.join(dir, 'mus_'+file_name))

def main(_):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('alexnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.train_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.train_root_dir): os.mkdir(FLAGS.train_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('dropout_keep_prob={}\n'.format(FLAGS.dropout_keep_prob))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))
    flags_file.write('train_root_dir={}\n'.format(FLAGS.train_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()

    adlamb=tf.compat.v1.placeholder(tf.compat.v1.float32,name='adlamb')
    sntglamb=tf.compat.v1.placeholder(tf.compat.v1.float32,name='sntglamb')
    decay_learning_rate=tf.compat.v1.placeholder(tf.compat.v1.float32)
    dropout_keep_prob = tf.compat.v1.placeholder(tf.compat.v1.float32)
    is_training=tf.compat.v1.placeholder(tf.compat.v1.bool)
    weightlamb = tf.compat.v1.placeholder(tf.compat.v1.float32)

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = LeNetModel(num_classes=NUM_CLASSES, image_size=9,is_training=is_training,dropout_keep_prob=dropout_keep_prob)######32
    # Placeholders
    if FLAGS.source == "usps":
        x_s = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 1, 9],name='x')######32

    if FLAGS.target == "mnist":
        x_t = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 1, 9],name='xt')########32

    x=x_s
    
    xt=x_t
    #print(xt.shape)
    #tf.compat.v1.summary.image('Source Images',x)
    #tf.compat.v1.summary.image('Target Images',xt)

    y = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, NUM_CLASSES],name='y')
    yt = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, NUM_CLASSES],name='yt')
    y_predict, loss = model.loss(x, y)
    # Training accuracy of the model
    source_correct_pred = tf.compat.v1.equal(tf.compat.v1.argmax(y_predict, 1), tf.compat.v1.argmax(y, 1))
    source_correct=tf.compat.v1.reduce_sum(tf.compat.v1.cast(source_correct_pred,tf.float64))
    #print(source_correct)
    source_accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(source_correct_pred, tf.float64))

    G_loss,D_loss=model.adloss(x,xt,y,yt, FLAGS.LAMBDA,weightlamb)

    # Testing accuracy of the model
    with tf.compat.v1.variable_scope('reuse_inference') as scope:
        scope.reuse_variables()
        target_feature_test = model.g(xt,training=False)
    with tf.compat.v1.variable_scope('reuse_inference') as scope:
        scope.reuse_variables()
        target_pred_test = model.f(target_feature_test, training=False)
        correct_pred = tf.compat.v1.equal(tf.compat.v1.argmax(target_pred_test, 1), tf.compat.v1.argmax(yt, 1))
        correct = tf.compat.v1.reduce_sum(tf.compat.v1.cast(correct_pred, tf.float64))

        TP = tf.compat.v1.count_nonzero(tf.compat.v1.argmax(target_pred_test, 1) * tf.compat.v1.argmax(yt, 1))
        TN = tf.compat.v1.count_nonzero((tf.compat.v1.argmax(target_pred_test, 1) - 1) * (tf.compat.v1.argmax(yt, 1) - 1))
        FP = tf.compat.v1.count_nonzero(tf.compat.v1.argmax(target_pred_test, 1) * (tf.compat.v1.argmax(yt, 1) - 1))
        FN = tf.compat.v1.count_nonzero((tf.compat.v1.argmax(target_pred_test, 1) - 1) * tf.compat.v1.argmax(yt, 1))
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        
        accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_pred, tf.float64))


    update_op = model.optimize(decay_learning_rate,train_layers,adlamb,sntglamb,FLAGS.unbalance, FLAGS.revgrad)

    D_op=model.adoptimize(decay_learning_rate,train_layers)
    optimizer=tf.compat.v1.group(update_op,D_op)

    train_writer=tf.compat.v1.summary.FileWriter(tensorboard_dir)
    train_writer.add_graph(tf.compat.v1.get_default_graph())
    '''config=projector.ProjectorConfig()
    embedding=config.embeddings.add()
    embedding.tensor_name=model.feature.name
    embedding.metadata_path='domain.csv'
    projector.visualize_embeddings(train_writer,config)'''
    tf.compat.v1.summary.scalar('G_loss',model.G_loss)
    tf.compat.v1.summary.scalar('D_loss',model.D_loss)
    tf.compat.v1.summary.scalar('C_loss',model.loss)
    tf.compat.v1.summary.scalar('Training Accuracy',source_accuracy)
    tf.compat.v1.summary.scalar('Testing Accuracy',accuracy)
    merged=tf.compat.v1.summary.merge_all()

    print('============================GLOBAL TRAINABLE VARIABLES ============================')
    print(tf.compat.v1.trainable_variables())
    #print '============================GLOBAL VARIABLES ======================================'
    #print tf.global_variables()

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver=tf.compat.v1.train.Saver()
        train_writer.add_graph(sess.graph)

        print("{} Start training...".format(datetime.datetime.now()))
        gd=0
        step = 1
        max_acc = 0
        max_fscore = 0
        times = -time.time()
        error_list = []
        for epoch in range(40000):
            # Start training
            gd+=1
            lamb=adaptation_factor(gd*1.0/40000)
            if FLAGS.unbalance == 1.:
                lamb2=math.exp(-(1-min((gd - 5000)*1.0/10000, 1.))*10) if gd >= 5000 else 0.
                if FLAGS.source == "mnist" and FLAGS.target == "usps":
                        lamb2=math.exp(-(1-min((gd - 2000)*1.0/5000, 1.))*10) if gd >= 2000 else 0.
            else:
                lamb2=math.exp(-(1-min((gd - 2000)*1.0/2000, 1.))*10) if gd >= 2000 else 0.
            #rate=decay(FLAGS.learning_rate,gd,MAX_STEP)
            rate=FLAGS.learning_rate
            batch_xs, batch_ys = TRAIN.next_batch(FLAGS.batch_size)
            Tbatch_xs, Tbatch_ys = VALID.next_batch(FLAGS.batch_size)
            
            summary,_,closs,gloss,dloss,sn_loss, chose_rate_=sess.run([merged,optimizer,model.loss,model.G_loss,model.D_loss,model.sntg_loss, model.chose_rate], feed_dict={x_s: batch_xs,x_t: Tbatch_xs,decay_learning_rate:rate,adlamb:lamb,y: batch_ys,yt:Tbatch_ys,sntglamb:lamb2})
            #print('YESSSSSS')
            train_writer.add_summary(summary,gd)
            
            step += 1
            if gd%250==0:
                epoch=gd/(72357/100)
                print('Epoch {} time {}s Step {} lambda {:.4f} lamb2 {:.4f} rate {:.4f} C_loss {:.4f} G_loss {:.4f} D_loss {:.4f} SNTG_loss {:.4f} chose_rate {:.4f}'.format(epoch, times+time.time(), gd, lamb, lamb2, rate, closs,gloss,dloss,sn_loss, chose_rate_))

                test_acc = 0.
                test_count = 0
                F1score = 0.
                tt_embs = []
                tt_y = []
                for _ in range(int((len(TEST.labels))/100)):
                    batch_tx, batch_ty = TEST.next_batch(100)
                    #print TEST.pointer,'   ',TEST.shuffle
                    acc, t_embs = sess.run([correct, model.feature], feed_dict={x_t: batch_tx, yt: batch_ty})
                    f1score, t_embs = sess.run([f1, model.feature], feed_dict={x_t: batch_tx, yt: batch_ty})
                    
                    tt_embs.append(t_embs)
                    tt_y.append(batch_ty)
                    test_acc += acc
                    F1score += f1score
                    test_count += 100

                test_acc /= test_count
                F1score /= (test_count/100)
                max_acc = max(max_acc, test_acc)
                max_fscore = max(max_fscore, F1score)
                print("Validation Accuracy = {:.4f} Max_Accuracy = {:.4f}".format(test_acc, max_acc))
                print("Validation F1score = {:.4f} Max_f1score = {:.4f}".format(F1score, max_fscore))
                error_list.append(test_acc)

                if gd % 5000 == 0 and False:
                        tt_embs_s = []
                        tt_y_s = []
                        TRAIN.reset_pointer()
                        for _ in xrange((len(TRAIN.labels))/100):
                            batch_tx, batch_ty = TRAIN.next_batch(100)
                            #print TEST.pointer,'   ',TEST.shuffle
                            t_embs = sess.run(model.source_feature, feed_dict={x_s: batch_tx, y: batch_ty})
                            tt_embs_s.append(t_embs)
                            tt_y_s.append(batch_ty)

                        #np.savez("features_{}.npz".format(FLAGS.tag), x1=np.vstack(tt_embs_s), y1=np.argmax(np.vstack(tt_y_s), axis=1), x2=np.vstack(tt_embs), y2=np.argmax(np.vstack(tt_y), axis=1))
                        if True:
                                if NUM_CLASSES == 10:
                                        test_h_s = np.vstack(tt_embs_s)[:5000]
                                        y_test_s = np.ones((test_h_s.shape[0],))*8

                                        test_h = np.concatenate([np.vstack(tt_embs)[:5000], test_h_s], 0)
                                        y_test = np.argmax(np.vstack(tt_y)[:5000], axis=1)
                                        y_test[y_test == 8] = 10
                                        y_test = np.concatenate([y_test, y_test_s], 0)
                                        z_dev_2D = TSNE().fit_transform(test_h)
                                        '''scatter(data=z_dev_2D, label=y_test,
                                                          dir="./embedding",
                                                          file_name='s{}_{}_epoch{:03d}.png'.format(FLAGS.source, FLAGS.target, gd/5000))'''
                                else:
                                        test_h_s = np.vstack(tt_embs_s)
                                        y_test_s = np.argmax(np.vstack(tt_y_s), axis=1)
                                        y_test_s[y_test_s == 0] = 2
                                        y_test_s[y_test_s == 1] = 3

                                        test_h = np.concatenate([np.vstack(tt_embs), test_h_s], 0)
                                        y_test = np.argmax(np.vstack(tt_y), axis=1)
                                        y_test = np.concatenate([y_test, y_test_s], 0)
                                        z_dev_2D = TSNE().fit_transform(test_h)
                                        '''scatter(data=z_dev_2D, label=y_test,
                                                          dir="./embedding",
                                                          file_name='cattsne_manifold_epoch{:03d}.png'.format(gd/5000))'''

                if gd%5000==0 and gd>0:
                    print()#error_list

                times = -time.time()

            #save checkpoint of the model
            #checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch'+str(epoch+1)+'.ckpt')
            #save_path = saver.save(sess, checkpoint_path)

            #print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_path))

if __name__ == '__main__':
    tf.compat.v1.app.run()
