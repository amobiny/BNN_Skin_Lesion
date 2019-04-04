import tensorflow as tf
from tqdm import tqdm
from utils.augmentation_utils import tf_aug
from utils.eval_utils import save_confusion_matrix
from utils.loss_utils import cross_entropy, weighted_cross_entropy
import os
import numpy as np


class BaseModel(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.input_shape = [None, self.conf.height, self.conf.width, self.conf.channel]
        self.output_shape = [None, self.conf.num_cls]
        self.create_placeholders()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.inputs_pl = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.labels_pl = tf.placeholder(tf.float32, self.output_shape, name='annotation')
            self.keep_prob_pl = tf.placeholder(tf.float32)
            self.is_train_pl = tf.placeholder_with_default(True, shape=(), name="is_train")  # for augmentation
            self.inputs_aug = tf.cond(self.is_train_pl,
                                      lambda: tf.map_fn(lambda img: tf_aug(img), self.inputs_pl),
                                      lambda: self.inputs_pl)

    def loss_func(self):
        with tf.name_scope('Loss'):
            self.y_prob = tf.nn.softmax(self.logits, axis=-1)
            with tf.name_scope('cross_entropy'):
                if self.conf.weighted_loss:
                    loss = weighted_cross_entropy(self.labels_pl, self.logits, self.conf.num_cls)
                else:
                    loss = cross_entropy(self.labels_pl, self.logits)
            with tf.name_scope('total'):
                if self.conf.use_reg:
                    with tf.name_scope('L2_loss'):
                        l2_loss = tf.reduce_sum(
                            self.conf.lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))
                        self.total_loss = loss + l2_loss
                else:
                    self.total_loss = loss
                self.mean_loss, self.mean_loss_op = tf.metrics.mean(self.total_loss)

    def accuracy_func(self):
        with tf.name_scope('Accuracy'):
            self.y = tf.argmax(self.labels_pl, axis=1)
            self.y_pred = tf.argmax(self.logits, axis=1, name='y_pred')
            correct_prediction = tf.equal(self.y, self.y_pred, name='correct_pred')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')
            self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean(accuracy)

    def configure_network(self):
        self.loss_func()
        self.accuracy_func()
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                   global_step,
                                                   decay_steps=2000,
                                                   decay_rate=0.97,
                                                   staircase=True)
        self.learning_rate = tf.maximum(learning_rate, self.conf.lr_min)
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.total_loss, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=1000)
        self.train_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/train/', self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/valid/')
        self.configure_summary()
        print('*' * 50)
        print('Total number of trainable parameters: {}'.
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        print('*' * 50)

    def configure_summary(self):
        summary_list = [tf.summary.scalar('learning_rate', self.learning_rate),
                        tf.summary.scalar('loss', self.mean_loss),
                        tf.summary.scalar('accuracy', self.mean_accuracy)]
        self.merged_summary = tf.summary.merge(summary_list)

    def save_summary(self, summary, step, mode):
        # print('----> Summarizing at step {}'.format(step))
        if mode == 'train':
            self.train_writer.add_summary(summary, step)
        elif mode == 'valid':
            self.valid_writer.add_summary(summary, step)
        self.sess.run(tf.local_variables_initializer())

    def train(self):
        self.sess.run(tf.local_variables_initializer())
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
            print('----> Continue Training from step #{}'.format(self.conf.reload_step))
            self.best_validation_loss = input('Enter the approximate best validation loss you got last time')
            self.best_accuracy = input('Enter the approximate best validation accuracy (in range [0, 1])')
        else:
            self.best_validation_loss = 100
            self.best_accuracy = 0
            print('----> Start Training')
        if self.conf.data == 'mnist':
            from DataLoaders.mnist_loader import DataLoader
        elif self.conf.data == 'cifar':
            from DataLoaders.cifar_loader import DataLoader
        elif self.conf.data == 'skin':
            from DataLoaders.skin_lesion_loader import DataLoader
        else:
            print('wrong data name')
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_data(mode='train')
        self.data_reader.get_data(mode='valid')
        self.num_train_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode='train')
        self.num_val_batch = self.data_reader.count_num_batch(self.conf.val_batch_size, mode='valid')
        for epoch in range(self.conf.max_epoch):
            self.data_reader.randomize()
            for train_step in range(self.num_train_batch):
                glob_step = epoch * self.num_train_batch + train_step
                start = train_step * self.conf.batch_size
                end = (train_step + 1) * self.conf.batch_size
                x_batch, y_batch = self.data_reader.next_batch(start, end, mode='train')
                feed_dict = {self.inputs_pl: x_batch, self.labels_pl: y_batch, self.keep_prob_pl: self.conf.keep_prob}
                if train_step % self.conf.SUMMARY_FREQ == 0 and train_step != 0:
                    _, _, _, summary = self.sess.run([self.train_op,
                                                      self.mean_loss_op,
                                                      self.mean_accuracy_op,
                                                      self.merged_summary], feed_dict=feed_dict)
                    loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                    self.save_summary(summary, glob_step + self.conf.reload_step, mode='train')
                    print('epoch {0}/{1}, step: {2:<6}, train_loss= {3:.4f}, train_acc={4:.02%}'.
                          format(epoch, self.conf.max_epoch, glob_step, loss, acc))
                else:
                    self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
                if train_step % self.conf.VAL_FREQ == 0 and train_step != 0:
                    self.evaluate(train_step=glob_step, dataset='valid')

    def test(self, step_num):
        self.sess.run(tf.local_variables_initializer())
        print('loading the model.......')
        self.reload(step_num)
        if self.conf.data == 'ct':
            from DataLoaders.mnist_loader import DataLoader
        elif self.conf.data == 'cifar':
            from DataLoaders.cifar_loader import DataLoader
        elif self.conf.data == 'skin':
            from DataLoaders.skin_lesion_loader import DataLoader
        else:
            print('wrong data name')

        self.data_reader = DataLoader(self.conf)
        self.numTest = self.data_reader.count_num_samples(mode='test')
        self.num_test_batch = int(self.numTest / self.conf.val_batch_size)

        print('-' * 25 + 'Test' + '-' * 25)
        if not self.conf.bayes:
            self.evaluate(dataset='test', train_step=step_num)
        else:
            self.MC_evaluate(dataset='test', train_step=step_num)

    def save(self, step):
        print('----> Saving the model at step #{0}'.format(step))
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the model...')
        self.saver.restore(self.sess, model_path)
        print('----> Model successfully restored')

    def evaluate(self, dataset='valid', train_step=None):
        num_batch = self.num_test_batch if dataset == 'test' else self.num_val_batch
        self.sess.run(tf.local_variables_initializer())
        y_true, y_pred = np.zeros(num_batch*self.conf.val_batch_size), np.zeros(num_batch*self.conf.val_batch_size)
        for step in range(num_batch):
            start = self.conf.val_batch_size * step
            end = self.conf.val_batch_size * (step + 1)
            data_x, data_y = self.data_reader.next_batch(start=start, end=end, mode=dataset)
            feed_dict = {self.inputs_pl: data_x,
                         self.labels_pl: data_y,
                         self.keep_prob_pl: 1,
                         self.is_train_pl: False}
            y_true[start:end], y_pred[start:end], _, _ = self.sess.run([self.y, self.y_pred,
                                                                        self.mean_loss_op, self.mean_accuracy_op],
                                                                       feed_dict=feed_dict)

        loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        if dataset == "valid":  # save the summaries and improved model in validation mode
            print('-' * 30)
            print('valid_loss = {0:.4f}, val_acc = {1:.02%}'.format(loss, acc))
            summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
            self.save_summary(summary_valid, train_step, mode='valid')
            if loss < self.best_validation_loss:
                self.best_validation_loss = loss
                if acc > self.best_accuracy:
                    self.best_accuracy = acc
                    print('>>>>>>>> Both model validation loss and accuracy improved; saving the model......')
                else:
                    print('>>>>>>>> model validation loss improved; saving the model......')
                self.save(train_step)
            elif acc > self.best_accuracy:
                self.best_accuracy = acc
                print('>>>>>>>> model accuracy improved; saving the model......')
                self.save(train_step)
            print('-' * 30)
            fig_path = self.conf.imagedir + self.conf.run_name + '/' + str(train_step) + '.png'
        else:
            fig_path = self.conf.imagedir + self.conf.run_name + '/test_' + str(train_step) + '.png'
        save_confusion_matrix(y_true.astype(int), y_pred.astype(int),
                              classes=np.array(self.conf.label_name),
                              dest_path=fig_path,
                              title='Confusion matrix, without normalization')

    def MC_evaluate(self, dataset='valid', train_step=None):
        num_batch = self.num_test_batch if dataset == 'test' else self.num_val_batch
        self.sess.run(tf.local_variables_initializer())
        y_pred = np.zeros((self.conf.monte_carlo_simulations, self.conf.val_batch_size, self.conf.num_cls))
        for step in tqdm(range(num_batch)):
            start = self.conf.val_batch_size * step
            end = self.conf.val_batch_size * (step + 1)
            data_x, data_y = self.data_reader.next_batch(start=start, end=end, mode=dataset)
            feed_dict = {self.inputs_pl: data_x,
                         self.labels_pl: data_y,
                         self.keep_prob_pl: 1,
                         self.is_train_pl: False}
            for sample_id in range(self.conf.monte_carlo_simulations):
                # save predictions from a sample pass
                y_pred[sample_id] = self.sess.run(y_prob, feed_dict=feed_dict)

            # average and variance over all passes
            mean_pred[start:end] = y_pred.mean(axis=0)
            var_pred[start: end] = predictive_entropy(mean_pred[start:end])

            # compute batch error
            err += np.count_nonzero(np.not_equal(mean_pred[start:end].argmax(axis=1),
                                                 y_batch.argmax(axis=1)))

            mask_pred_mc = [np.zeros((self.conf.val_batch_size, self.conf.height, self.conf.width))
                            for _ in range(self.conf.monte_carlo_simulations)]
            mask_prob_mc = [np.zeros((self.conf.val_batch_size, self.conf.height, self.conf.width, self.conf.num_cls))
                            for _ in range(self.conf.monte_carlo_simulations)]
            feed_dict = {self.inputs_pl: data_x,
                         self.labels_pl: data_y,
                         self.is_training_pl: True,
                         self.with_dropout_pl: True,
                         self.keep_prob_pl: self.conf.keep_prob}
            for mc_iter in range(self.conf.monte_carlo_simulations):
                inputs, mask, mask_prob, mask_pred = self.sess.run([self.inputs_pl,
                                                                    self.labels_pl,
                                                                    self.y_prob,
                                                                    self.y_pred], feed_dict=feed_dict)
                mask_prob_mc[mc_iter] = mask_prob
                mask_pred_mc[mc_iter] = mask_pred
