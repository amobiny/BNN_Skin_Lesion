import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('mode', 'test', 'train or test')
flags.DEFINE_string('model', 'resnet', 'lenet, resnet, densenet')
flags.DEFINE_boolean('bayes', True, 'Whether to use Bayesian network or not')
flags.DEFINE_integer('monte_carlo_simulations', 50, 'The number of monte carlo simulation runs')
flags.DEFINE_integer('reload_step', 18923, 'Reload step to continue training')
flags.DEFINE_boolean('weighted_loss', True, 'Whether to use weighted loss or not')

# Training logs
flags.DEFINE_integer('max_epoch', 1000, 'maximum number of training epochs')
flags.DEFINE_integer('max_step', 30000000, '# of step for training')
flags.DEFINE_integer('SUMMARY_FREQ', 100, 'Number of step to save summary')
flags.DEFINE_integer('VAL_FREQ', 500, 'Number of step to evaluate the network on Validation data')
flags.DEFINE_float('init_lr', 1e-3, 'Initial learning rate')
flags.DEFINE_float('lr_min', 1e-5, 'Minimum learning rate')

# Hyper-parameters
flags.DEFINE_integer('batch_size', 10, 'training batch size')
flags.DEFINE_integer('val_batch_size', 5, 'validation batch size')
flags.DEFINE_float('lmbda', 1e-4, 'L2 regularization coefficient')
flags.DEFINE_float('keep_prob', 0.7, 'keep prob of the dropout')
flags.DEFINE_boolean('use_reg', True, 'Use L2 regularization on weights')

# data
flags.DEFINE_string('data', 'skin', 'mnist or skin')
flags.DEFINE_boolean('data_augment', True, 'whether to apply data augmentation or not')
flags.DEFINE_integer('max_angle', 360, 'maximum rotation angle')
flags.DEFINE_integer('height', 224, 'input image height size')
flags.DEFINE_integer('width', 224, 'input image width size')
flags.DEFINE_integer('channel', 3, 'input image channel size')
flags.DEFINE_integer('num_cls', 7, 'Number of output classes')
flags.DEFINE_list('label_name', ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'], 'class names')   #####

# Directories
flags.DEFINE_string('run_name', 'run1_resnet', 'Run name')
flags.DEFINE_string('logdir', './Results/log_dir/', 'Logs directory')
flags.DEFINE_string('modeldir', './Results/model_dir/', 'Model directory')
flags.DEFINE_string('imagedir', './Results/image_dir/', 'Directory to save sample predictions')
flags.DEFINE_string('model_name', 'model', 'Model file name')

# network structure
flags.DEFINE_integer('growth_rate', 24, 'Growth rate of DenseNet')
flags.DEFINE_integer('num_levels', 3, '# of levels (dense block + Transition Layer) in DenseNet')
flags.DEFINE_list('num_BBs', [6, 8, 10], '# of bottleneck-blocks at each level')
flags.DEFINE_float('theta', 1, 'Compression factor in DenseNet')

args = tf.app.flags.FLAGS
