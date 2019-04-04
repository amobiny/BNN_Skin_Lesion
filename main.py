import os
from config import args
import tensorflow as tf
if args.model == 'lenet':
    from models.lenet import LeNet as Model
elif args.model == 'resnet':
    from models.resnet import ResNet as Model
elif args.model == 'densenet':
    from models.densenet import DenseNet as Model

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1, 3'
    model = Model(tf.Session(), args)
    if not os.path.exists(args.modeldir + args.run_name):
        os.makedirs(args.modeldir + args.run_name)
    if not os.path.exists(args.logdir + args.run_name):
        os.makedirs(args.logdir + args.run_name)
    if not os.path.exists(args.imagedir + args.run_name):
        os.makedirs(args.imagedir + args.run_name)
    if args.mode == 'train':
        model.train()
    elif args.mode == 'test':
        model.test(step_num=args.reload_step)
