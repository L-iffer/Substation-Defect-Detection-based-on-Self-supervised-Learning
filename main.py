import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import simsiam.loader
import simsiam.builder_4
import simsiam.builder_5
import simsiam.builder
import simsiam.emd
import simsiam.averCosSim

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch COCO Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',  # 模型的backbone，默认为resnet50
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')  # 线程数，也就是num_workers，在3090上最好设置为8或者4
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')  # 训练的epochs
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')   # 起始的epoch
parser.add_argument('-b', '--batch-size', default=96, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')  # batch_size
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')  # 学习率
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')  # SGD优化器中的动量参数
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',  
                    dest='weight_decay')  # 权重衰减
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')  # 打印频率
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')  # 是否接着上次的训练，是的话给出保存模型的路径
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')  # 参与工作的进程数
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')  # 当前进程的rank
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')  # 后端选择，默认为nccl
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')   # 初始化训练的随机种子
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')  # 可供使用的gpu编号
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# parser.add_argument('--pretrain', default='./checkpoint_0300.pth.tar', type=str, metavar='PATH',
#                     help='the pretrained model we need load')  # 导入的预训练模型
# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')  # resnet50输出的特征维度
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')  # 预测器全连接层输出的特征维度
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')


def main():
    args = parser.parse_args()

    if args.seed is not None:  # 如果默认参数设置了随机种子数值,默认不设置随机种子 跳过
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:  # 如果默认参数给出了特定的gpu编号 跳过
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:  # 跳过
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed  # true

    ngpus_per_node = torch.cuda.device_count()  # 每个节点的gpu数量
    if args.multiprocessing_distributed:  # true
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size  # 每个节点的gpu数*进程数=总的进程数 在3090上是1
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))  # 在一个节点启动该节点所有进程
    else:  # 不使用多进程分布式训练的话，简单执行main_worker()函数即可
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:  # 默认为None,跳过
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:  # true
        if args.dist_url == "env://" and args.rank == -1:  # args.dist_url = 'tcp://' 跳过
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:  # true
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu  # 0
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)  # 初始化
        torch.distributed.barrier()  # 对其它非主进程进行阻塞，达到同步的目的
    # create model
    print("=> creating model '{}'".format(args.arch))  # 告诉用户开始创建模型了
    model = simsiam.builder_5.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim)  # 调用simsiam.builder.py文件中的SimSiam并实例化，传入的参数分别是backbone(resnet50)；resnet50最后全连接层的输入维度；预测器最后全连接层的输出维度

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256  # 初始学习率策略，batch_size越大，相应的学习率也会越大

    if args.distributed:  # 如果采取分布式训练 true
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 由于是分布式训练，将模型中的BN归一化形式转成同步BN归一化
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:  # 当gpu编号不为空
            torch.cuda.set_device(args.gpu)  # 指定gpu
            model.cuda(args.gpu)  # 模型设置gpu
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)  # 每个gpu的batch_size
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)  # 进程数
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm  # 打印模型信息

    # define loss function (criterion) and optimizer
    # criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)  # 定义相似度计算
    # criterion = simsiam.emd.emd_forward  # 定义EMD计算
    criterion = simsiam.averCosSim.averCosineSimilatiry  # 定义均值余弦相似度计算

    if args.fix_pred_lr:  # 固定预测器的学习率，因为作者发现固定预测器的参数效果最好
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters() # 定义模型的整个参数为需要优化的参数 

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)  # 定义优化器:SGD
    # optimizer = torch.optim.Adam(optim_params, init_lr)  # 定义Adam优化器

    # optionally resume from a checkpoint  接着上次的模型继续训练
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)  # 从保存模型的路径中导入上次训练的模型参数（包括model参数和optimizer参数）
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']  # 指定当前开始训练的epoch，就是上次结束的epoch
            model.load_state_dict(checkpoint['state_dict'])  # 将上次的backbone参数导入模型中
            optimizer.load_state_dict(checkpoint['optimizer'])  # 将上次的optimizer参数导入优化器中
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))  # 打印已经导入参数的信息
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))  # 没指定resume路径的话，直接打印信息说明没有checkpoint
    
    # 导入ImageNet的预训练模型
    # pretrained_dict = torch.load(args.pretrain)['state_dict']
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if ((k in model_dict) and (not k.startswith("module.predictor")))}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if ((not k.startswith("module.encoder.fc")) and (not k.startswith("module.predictor")))}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("module.predictor")}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # print("pretrained model loaded successfully")

    # model.load_state_dict(torch.load(args.pretrain)['state_dict'])
    # print("pretrained model loaded successfully")
    
    # 如何将A模型的前半部分参数+B模型的后半部分参数导入model
    pretrain_1_dict = torch.load("./final_checkpoint_0316.pth.tar")['state_dict']
    pretrain_2_dict = torch.load("./CSE_checkpoint_0119.pth.tar")['state_dict']
    optimizer_dict = torch.load("./CSE_checkpoint_0119.pth.tar")['optimizer']
    model_dict = model.state_dict()

    pretrained_1_dict = {k: v for k, v in pretrain_1_dict.items() if ((not k.startswith("module.encoder.fc")) and (not k.startswith("module.predictor")))}
    pretrained_2_dict = {k: v for k, v in pretrain_2_dict.items() if ((k.startswith("module.encoder.fc")) or (k.startswith("module.predictor")) or (k.startswith("module.encoder.sa")))}
    model_dict.update(pretrained_1_dict)
    model_dict.update(pretrained_2_dict)

    model.load_state_dict(model_dict)
    print("pretrained model loaded successfully")

    save_checkpoint({
        'epoch': 316,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer_dict,
    }, is_best=False, filename='Conv_SAM_Euclidean_0316.pth.tar')
    print("save model successfully")

    cudnn.benchmark = True  # 让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高校算法，来达到优化运行效率的问题。

    # Data loading code
    traindir = os.path.join(args.data, 'train')  # 保存训练集的路径
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # 归一化的参数：均值、方差

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709  # SimSiam的数据增强算法，和MoCo v2一样
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),  # 将图像随机裁剪成224*224大小，符合resnet50输入维度
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),  # 调整亮度、对比度、饱和度、色相
        transforms.RandomGrayscale(p=0.2),  # 以0.2的概率转成灰度图
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),  # 高斯模糊
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 将图片转成张量
        normalize  # 归一化
    ]
    # 返回参数有三种属性：self.classes用一个list保存类别名称；self.class_to_idx类别对应的索引，与不做任何转换返回的target对应；self.images保存(img-path,class)tuple的list
    train_dataset = datasets.ImageFolder(  # datasets.ImageFolder就是同样的数据加载器,traindir是训练集路径，加载到训练集图像后再进行图像增强
        traindir,
        simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))  # 返回的是query_crops和key_crops

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)  # 读取数据


    for epoch in range(args.start_epoch, args.epochs):  # 开始训练
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)  # 调整学习率

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)  # 训练当前epoch

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({  # 保存当前模型参数
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))
    

def train(train_loader, model, criterion, optimizer, epoch, args):  # 训练模块，参数分别为：训练数据集、模型、相似度计算函数、优化器、训练轮数、args参数
    batch_time = AverageMeter('Time', ':6.3f')  # 批次时间
    data_time = AverageMeter('Data', ':6.3f')  # 数据时间
    losses = AverageMeter('Loss', ':.4f')  # 损失
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()  # 模型设置为训练模式

    end = time.time()  
    for i, (images, _) in enumerate(train_loader):  # 遍历训练集
        # measure data loading time
        data_time.update(time.time() - end)  # 每读取一个批次的训练集时，都更新数据加载时间，data_time.count中会记录迄今为止一共读了个batch数据，date_time.count中记录每个batch平均读取时间

        if args.gpu is not None:  # 给训练集添加gpu设备
            images[0] = images[0].cuda(args.gpu, non_blocking=True)  # images[0]表示当前批次图片的第一种数据增强视图，在simsiam.loader.TwoCropsTransform中已经体现出来
            images[1] = images[1].cuda(args.gpu, non_blocking=True)  # images[1]表示当前批次图片的第二种数据增强视图

        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])  # 将每一批次中两种不同的视图分别输入到模型中得到p1,p2和z1,z2(no gradient)
        # loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5  # 计算损失,criterion()为相似度计算函数，分别计算每一批图片中的数据对应两幅视图的相似度，然后求这一批图片相似度的平均
        # loss = 2 * (1 - (criterion(F.normalize(p1, dim=1), F.normalize(z2, dim=1)).mean() + criterion(F.normalize(p2, dim=1), F.normalize(z1, dim=1)).mean()) * 0.5)  # 归一化后的欧氏距离的平方等于2*(1-余弦相似度)
        loss = 2 * (1 - (criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5)    

        losses.update(loss.item(), images[0].size(0))  # 更新一下当前为止所有的batch平均损失

        # compute gradient and do SGD step
        optimizer.zero_grad()  # 优化器更新梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 编码网络参数更新

        # measure elapsed time
        batch_time.update(time.time() - end)  # 更新当前为止所有批次的平均训练时间
        end = time.time()

        if i % args.print_freq == 0:  # 到达指定迭代次数打印相关信息
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value""" # 计算和存储当前值
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val  # 当前值
        self.sum += val * n  # 总值
        self.count += n  # 总值中含有多少个单个的值
        self.avg = self.sum / self.count  # 平均值：总值 / 单个值的个数

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):  # 按照设定的策略衰减学习率，余弦衰减策略
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))  # 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))的值在[1, 0]区间内，随着epoch的增加，值慢慢变小
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:  # 学习率固定的情况下，当前学习率为初始学习率
            param_group['lr'] = init_lr
        else:  # 学习率不固定的情况下，当前学习率按照既定策略发生变化
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
