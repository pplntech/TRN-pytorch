import argparse
import os
import time
import json
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from dataset import TSNDataSet
from models import TSN
from transforms import *
from opts import parser
import datasets_video

from tensorboardX import SummaryWriter


best_prec1 = 0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def main():
    global args, best_prec1, num_train_dataset, num_val_dataset, writer
    args = parser.parse_args()
    # if args.no_cudnn:
    #     torch.backends.cudnn.benchmark = False
    # print (torch.backends.cudnn.benchmark)
    # asdf
    _fill_in_None_args()
    _join_result_path()
    check_rootfolders()
    with open(os.path.join(args.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(args), opt_file)

    categories, args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.modality, args.root_path, args.file_type)
    # print(categories, args.train_list, args.val_list, args.root_path, prefix)
    num_class = len(categories)


    args.store_name = '_'.join([args.consensus_type, args.dataset, args.modality, args.arch, args.consensus_type, 'segment%d'% args.num_segments, \
        'key%d'%args.key_dim, 'value%d'%args.value_dim, 'query%d'%args.query_dim, 'queryUpdateby%s'%args.query_update_method,\
        'NoSoftmax%s'%args.no_softmax_on_p, 'hopMethod%s'%args.hop_method])
    print('storing name: ' + args.store_name)

    model = TSN(num_class, args.num_segments, args.modality,
                base_model = args.arch,
                consensus_type = args.consensus_type,
                dropout = args.dropout,
                key_dim = args.key_dim,
                value_dim = args.value_dim,
                query_dim = args.query_dim,
                query_update_method = args.query_update_method,
                partial_bn = not args.no_partialbn,
                freezeBN_Eval = args.freezeBN_Eval,
                freezeBN_Require_Grad_True = args.freezeBN_Require_Grad_True,
                num_hop = args.hop,
                hop_method = args.hop_method,
                num_CNNs = args.num_CNNs,
                no_softmax_on_p = args.no_softmax_on_p,
                freezeBackbone = args.freezeBackbone,
                CustomPolicy = args.CustomPolicy,
                sorting = args.sorting,
                MultiStageLoss=args.MultiStageLoss,
                MultiStageLoss_MLP=args.MultiStageLoss_MLP,
                how_to_get_query=args.how_to_get_query,
                only_query=args.only_query,
                CC=args.CC, channel=args.channel,
                memory_dim=args.memory_dim,
                image_resolution=args.image_resolution,
                how_many_objects=args.how_many_objects,
                Each_Embedding=args.Each_Embedding,
                Curriculum=args.Curriculum,
                Curriculum_dim=args.Curriculum_dim,
                lr_steps=args.lr_steps,
                )


    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
    # asdf
    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_data = TSNDataSet(args.root_path, args.train_list, args.file_type, num_segments=args.num_segments, MoreAug_Rotation=args.MoreAug_Rotation, MoreAug_ColorJitter=args.MoreAug_ColorJitter,
                       new_length=data_length,
                       modality=args.modality,
                       image_tmpl=prefix,
                       phase='train',
                       transform1=torchvision.transforms.Compose([
                           train_augmentation, # GroupMultiScaleCrop[1, .875, .75, .66] AND GroupRandomHorizontalFlip
                       ]),
                       transform2=torchvision.transforms.Compose([
                           Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                           ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                           normalize, # GroupNormalize
                       ]), image_resolution=args.image_resolution)
    train_loader = torch.utils.data.DataLoader(train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    val_data = TSNDataSet(args.root_path, args.val_list, args.file_type,num_segments=args.num_segments, MoreAug_Rotation=args.MoreAug_Rotation, MoreAug_ColorJitter=args.MoreAug_ColorJitter,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   phase='test',
                   transform1=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size)
                   ]),
                   transform2=torchvision.transforms.Compose([
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       normalize,
                   ]), image_resolution=args.image_resolution)
    val_loader = torch.utils.data.DataLoader(val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=True)
    num_train_dataset = len(train_data)
    num_val_dataset = len(val_data)

    # print (num_train_dataset, num_val_dataset)
    # print (len(train_loader), len(val_loader))
    # asdf
    

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss(reduce=False).cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.optimizer=='sgd':
        optimizer = torch.optim.SGD(policies,
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer=='adam':
        optimizer = torch.optim.Adam(policies, lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = torch.optim.SGD(policies,
        #                             args.lr,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay)

    if args.evaluate:
        json_file_path = os.path.join(args.result_path, 'results_epoch%d.json'%args.evaluation_epoch)
        validate(val_loader, model, criterion, 0, json_file=json_file_path, idx2class=categories, epoch = args.evaluation_epoch)
        return


    writer = SummaryWriter(args.result_path)
    log_training = open(os.path.join(args.root_log, '%s.csv' % args.store_name), 'a')
    # print (count_parameters(model))
    # asdf

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            json_file_path = os.path.join(args.result_path, 'results_epoch%d.json'%(epoch + 1))
            # prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), log=log_training, json_file=json_file_path, idx2class=categories)
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * num_train_dataset, log=log_training, json_file=json_file_path, idx2class=categories, epoch=epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
    log_training.close()
    writer.close()



def train(train_loader, model, criterion, optimizer, epoch, log):
    # num_iter = epoch * (num_train_dataset)

    policies = model.module.get_optim_policies(epoch)

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.optimizer=='sgd':
        optimizer = torch.optim.SGD(policies,
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer=='adam':
        optimizer = torch.optim.Adam(policies, lr=args.lr, weight_decay=args.weight_decay)
            
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, ids, indices) in enumerate(train_loader):
        # optimizer = torch.optim.SGD(policies,
        #                             args.lr,
        #                             momentum=args.momentum,
        #          
        # num_iter += input.size(0)
        # measure data loading time
        # print (input.size()) # [72, 6, 224, 224]
        # asdf
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, loss = model(input_var, criterion, phase='train', target=target_var, epoch=epoch) # torch.nn.CrossEntropyLoss().cuda()
        # print (loss)
        # asdf
        loss = loss.mean()
        # output = model(input_var)
        # loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        # Clips gradient norm of an iterable of parameters.
        # if args.clip_gradient is not None:
        if not args.no_clip:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient and total_norm > 100:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.8f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            log.write(output + '\n')
            log.flush()
        if args.break_while_train:
            break

    # final print
    output = ('**Sum Up**, lr: {lr:.8f}\t'
            'Total Time {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
            'Total Data Load Time {data_time.sum:.3f} ({data_time.avg:.3f})\t'
            'Loss  ({loss.avg:.4f})\t'
            'Prec@1  ({top1.avg:.3f})\t'
            'Prec@5  ({top5.avg:.3f})'.format(batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
    print(output)
    log.write(output + '\n')
    log.flush()

    num_iter = (epoch + 1) * (num_train_dataset)
    writer.add_scalar('train_loss', float(losses.avg), num_iter)
    writer.add_scalar('train_acc_top1', float(top1.avg), num_iter)




def validate(val_loader, model, criterion, iter, log=None, json_file=None, idx2class=None, epoch=None):
    if json_file is not None and args.consensus_type in ['MemNN']:
        dicts = {}
        for idx, classstr in enumerate(idx2class):
            dicts[str(idx)] = []


    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, ids, indices) in enumerate(val_loader):
        if json_file is not None and args.consensus_type in ['MemNN']:
            bs = indices.size()[0]
            ids_list = [int(x) for x in ids]
            target_list = target.numpy().tolist()
            indices_list = indices.numpy().tolist()

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # print (input_var)
        # asdf


        # compute output
        if json_file is not None and args.consensus_type in ['MemNN']:
            # output, loss = model(input_var, criterion, phase='eval', target=target_var, eval=False)
            if args.how_many_objects == 2:
                output, attentions, attentions_2, loss = model(input_var, criterion, phase='eval', target=target_var, eval=True, epoch=epoch)
                attentions = attentions.cpu().data.numpy().tolist()
                attentions_2 = attentions_2.cpu().data.numpy().tolist()
            else:
                output, attentions, loss = model(input_var, criterion, phase='eval', target=target_var, eval=True, epoch=epoch)
                attentions = attentions.cpu().data.numpy().tolist()
            # output, attentions = model(input_var, eval=True)
        else:
            # print (input_var)
            output, loss = model(input_var, criterion, phase='eval', target=target_var, eval=False)
            # output = model(input_var)
        # loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        if json_file is not None and args.consensus_type in ['MemNN']:
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            pred = pred.cpu().data.numpy().tolist()[0]

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if json_file is not None and args.consensus_type in ['MemNN']:
            for each_bs in range(bs):
                each_dict = {}
                each_dict['id'] = ids_list[each_bs]
                each_dict['GT'] = target_list[each_bs]
                each_dict['framenums'] = indices_list[each_bs]

                each_dict['hop_probabilities'] = attentions[each_bs]
                if args.how_many_objects == 2:
                    each_dict['hop_probabilities_2'] = attentions_2[each_bs]
                each_dict['Predict'] = pred[each_bs]
                # print (each_dict) # , sum(attentions[each_bs][0]), sum(attentions[each_bs][1]))
                dicts[str(each_dict['GT'])].append(each_dict)

        if i % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            print(output)
            if log is not None:
                log.write(output + '\n')
                log.flush()

        if args.break_while_val:
            break

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)
    if log is not None:
        log.write(output + ' ' + output_best + '\n')
        log.flush()

    if json_file is not None and args.consensus_type in ['MemNN']:
        with open(json_file, 'w') as f:
            json.dump(dicts,f)

    writer.add_scalar('val_loss', float(losses.avg), iter)
    writer.add_scalar('val_acc_top1', float(top1.avg), iter)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name),'%s/%s_best.pth.tar' % (args.root_model, args.store_name))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            # os.mkdir(folder)
            os.makedirs(folder)

def _join_result_path():
    args.root_log = os.path.join(args.result_path, args.root_log)
    args.root_model = os.path.join(args.result_path, args.root_model)
    args.root_output = os.path.join(args.result_path, args.root_output)

def _fill_in_None_args():
    if args.key_dim is None: args.key_dim = args.img_feature_dim
    if args.value_dim is None: args.value_dim = args.img_feature_dim
    if args.query_dim is None: args.query_dim = args.img_feature_dim
if __name__ == '__main__':
    main()