from copy import deepcopy
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import math
import time
import random
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

from util.emo_dataset import EmoDataset, EmoCREMADataset
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json
from util.augmentation import SpecAugment
from models import get_backbone_class

from pytorch_metric_learning import losses, miners, samplers, testers, trainers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save') 
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluation with pretrained encoder and classifier')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='weighted cross entropy loss (higher weights on abnormal class)')
    # dataset
    parser.add_argument('--dataset', type=str, default='tess')
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--train_annotation_file', type=str, default='./data/OAF_train_data.csv')
    parser.add_argument('--test_annotation_file', type=str, default='./data/OAF_test_data.csv')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    # icbhi dataset
    parser.add_argument('--class_split', type=str, default='emo',
                        help='emo: (neutral, happy, surprised, angry, fear, disgust, sad)')
    parser.add_argument('--n_cls', type=int, default=0,
                        help='set k-way classification problem for class')
    parser.add_argument('--sample_rate', type=int,  default=16000, 
                        help='sampling rate when load audio data, and it denotes the number of samples per one second')
    parser.add_argument('--desired_length', type=int,  default=6, 
                        help='fixed length size of individual cycle')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='the number of mel filter banks')
    parser.add_argument('--pad_types', type=str,  default='repeat', 
                        help='zero: zero-padding, repeat: padding with duplicated samples, aug: padding with augmented samples')
    parser.add_argument('--resz', type=float, default=1, 
                        help='resize the scale of mel-spectrogram')
    parser.add_argument('--raw_augment', type=int, default=0, 
                        help='control how many number of augmented raw audio samples')
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup',  ### check
                        help='policy (argument values) for SpecAugment')
    parser.add_argument('--specaug_mask', type=str, default='mean', 
                        help='specaug mask value', choices=['mean', 'zero'])

    # model
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--framework', type=str, default='transformers', 
                        help='using pretrained speech models from s3prl or huggingface', choices=['s3prl', 'transformers'])
    
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained encoder model')
    parser.add_argument('--from_sl_official', action='store_true',
                        help='load from supervised imagenet-pretrained model (official PyTorch)')
    parser.add_argument('--ma_update', action='store_true',
                        help='whether to use moving average update for model')
    parser.add_argument('--ma_beta', type=float, default=0,
                        help='moving average value')
    parser.add_argument('--audioset_pretrained', action='store_true',
                        help='load from imagenet- and audioset-pretrained model')
    parser.add_argument('--method', type=str, default='ce')
    
    
    parser.add_argument('--channel', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--mic', type=str, default='original', choices=['original', 'mems', 'gpas'])
    parser.add_argument('--test_origin', action='store_true',
                        help='set the test samples as original mic')
    
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)
    args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    if args.dataset == 'tess':
        if args.class_split == 'emo':  
            if args.n_cls == 7:
                args.cls_list = ['neutral', 'happy', 'surprised', 'angry', 'fear', 'disgust', 'sad']
            else:
                raise NotImplementedError
    elif args.dataset == 'crema':
        if args.class_split == 'emo':  
            if args.n_cls == 6:
                args.cls_list = ['neutral', 'happy', 'angry', 'fear', 'disgust', 'sad']
            else:
                raise NotImplementedError

    return args


def set_loader(args):
    if args.dataset in ['tess', 'crema']:
        
        args.h, args.w = 598, 128
        
        train_transform = [transforms.ToTensor(),
                            SpecAugment(args),
                            transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        val_transform = [transforms.ToTensor(),
                        transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        train_transform = transforms.Compose(train_transform)
        val_transform = transforms.Compose(val_transform)
        
        if args.dataset == 'tess':
            train_dataset = EmoDataset(train_flag=True, transform=None if args.model in ['facebook/wav2vec2-base', 'facebook/hubert-base-ls960', 'microsoft/wavlm-base-plus'] else train_transform, args=args, print_flag=True)
            val_dataset = EmoDataset(train_flag=False, transform=None if args.model in ['facebook/wav2vec2-base', 'facebook/hubert-base-ls960', 'microsoft/wavlm-base-plus'] else val_transform, args=args, print_flag=True)
        else:
            train_dataset = EmoCREMADataset(train_flag=True, transform=None if args.model in ['facebook/wav2vec2-base', 'facebook/hubert-base-ls960', 'microsoft/wavlm-base-plus'] else train_transform, args=args, print_flag=True)
            val_dataset = EmoCREMADataset(train_flag=False, transform=None if args.model in ['facebook/wav2vec2-base', 'facebook/hubert-base-ls960', 'microsoft/wavlm-base-plus'] else val_transform, args=args, print_flag=True)
        args.class_nums = train_dataset.class_nums
    else:
        raise NotImplemented
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, val_loader, args
    

def set_model(args):
    from transformers import Wav2Vec2Model, HubertModel, WavLMModel, AutoFeatureExtractor
    from models.speech import PretrainedSpeechModels    
    kwargs = {}
    if args.model == 'ast':
        kwargs['input_fdim'] = int(args.h * args.resz)
        kwargs['input_tdim'] = int(args.w * args.resz)
        kwargs['label_dim'] = args.n_cls
        kwargs['imagenet_pretrain'] = args.from_sl_official
        kwargs['audioset_pretrain'] = args.audioset_pretrained
        model = get_backbone_class(args.model)(**kwargs)
    
    elif args.model in ['resnet18', 'cnn6', 'efficientnet_b0']:        
        model = get_backbone_class(args.model)(**kwargs)
        
    elif args.model == 'facebook/hubert-base-ls960': #hubert-based models
        speech_extractor = HubertModel
        model = PretrainedSpeechModels(speech_extractor, args.model, 768)
    elif args.model == 'facebook/wav2vec2-base': #wav2vec2-based models
        speech_extractor = Wav2Vec2Model
        model = PretrainedSpeechModels(speech_extractor, args.model, 768)
    elif args.model == 'microsoft/wavlm-base-plus': #wavlm-based models
        speech_extractor = WavLMModel
        model = PretrainedSpeechModels(speech_extractor, args.model, 768)
    
    classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast'] else deepcopy(model.mlp_head)
    
    if not args.weighted_loss:
        weights = None
        criterion = nn.CrossEntropyLoss()
    else: #weighted_loss is used only for imbalanced setting
        weights = torch.tensor(args.class_nums, dtype=torch.float32)
        weights = 1.0 / (weights / weights.sum())
        weights /= weights.sum()
        
        criterion = nn.CrossEntropyLoss(weight=weights)
    
    if args.model not in ['ast', 'facebook/wav2vec2-base', 'facebook/hubert-base-ls960', 'microsoft/wavlm-base-plus'] and args.from_sl_official:
        model.load_sl_official_weights()
        print('pretrained model loaded from PyTorch ImageNet-pretrained')
    
    # load SSL pretrained checkpoint for linear evaluation
    if args.pretrained and args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt['model']
        # HOTFIX: always use dataparallel during SSL pretraining
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")
            if not 'mlp_head' in k: #del mlp_head
                new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        
        if ckpt.get('classifier', None) is not None:
            classifier.load_state_dict(ckpt['classifier'], strict=True)

        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))
    
    if args.method == 'ce':
        criterion = [criterion.cuda()]
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.cuda()
    classifier.cuda()
    
    optim_params = list(model.parameters()) + list(classifier.parameters())
    optimizer = set_optimizer(args, optim_params)
    
    return model, classifier, criterion, optimizer


def train(train_loader, model, classifier, criterion, optimizer, epoch, args, scaler=None):
    model.train()
    classifier.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        # data load
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        if args.ma_update:
            # store the previous iter checkpoint
            with torch.no_grad():
                ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]
                alpha = None

        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast():
            if args.method == 'ce':
                if args.model in ['facebook/hubert-base-ls960', 'facebook/wav2vec2-base', 'microsoft/wavlm-base-plus']:
                    images = torch.squeeze(images, 1)
                    features = model(images, args=args, alpha=alpha, training=True)
                else:
                    images = torch.squeeze(images, 1)
                    images = images.permute(0, 3, 1, 2)
                    features = model(args.transforms(images), args=args, alpha=alpha, training=True)
                        
                output = classifier(features)
                loss = criterion[0](output, labels)

        losses.update(loss.item(), bsz)
        [acc1], _ = accuracy(output[:bsz], labels, topk=(1,))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.ma_update:
            with torch.no_grad():
                # exponential moving average update
                model = update_moving_average(args.ma_beta, model, ma_ckpt[0])
                classifier = update_moving_average(args.ma_beta, classifier, ma_ckpt[1])

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, float(top1.avg)


def plot_losses(losses, name, args):
    epochs = range(1, len(losses) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.plot(epochs, losses, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")
    
    plt.savefig(os.path.join(args.save_folder, name))
    plt.cla()
    plt.clf()

def plot_metrics(acc, name, args):
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.plot(epochs, acc, label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")
    
    plt.savefig(os.path.join(args.save_folder, name))
    plt.cla()
    plt.clf()


from sklearn.metrics import accuracy_score, f1_score

def validate(val_loader, model, classifier, criterion, args, best_acc, best_model=None):
    save_bool = False
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls
    
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            

            with torch.cuda.amp.autocast():
                if args.model in ['facebook/hubert-base-ls960', 'facebook/wav2vec2-base', 'microsoft/wavlm-base-plus']:
                    images = torch.squeeze(images, 1)
                    features = model(images, args=args, training=False)
                else:
                    images = torch.squeeze(images, 1)
                    images = images.permute(0, 3, 1, 2)
                    features = model(images, args=args, training=False)
                output = classifier(features)
                loss = criterion[0](output, labels)

            output_f1 = output.cpu().numpy()
            output_f1 = np.argmax(output_f1, axis=1)
            labels_f1 = labels.cpu().numpy()
            
            all_y_true.extend(labels_f1)
            all_y_pred.extend(output_f1)
            
            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    final_f1 = f1_score(all_y_true, all_y_pred, average='macro')
    
    
    acc = float(top1.avg)
    if acc > best_acc[0]:
        save_bool = True
        best_acc = [acc]
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]

    print(' * Acc: {:.2f} (Best Acc: {:.2f})'.format(acc, best_acc[0]))
    print(' * F1: {:} '.format(final_f1))
    return best_acc, best_model, save_bool, losses.avg, acc


def main():
    args = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    #torch.autograd.set_detect_anomaly(True)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    best_model = None
    if args.dataset in ['tess', 'crema']:
        best_acc = [0]  # Accuracy
    
    args.transforms = SpecAugment(args)
    
    train_loader, val_loader, args = set_loader(args)
    model, classifier, criterion, optimizer = set_model(args)
    print('model', model)
    print('# of params', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    # use mix_precision:
    scaler = torch.cuda.amp.GradScaler()
    
    print('*' * 20)
    
    train_losses = []
    #dev_losses = []
    test_losses = []
    
    train_accs = []
    test_accs = []
     
    if not args.eval:
        print('Experiments {} start'.format(args.model_name))
        print('Training for {} epochs on {} dataset'.format(args.epochs, args.dataset))
        
        for epoch in range(args.start_epoch, args.epochs+1):
            adjust_learning_rate(args, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            
            train_loss, train_acc = train(train_loader, model, classifier, criterion, optimizer, epoch, args, scaler)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2-time1, train_acc))
            
            # eval for one epoch
            best_acc, best_model, save_bool, val_loss, val_acc = validate(val_loader, model, classifier, criterion, args, best_acc, best_model)
            # save a checkpoint of model and classifier when the best score is updated
            if save_bool:            
                save_file = os.path.join(args.save_folder, 'best_epoch_{}.pth'.format(epoch))
                print('Best ckpt is modified with Acc = {:.2f} when Epoch = {}'.format(best_acc[0], epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier)
                
            if epoch % args.save_freq == 0:
                save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier)
            
            train_losses.append(train_loss)
            test_losses.append(val_loss)
            
            train_accs.append(train_acc)
            test_accs.append(val_acc)

        # save a checkpoint of classifier with the best accuracy or score
        save_file = os.path.join(args.save_folder, 'best.pth')
        model.load_state_dict(best_model[0])
        classifier.load_state_dict(best_model[1])
        save_model(model, optimizer, args, epoch, save_file, classifier)
        
        
        plot_losses(train_losses, 'train.png', args)
        np.save(os.path.join(args.save_folder, 'train_losses'), train_losses)
                
        plot_losses(test_losses, 'test.png', args)
        np.save(os.path.join(args.save_folder, 'test_losses'), test_losses)
        
        plot_metrics(train_accs, 'train_acc.png', args)
        np.save(os.path.join(args.save_folder, 'train_accs'), train_accs)
                
        plot_metrics(test_accs, 'test_acc.png', args)
        np.save(os.path.join(args.save_folder, 'test_accs'), test_accs)
        
    else:
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        best_acc, _, _  = validate(val_loader, model, classifier, criterion, args, best_acc)
    
    print('{} finished'.format(args.model_name))
    update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results.json'))
    
if __name__ == '__main__':
    main()
