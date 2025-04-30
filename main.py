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

from util.dataset import DepressedDataset, AugDepressedDataset
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json
from transformers import WhisperProcessor, WhisperModel, WhisperForConditionalGeneration
from torch.nn.utils.rnn import pad_sequence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=100)
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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')
    # dataset
    parser.add_argument('--dataset', type=str, default='autumn')
    parser.add_argument('--data_folder', type=str, default='../Data/')    
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    
    parser.add_argument('--n_cls', type=int, default=2,
                        help='set k-way classification problem for class')
    parser.add_argument('--sample_rate', type=int,  default=16000, 
                        help='sampling rate when load audio data, and it denotes the number of samples per one second')
    parser.add_argument('--train_annotation_file', type=str, default='train.csv')
    parser.add_argument('--test_annotation_file', type=str, default='test.csv')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='weighted cross entropy loss (higher weights on abnormal class)')
    

    # model
    parser.add_argument('--model', type=str, default='facebook/wav2vec2-base')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--framework', type=str, default='transformers', 
                        help='using pretrained speech models from s3prl or huggingface', choices=['transformers'])
    
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained encoder model')
    parser.add_argument('--ma_update', action='store_true',
                        help='whether to use moving average update for model')
    parser.add_argument('--ma_beta', type=float, default=0.5,
                        help='moving average value')
    parser.add_argument('--method', type=str, default='ce')
    
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--T', type=int, default=5, help='len of waveform')
    parser.add_argument('--overlap', type=float, default=0.5, help='overlap for each segment')
    parser.add_argument('--num_lstm', type=int, default=1, help='# of lstm layers')
    
    parser.add_argument('--domain', type=str, default='gender', choices=['gender', 'age'])
    parser.add_argument('--domain_adaptation', action='store_true')
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)
    
    if args.domain_adaptation:
        args.save_folder = os.path.join(args.save_dir, 'da', args.model_name)
    else:
        args.save_folder = os.path.join(args.save_dir, args.model_name)
    
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.domain_adaptation:
        args.save_folder = os.path.join(args.save_dir, 'da', args.model_name)
        args.m_cls = 2
    
    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    
    if args.dataset == 'autumn':
        if args.n_cls == 2:
            args.cls_list = ['Normal', 'Suicidal']
        else:
            raise NotImplementedError

    return args


def collate_fn(batch):
    input_features, labels, genders, ages = zip(*batch)    
    input_features_padded = pad_sequence(input_features, batch_first=True, padding_value=0.0)

    return input_features_padded, labels, genders, ages

def set_loader(args):
    if args.augment:
        train_dataset = AugDepressedDataset(train_flag=True, transform=None, args=args, print_flag=True)
    else:
        train_dataset = DepressedDataset(train_flag=True, transform=None, args=args, print_flag=True)
    test_dataset = DepressedDataset(train_flag=False, transform=None, args=args, print_flag=True)
    
    args.class_nums = train_dataset.class_nums
    
    #print('class_nums', args.class_nums)
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    
    return train_loader, test_loader, args
    


def set_model(args):
    from transformers import Wav2Vec2Model, HubertModel, WavLMModel, AutoFeatureExtractor, Wav2Vec2Processor
    from models.speech import PretrainedSpeechModels
    
    if args.model == 'facebook/wav2vec2-base': #wav2vec2-based models
        speech_extractor = Wav2Vec2Model
        args.processor = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base')
        model = PretrainedSpeechModels(speech_extractor, args.model, 768, args.num_lstm)
    
    elif args.model == 'facebook/hubert-base-ls960': #hubert-based models
        speech_extractor = HubertModel
        args.processor = AutoFeatureExtractor.from_pretrained('facebook/hubert-base-ls960')        
        model = PretrainedSpeechModels(speech_extractor, args.model, 768, args.num_lstm)
    
    elif args.model == 'microsoft/wavlm-base-plus': #wavlm-based models
        speech_extractor = WavLMModel
        args.processor = AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base-plus')        
        model = PretrainedSpeechModels(speech_extractor, args.model, 768, args.num_lstm)
    
    if args.domain_adaptation:
        class_classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast'] else deepcopy(model.mlp_head)
        domain_classifier = nn.Linear(model.final_feat_dim, args.m_cls) if args.model not in ['ast'] else deepcopy(model.domain_mlp_head)
    else:
        classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast'] else deepcopy(model.mlp_head)
    
    criterion = nn.CrossEntropyLoss()
    if args.domain_adaptation:
        criterion2 = nn.CrossEntropyLoss()
    '''
    if not args.weighted_loss:
        weights = None
        criterion = nn.CrossEntropyLoss()
    else: #weighted_loss is used only for imbalanced setting
        weights = torch.tensor(args.class_nums, dtype=torch.float32)
        weights = 1.0 / (weights / weights.sum())
        weights /= weights.sum()
        
        criterion = nn.CrossEntropyLoss(weight=weights)
    '''
    
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
        if args.domain_adaptation:
            criterion = [criterion.cuda(), criterion2.cuda()]
        else:
            criterion = [criterion.cuda()]    
    
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.cuda()
    if args.domain_adaptation:
        classifier = [class_classifier.cuda(), domain_classifier.cuda()]
    else:
        classifier.cuda()
    
    if args.domain_adaptation:
        optim_params = list(model.parameters()) + list(classifier[0].parameters()) + list(classifier[-1].parameters())
    else:
        optim_params = list(model.parameters()) + list(classifier.parameters())
    
    optimizer = set_optimizer(args, optim_params)
    
    return model, classifier, criterion, optimizer


def train(train_loader, model, classifier, criterion, optimizer, epoch, args, scaler=None):
    model.train()
    if args.domain_adaptation:
        domain_classifier = classifier[1]
        domain_classifier.train()
        classifier = classifier[0]
        classifier.train()
    else:
        classifier.train()
    
    #print('in training')        
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels, genders, ages) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        labels = torch.stack(labels).cuda(non_blocking=True)
        genders = torch.stack(genders).cuda(non_blocking=True)
        ages = torch.stack(ages).cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        if args.domain_adaptation:
            if args.domain == 'gender':
                domain_labels = genders
            elif args.domain == 'age':
                domain_labels = ages
                                
        if args.ma_update:
            # store the previous iter checkpoint
            with torch.no_grad():
                if args.domain_adaptation:
                    ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict()), deepcopy(domain_classifier.state_dict())]
                    p = float(idx + epoch * len(train_loader)) / args.epochs + 1 / len(train_loader)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                else:
                    ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]
                    alpha = None
                
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast():
            if args.method == 'ce':
                if args.model in ['facebook/hubert-base-ls960', 'facebook/wav2vec2-base', 'microsoft/wavlm-base-plus']:
                    images = torch.squeeze(images, 1)        
                    features = model(images, args=args, alpha=alpha, training=True)
                    #print('features', features.size())
                    
                    if args.domain_adaptation:
                        #features = (features, domain_features) # domain_features -> ReverseLayerF                    
                        #print('label', labels)
                        output = classifier(features)
                        class_loss = criterion[0](output, labels)
                        #print('output {} labels {}'.format(output.size(), labels.size()))
                                       
                        domain_output = ReverseLayerF.apply(features, alpha)
                        domain_output = domain_classifier(domain_output)
                        #print('domain_output {} domain_labels {}'.format(domain_output.size(), domain_labels.size()))
                        #print('domain_labels', domain_labels)                        
                        domain_loss = criterion[1](domain_output, domain_labels)
                                                
                        loss = class_loss +  domain_loss
                    else:
                        output = classifier(features)
                        #print('output', output.size())
                        #print('labels', labels)
                        loss = criterion[0](output, labels)
                        
                    
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
                if args.domain_adaptation:
                    domain_classifier = update_moving_average(args.ma_beta, domain_classifier, ma_ckpt[1])

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


from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

from sklearn.metrics import accuracy_score, f1_score
def validate(val_loader, model, classifier, criterion, args, best_acc, best_model=None):
    save_bool = False
    model.eval()
    if args.domain_adaptation:
        classifier = classifier[0]
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls
    
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, genders, ages) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = torch.stack(labels).cuda(non_blocking=True)
            #genders = torch.stack(genders).cuda(non_blocking=True)
            #ages = torch.stack(ages).cuda(non_blocking=True)
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
    
    best_acc = [0]
    
    model, classifier, criterion, optimizer = set_model(args)
    print('# of params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    train_loader, test_loader, args = set_loader(args)
    
    if args.weighted_loss:
        #weighted_loss is used only for imbalanced setting
        weights = torch.tensor(args.class_nums, dtype=torch.float32)
        weights = 1.0 / (weights / weights.sum())
        weights /= weights.sum() 
        
        new_criterion = nn.CrossEntropyLoss(weight=weights)
        criterion[0] = new_criterion.cuda()
        #criterion = [criterion.cuda()]
    
    print('# of train_loader {} test_loader {}'.format(len(train_loader), len(test_loader)))
    
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
    
    total_loss = 0.0
     
    if not args.eval:
        print('Experiments {} start'.format(args.model_name))
        print('Training for {} epochs on {} dataset'.format(args.epochs, args.dataset))
        
        train_losses = []
        test_losses = []
        test_accs = []
        
        for epoch in range(args.start_epoch, args.epochs+1):
            adjust_learning_rate(args, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            
            loss, _ = train(train_loader, model, classifier, criterion, optimizer, epoch, args, scaler)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, Loss {:.2f}'.format(epoch, time2-time1, loss))
            
            # eval for one epoch
            
            best_acc, best_model, save_bool, test_loss, _ = validate(test_loader, model, classifier, criterion, args, best_acc, best_model)
            print('Test Acc = {}'.format(best_acc[0]))
                                    
            train_losses.append(loss)
            test_losses.append(test_loss)
            test_accs.append(best_acc[0])
            
        plot_metrics(train_losses, 'train.png', args)
        np.save(os.path.join(args.save_folder, 'train_losses'), train_losses)
        
        plot_metrics(test_losses, 'test.png', args)
        np.save(os.path.join(args.save_folder, 'test_losses'), test_losses)

        plot_metrics(test_accs, 'test_acc.png', args)
        np.save(os.path.join(args.save_folder, 'test_accs'), test_accs)

        # save a checkpoint of classifier with the best accuracy or score
        save_file = os.path.join(args.save_folder, 'best_test.pth')
        model.load_state_dict(best_model[0])
        classifier[0].load_state_dict(best_model[1]) if args.domain_adaptation else classifier.load_state_dict(best_model[1])
        save_model(model, optimizer, args, epoch, save_file, classifier[0] if args.domain_adaptation else classifier)
    else:
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        best_acc, _, _  = validate(test_loader, model, criterion, args, best_acc)
    
    print('{} finished'.format(args.model_name))
    update_json('%s' % args.model_name, best_model, path=os.path.join(args.save_dir, 'results.json'))
    
if __name__ == '__main__':
    main()
