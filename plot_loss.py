import matplotlib.pyplot as plt
import numpy as np
import os

import argparse
parser = argparse.ArgumentParser('argument for supervised training')
parser.add_argument('--checkpoint', type=str, default='tess_ast_ce_ast_bs8_lr5e-5_ep50_seed1_yaf')
parser.add_argument('--save_dir', type=str, default='./analysis')
args = parser.parse_args()



def save_fig(args, object, files, data, epochs):
    save_dir = os.path.join(args.save_dir, args.checkpoint)
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for model, loss in data.items():
        plt.plot(range(1, epochs + 1), loss, label=model)
    
    text = ''
    
    if object == 'train_losses':
        text = 'Training Loss'
    elif object == 'test_losses':
        text = 'Test Loss'
    elif object == 'train_accs':
        text = 'Training Accuracy'
    elif object == 'test_accs':
        text = 'Test Accuracy'
    
    plt.xlabel('Epoch')
    plt.ylabel(text)
    plt.title('{} Across Microphones'.format(text))
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(os.path.join(save_dir, object+'.png'))
    

def parse(args, object):
    files = {
        'MEMS': './save/{}/{}.npy'.format(args.checkpoint, object),
        'ch1': './save/{}_ch1/{}.npy'.format(args.checkpoint, object),
        'ch2': './save/{}_ch2/{}.npy'.format(args.checkpoint, object),
        'ch3': './save/{}_ch3/{}.npy'.format(args.checkpoint, object),
        'ch4': './save/{}_ch4/{}.npy'.format(args.checkpoint, object),
        'ch5': './save/{}_ch5/{}.npy'.format(args.checkpoint, object),
        'ch6': './save/{}_ch6/{}.npy'.format(args.checkpoint, object),
        'ch7': './save/{}_ch7/{}.npy'.format(args.checkpoint, object),    
    }
    
    data = {model: np.load(path) for model, path in files.items()}
    epochs = len(next(iter(data.values())))
    save_fig(args, object, files, data, epochs)
'''


training_loss_files = {
    'MEMS': './save/{}/train_losses.npy'.format(args.checkpoint),
    'ch1': './save/{}_ch1/train_losses.npy'.format(args.checkpoint),
    'ch2': './save/{}_ch2/train_losses.npy'.format(args.checkpoint),
    'ch3': './save/{}_ch3/train_losses.npy'.format(args.checkpoint),
    'ch4': './save/{}_ch4/train_losses.npy'.format(args.checkpoint),
    'ch5': './save/{}_ch5/train_losses.npy'.format(args.checkpoint),
    'ch6': './save/{}_ch6/train_losses.npy'.format(args.checkpoint),
    'ch7': './save/{}_ch7/train_losses.npy'.format(args.checkpoint),    
} 

test_loss_files = {
    'MEMS': './save/{}/test_losses.npy'.format(args.checkpoint),
    'ch1': './save/{}_ch1/test_losses.npy'.format(args.checkpoint),
    'ch2': './save/{}_ch2/test_losses.npy'.format(args.checkpoint),
    'ch3': './save/{}_ch3/test_losses.npy'.format(args.checkpoint),
    'ch4': './save/{}_ch4/test_losses.npy'.format(args.checkpoint),
    'ch5': './save/{}_ch5/test_losses.npy'.format(args.checkpoint),
    'ch6': './save/{}_ch6/test_losses.npy'.format(args.checkpoint),
    'ch7': './save/{}_ch7/test_losses.npy'.format(args.checkpoint),    
} 

training_acc_files = {
    'MEMS': './save/{}/train_accs.npy'.format(args.checkpoint),
    'ch1': './save/{}_ch1/train_accs.npy'.format(args.checkpoint),
    'ch2': './save/{}_ch2/train_accs.npy'.format(args.checkpoint),
    'ch3': './save/{}_ch3/train_accs.npy'.format(args.checkpoint),
    'ch4': './save/{}_ch4/train_accs.npy'.format(args.checkpoint),
    'ch5': './save/{}_ch5/train_accs.npy'.format(args.checkpoint),
    'ch6': './save/{}_ch6/train_accs.npy'.format(args.checkpoint),
    'ch7': './save/{}_ch7/train_accs.npy'.format(args.checkpoint),    
} 

test_acc_files = {
    'MEMS': './save/{}/test_accs.npy'.format(args.checkpoint),
    'ch1': './save/{}_ch1/test_accs.npy'.format(args.checkpoint),
    'ch2': './save/{}_ch2/test_accs.npy'.format(args.checkpoint),
    'ch3': './save/{}_ch3/test_accs.npy'.format(args.checkpoint),
    'ch4': './save/{}_ch4/test_accs.npy'.format(args.checkpoint),
    'ch5': './save/{}_ch5/test_accs.npy'.format(args.checkpoint),
    'ch6': './save/{}_ch6/test_accs.npy'.format(args.checkpoint),
    'ch7': './save/{}_ch7/test_accs.npy'.format(args.checkpoint),    
} 
'''
parse(args, 'train_losses')
parse(args, 'test_losses')
parse(args, 'train_accs')
parse(args, 'test_accs')
'''
plt.figure(figsize=(10, 6))
for model, loss in loss_data.items():
    plt.plot(range(1, epochs + 1), loss, label=model)

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Across Models')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('./check.png')
'''

  