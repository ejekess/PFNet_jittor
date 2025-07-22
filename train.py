import datetime
import time
import os


import  jittor as jt
from jittor import nn, optim
from tqdm import tqdm
from jittor.dataset import DataLoader
from Src.dataset import joint_transforms
from tensorboardX import SummaryWriter
import jittor.transform as transforms
from Src.PFNet import PFNet
from Src.dataset.datasets import ImageSet
from Src.misc import AvgMeter, check_mkdir
from config import backbone_path
from config import cod_training_root
import Src.loss as loss



print(jt.__version__)
jt.flags.use_cuda = 1  # 使用CUDA

jt.set_global_seed(2021)  # 设置随机种子

ckpt_path = './ckpt_jittor'

exp_name = 'PFNet'

args = {
    'epoch_num': 45,
    'train_batch_size': 16,
    'last_epoch': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 416,
    'save_point': [5],
    'poly_train': True,
    'optimizer': 'SGD',
}


check_mkdir(ckpt_path)

check_mkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")) + '.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)

# Transform Data.
# 图像、掩码一起翻转缩放
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])


img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



target_transform = transforms.ToTensor()


train_set = ImageSet(root=cod_training_root, joint_transform=joint_transform, img_transform=img_transform, target_transform=target_transform)
print("Train set : {}".format(train_set.total_len ))
print(args['train_batch_size'])

train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=16, shuffle=True)
print(len(train_loader))
print(train_loader.num_workers)

total_epoch = args['epoch_num'] * len(train_loader)

# loss function
structure_loss = loss.structure_loss()
bce_loss = nn.BCEWithLogitsLoss()
iou_loss = loss.IOU()

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + iou_out

    return loss

def main():
    print(args)
    print(exp_name)
    if(os.path.isfile(backbone_path)):
        print("文件存在!")


    net = PFNet(backbone_path).train()


    if args['optimizer'] == 'Adam':
        print("Adam")
        optimizer = optim.Adam([
            {'params': [param for name , param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ])
    else:
        print("SGD")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1* args['lr'], 'weight_decay': args['weight_decay']}
        ],lr=args['lr'], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('Trainnig Resumes From \'%s\''%args['snapshot'])
        net.load_state_dict(jt.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pkl')))
        #total_epoch = (args['epoch_num'] - int(args['snapshot']))*len(train_loader)

    print(total_epoch)



    open(log_path, 'w').write(str(args)+ '\n\n')
    train(net, optimizer)
    writer.close()

def train(net, optimizer):

    if len(args['snapshot']) > 0:
        curr_iter = int(args['snapshot']) * len(train_loader) + 1

    else:
        curr_iter = 1

    start_time = time.time()



    for epoch in range(args['last_epoch'] +1, args['last_epoch']+ 1 + args['epoch_num']):
        loss_record, loss_1_record, loss_2_record, loss_3_record, loss_4_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()



        train_iterator = tqdm(train_loader, total=len(train_loader))

        for data in train_iterator:

            # 使用 Poly 衰减策略，根据训练进度调整学习率
            if args['poly_train']:
                base_lr = args['lr'] * (1- float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, labels = data
            batch_size = inputs.size(0)

            optimizer.zero_grad()

            predict_1, predict_2, predict_3, predict_4 = net(inputs)


            loss_1 = bce_iou_loss(predict_1, labels)
            loss_2 = structure_loss(predict_2, labels)
            loss_3 = structure_loss(predict_3, labels)
            loss_4 = structure_loss(predict_4, labels)

            loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4

            optimizer.step(loss)



            loss_record.update(loss.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)

            if curr_iter % 10 == 0 :

                writer.add_scalar('loss', loss.numpy() , curr_iter)
                writer.add_scalar('loss_1', loss_1.numpy(), curr_iter)
                writer.add_scalar('loss_2', loss_2.numpy(), curr_iter)
                writer.add_scalar('loss_3', loss_3.numpy(), curr_iter)
                writer.add_scalar('loss_4', loss_4.numpy(), curr_iter)


            log = '[%3d], [%6d], [%.6f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f]' % \
                   (epoch, curr_iter, base_lr, loss_record.avg, loss_1_record.avg, loss_2_record.avg, loss_3_record.avg, loss_4_record.avg)

            train_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')

            curr_iter += 1
            jt.sync_all()
            jt.gc()

        if epoch in args['save_point']:

                net.save(os.path.join(ckpt_path, exp_name, '%d.pkl' % epoch))
                return
        if epoch >= args['epoch_num']:

            net.save(os.path.join(ckpt_path, exp_name, '%d.pkl' % epoch))
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            print(exp_name)
            print("Optimization Have Done!")
            return


if __name__ == '__main__':
    main()


