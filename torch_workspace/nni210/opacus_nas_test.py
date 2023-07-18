import nni
import argparse
import numpy as np
import opacus.validators
import torch
import torch.nn as nn
#import nni.nas.nn.pytorch as nn

#import nni.retiarii.nn.pytorch as nn
#nni.nas.nn.pytorch._layers

#import torch.nn.functional as F
from nni.retiarii.nn.pytorch.nn import functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
from nni.utils import merge_parameter
import time
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,CosineAnnealingLR
from nni.retiarii import fixed_arch
from nni.nas.hub.pytorch import ENAS as EnasSpace
from nni.nas.hub.pytorch import DARTS as DARTSSpace
import random
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/home/xuedaxuan/torch_workspace/nni210/runs/enas50')
seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class PoissonSampler(torch.utils.data.Sampler):
    def __init__(self, num_examples, batch_size):
        self.inds = np.arange(num_examples)
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(num_examples / batch_size))
        self.sample_rate = self.batch_size / (1.0 * num_examples)
        super().__init__(None)

    def __iter__(self):
        # select each data point independently with probability `sample_rate`
        for i in range(self.num_batches):
            batch_idxs = np.random.binomial(n=1, p=self.sample_rate, size=len(self.inds))
            # if sample number not equal to batch size, privacy engine will report error
            while np.sum(batch_idxs) != self.batch_size:
                batch_idxs = np.random.binomial(n=1, p=self.sample_rate, size=len(self.inds))
            batch = self.inds[batch_idxs.astype(np.bool)]
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

def random_sample_loader(loader, device, drop_last=False, sample_batches=False):
    datas = []
    targets = []

    for (data, target) in loader:
        data, target = data.to(device), target.to(device)
        datas.append(data)
        targets.append(target)

    datas = torch.cat(datas, axis=0)
    targets = torch.cat(targets, axis=0)
    data = torch.utils.data.TensorDataset(datas, targets)
    if sample_batches:
        sampler = PoissonSampler(len(datas), loader.batch_size)
        return torch.utils.data.DataLoader(data, batch_sampler=sampler,
                                           num_workers=0, pin_memory=False)
    else:
        shuffle = isinstance(loader.sampler, torch.utils.data.RandomSampler)
        return torch.utils.data.DataLoader(data,
                                           batch_size=loader.batch_size,
                                           shuffle=shuffle,
                                           num_workers=0,
                                           pin_memory=False,
                                           drop_last=drop_last)

def cutout_transform(img, length: int = 16):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img

def get_nas_model():
    exported_arch = torch.load('/home/xuedaxuan/torch_workspace/nni210/exported_arch/enas_notrick_reduction_10StdAct_reluconvbn_stdsep_all_5node_50_200.pth')
    with fixed_arch(exported_arch):
        final_model = EnasSpace(width=16, num_cells=6, dataset='cifar')
        #final_model = DARTSSpace(width=16, num_cells=6, dataset='cifar', auxiliary_loss=True, drop_path_prob=0.2)
    # print(opacus.validators.ModuleValidator.validate(final_model))
    # print(opacus.validators.ModuleValidator.is_valid(final_model))
    print(final_model)
    return final_model


def train(args, model, device, train_loader, optimizer, privacy_engine, lr_scheduler, epoch):
    model.train()
    #交叉熵损失函数用于分类
    criterion = nn.CrossEntropyLoss()
    losses = []
    acc = 0
    #enumerate()将一个可遍历的数据对象组合为一个索引序列
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()#清空过往梯度，把loss关于weight的导数变成0
        output = model(data)
        loss = criterion(output, target)
        # logits, logits_aux = output
        #
        # main_loss = criterion(logits, target)
        # aux_loss = criterion(logits_aux,target)
        #
        # loss = main_loss + 0.4 * aux_loss
        loss.backward()
        optimizer.step()
        # lrs.append(lr_scheduler.get_last_lr()[0])
        losses.append(loss.item())
        # lr_scheduler.step()
        pred = output.argmax(dim=1, keepdim=True)
        acc += pred.eq(target.view_as(pred)).sum().item()  # torch.eq()函数就是用来比较对应位置数字，相同则为1，否则为0;torch.eq().sum()就是将所有值相加;
    a = 100.0 * acc / len(train_loader.dataset)
    # plt.plot(lrs, losses)
    # plt.xscale('log')
    # plt.xlabel('Learning Rate')
    # plt.ylabel('Loss')
    # plt.show()
    if not args.disable_dp:
        epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha} "
            f"Train ACC: {a} "
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} \t Train Acc: {a} ")
    return np.mean(losses), a

def mm_test(model, device, test_loader,epoch):
    model.eval()#在模型中，我们通常会加上Dropout层和batch normalization层，在模型预测阶段，我们需要将这些层设置到预测模式，model.eval()就是帮我们一键搞定的，如果在预测的时候忘记使用model.eval()，会导致不一致的预测结果。
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():#当requires_grad设置为False时,反向传播时就不会自动求导了
        for data, target in tqdm(test_loader):#tqdm进度条库
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()#torch.eq()函数就是用来比较对应位置数字，相同则为1，否则为0;torch.eq().sum()就是将所有值相加;

    test_loss /= len(test_loader.dataset)


    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_loss, correct / len(test_loader.dataset)

def get_data(name, augment=False):
    if name == "cifar10":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            train_transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            print('\n', train_transforms, '\n')
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # normalize,
                # cutout_transform,
            ]

        train_set = datasets.CIFAR10(root="/home/xuedaxuan/torch_workspace/pt.darts/data", train=True,
                                     transform=transforms.Compose(train_transforms),
                                     download=True)

        test_set = datasets.CIFAR10(root="/home/xuedaxuan/torch_workspace/pt.darts/data", train=False,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), normalize]
                                    ))

    elif name == "fmnist":
        train_set = datasets.FashionMNIST(root='./datasets1', train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

        test_set = datasets.FashionMNIST(root='./datasets1', train=False,
                                         transform=transforms.ToTensor(),
                                         download=True)

    elif name == "mnist":
        train_set = datasets.MNIST(root='.data', train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

        test_set = datasets.MNIST(root='.data', train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)

    else:
        raise ValueError(f"unknown dataset {name}")

    return train_set, test_set

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-dp", action="store_true", default=False,help="Disable privacy training and just train with vanilla SGD", )
    parser.add_argument('--dataset', choices=['cifar10', 'fmnist', 'mnist'], default='cifar10')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--mini_batch_size', type=int, default=512)  # 128-66.96%/256-66.42%/512-67.55%/如果不好的话可以把学习率调高一点
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--noise_multiplier', type=float, default=1.2890625)#-1.09375、30-1.093878173828125
    parser.add_argument('--max_grad_norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument("-r", "--n-runs", type=int, default=1, metavar="R", help="number of runs to average on", )
    parser.add_argument("--delta", type=float, default=1e-5, metavar="D", help="Target delta", )
    parser.add_argument("--secure_rng", action="store_true", default=False,help="Enable Securdia-smie RNG to have trustworthy privacy guarantees. Comes at a performance cost", )
    parser.add_argument("--device", type=str, default="cuda", help="GPU ID for this process", )
    args = parser.parse_args()

    device = torch.device(args.device)
    train_data, test_data = get_data(args.dataset, augment=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.mini_batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.mini_batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # train_loader = random_sample_loader(train_loader, device, drop_last=True)
    # test_loader = random_sample_loader(test_loader, device)

    run_results = []
    #model = convnet(num_classes=10)
    model = get_nas_model()
    #model = opacus.validators.ModuleValidator.fix(model)
    model.to(device)
    #print(model)
    for _ in range(args.n_runs):
        #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.7276715037905593, weight_decay=0.001)
        #optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08 )#weight_decay=0.001, amsgrad=False
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=1.1)
        privacy_engine = None
        if not args.disable_dp:
            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private(#make_private方法中包含模型、优化器、数据loader；
                #模型也被包装以计算每个样本的梯度。
                #优化器现在负责梯度剪辑和添加噪音的梯度。
                #更新DataLoader以执行泊松采样。
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.noise_multiplier,#高斯噪声的标准差与加噪声的函数l2灵敏度之比(加多少噪声)
                max_grad_norm=args.max_grad_norm,#每个样本梯度的最大范数。任何范数高于此值的梯度都将被剪切到此值。
                #clipping：每个样本的梯度裁剪机制(“平面”或“per_layer”或“自适应”)。平面裁剪计算所有参数的整个梯度的范数，每层裁剪为每个参数张量设置单独的范数，自适应裁剪在每次迭代中更新裁剪界。平面剪辑通常是首选，但是将每层剪辑与分布式训练结合使用可以提供显著的性能提升。
                    )
        #scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=0, last_epoch=-1)
        #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        #scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train(args, model, device, train_loader, optimizer, privacy_engine,lr_scheduler, epoch)
            test_loss, test_acc = mm_test(model, device, test_loader, epoch)
            #scheduler.step()
        #     writer.add_scalar('train/loss', np.mean(train_loss), epoch)
        #     writer.add_scalar('train/acc', train_acc, epoch)
        #     writer.add_scalar('test/loss', test_loss, epoch)
        #     writer.add_scalar('test/acc', test_acc, epoch)
        # writer.close()
        writer.close()
        run_results.append(mm_test(model, device, test_loader,epoch))

        if len(run_results) > 1:
                print(
                    "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                        len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
                    )
                )

if __name__ == '__main__':
    #main()
    model = get_nas_model()
    print(model)

# import nni
# import argparse
# import numpy as np
# import opacus.validators
# import torch
# import torch.nn as nn
# #import nni.nas.nn.pytorch as nn
#
# #import nni.retiarii.nn.pytorch as nn
# #nni.nas.nn.pytorch._layers
#
# #import torch.nn.functional as F
# from nni.retiarii.nn.pytorch.nn import functional as F
# import torch.optim as optim
# from opacus import PrivacyEngine
# from torchvision import datasets, transforms
# from tqdm import tqdm
# from nni.utils import merge_parameter
# from torchvision import transforms
# from nni.retiarii import fixed_arch
# from nni.nas.hub.pytorch import ENAS as EnasSpace
# import random
# import os
# import logging
# #logger = logging.getLogger('cifar_AutoML')
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#
# seed = 3407
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
#
# class PoissonSampler(torch.utils.data.Sampler):
#     def __init__(self, num_examples, batch_size):
#         self.inds = np.arange(num_examples)
#         self.batch_size = batch_size
#         self.num_batches = int(np.ceil(num_examples / batch_size))
#         self.sample_rate = self.batch_size / (1.0 * num_examples)
#         super().__init__(None)
#
#     def __iter__(self):
#         # select each data point independently with probability `sample_rate`
#         for i in range(self.num_batches):
#             batch_idxs = np.random.binomial(n=1, p=self.sample_rate, size=len(self.inds))
#             # if sample number not equal to batch size, privacy engine will report error
#             while np.sum(batch_idxs) != self.batch_size:
#                 batch_idxs = np.random.binomial(n=1, p=self.sample_rate, size=len(self.inds))
#             batch = self.inds[batch_idxs.astype(np.bool)]
#             np.random.shuffle(batch)
#             yield batch
#
#     def __len__(self):
#         return self.num_batches
#
# def get_data(name, augment=False):
#     if name == "cifar10":
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#
#         if augment:
#             train_transforms = [
#                     transforms.RandomHorizontalFlip(),
#                     transforms.RandomCrop(32, 4),
#                     transforms.ToTensor(),
#                     normalize,
#                 ]
#             print('\n', train_transforms, '\n')
#         else:
#             train_transforms = [
#                 transforms.ToTensor(),
#                 normalize,
#                 # transforms.RandomCrop(32, padding=4),
#                 # transforms.RandomHorizontalFlip(),
#                 # transforms.ToTensor(),
#                 # normalize,
#                 # cutout_transform,
#             ]
#
#         train_set = datasets.CIFAR10(root="/home/xuedaxuan/torch_workspace/pt.darts/data", train=True,
#                                      transform=transforms.Compose(train_transforms),
#                                      download=True)
#
#         test_set = datasets.CIFAR10(root="/home/xuedaxuan/torch_workspace/pt.darts/data", train=False,
#                                     transform=transforms.Compose(
#                                         [transforms.ToTensor(), normalize]
#                                     ))
#
#     elif name == "fmnist":
#         train_set = datasets.FashionMNIST(root='./datasets1', train=True,
#                                           transform=transforms.ToTensor(),
#                                           download=True)
#
#         test_set = datasets.FashionMNIST(root='./datasets1', train=False,
#                                          transform=transforms.ToTensor(),
#                                          download=True)
#
#     elif name == "mnist":
#         train_set = datasets.MNIST(root='.data', train=True,
#                                    transform=transforms.ToTensor(),
#                                    download=True)
#
#         test_set = datasets.MNIST(root='.data', train=False,
#                                   transform=transforms.ToTensor(),
#                                   download=True)
#
#     else:
#         raise ValueError(f"unknown dataset {name}")
#
#     return train_set, test_set
#
# def random_sample_loader(loader, device, drop_last=False, sample_batches=False):
#     datas = []
#     targets = []
#
#     for (data, target) in loader:
#         data, target = data.to(device), target.to(device)
#         datas.append(data)
#         targets.append(target)
#
#     datas = torch.cat(datas, axis=0)
#     targets = torch.cat(targets, axis=0)
#     data = torch.utils.data.TensorDataset(datas, targets)
#     if sample_batches:
#         sampler = PoissonSampler(len(datas), loader.batch_size)
#         return torch.utils.data.DataLoader(data, batch_sampler=sampler,
#                                            num_workers=0, pin_memory=False)
#     else:
#         shuffle = isinstance(loader.sampler, torch.utils.data.RandomSampler)
#         return torch.utils.data.DataLoader(data,
#                                            batch_size=loader.batch_size,
#                                            shuffle=shuffle,
#                                            num_workers=0,
#                                            pin_memory=False,
#                                            drop_last=drop_last)
#
# def get_nas_model():
#     model_space = EnasSpace(
#         width=16,  # the initial filters (channel number) for the model
#         num_cells=6,  # the number of stacked cells in total
#         dataset='cifar'  # to give a hint about input resolution, here is 32x32
#     )
#     exported_arch = torch.load('/home/xuedaxuan/torch_workspace/nni210/exported_arch/exported_arch_enas_10OPS_50_GN8_fixAdapt.pth')
#     with fixed_arch(exported_arch):
#         final_model = EnasSpace(width=16, num_cells=6, dataset='cifar')
#
#     return final_model
#
# def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
#     model.train()
#     #交叉熵损失函数用于分类
#     criterion = nn.CrossEntropyLoss()
#     losses = []
#     #enumerate()将一个可遍历的数据对象组合为一个索引序列
#     for _batch_idx, (data, target) in enumerate((train_loader)):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()#清空过往梯度，把loss关于weight的导数变成0
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.item())
#
#     if not args['disable_dp']:
#         epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(delta=args['delta'])
#         print(
#             f"Train Epoch: {epoch} \t"
#             f"Loss: {np.mean(losses):.6f} "
#             f"(ε = {epsilon:.2f}, δ = {args['delta']}) for α = {best_alpha}"
#         )
#     else:
#         print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")
#
# def mm_test(model, device, test_loader):
#     model.eval()#在模型中，我们通常会加上Dropout层和batch normalization层，在模型预测阶段，我们需要将这些层设置到预测模式，model.eval()就是帮我们一键搞定的，如果在预测的时候忘记使用model.eval()，会导致不一致的预测结果。
#     criterion = nn.CrossEntropyLoss()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():#当requires_grad设置为False时,反向传播时就不会自动求导了
#         for data, target in tqdm(test_loader):#tqdm进度条库
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()  # sum up batch loss
#             pred = output.argmax(
#                 dim=1, keepdim=True
#             )  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()#torch.eq()函数就是用来比较对应位置数字，相同则为1，否则为0;torch.eq().sum()就是将所有值相加;
#
#     test_loss /= len(test_loader.dataset)
#     test_acc = 100. * correct / len(test_loader.dataset)
#
#     print(
#         "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
#             test_loss,
#             correct,
#             len(test_loader.dataset),
#             100.0 * correct / len(test_loader.dataset),
#         )
#     )
#     #return correct / len(test_loader.dataset)
#     return test_loss, test_acc
#
# def main(args):
#
#     device = torch.device(args['device'])
#     train_data, test_data = get_data(args['dataset'], augment=False)
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=args['mini_batch_size'], shuffle=True, num_workers=1, pin_memory=True)
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size=args['mini_batch_size'], shuffle=False, num_workers=1, pin_memory=True)
#
#     train_loader = random_sample_loader(train_loader, device, drop_last=True)
#     test_loader = random_sample_loader(test_loader, device)
#
#     run_results = []
#     model = get_nas_model()
#     #model = opacus.validators.ModuleValidator.fix(model)
#     model.to(device)
#     for _ in range(args['n_runs']):
#         optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
#         #optimizer = optim.Adam(model.parameters(), lr=args.lr)
#         privacy_engine = None
#         if not args['disable_dp']:
#             privacy_engine = PrivacyEngine()
#             if args['mini_batch_size'] == 128:
#                 args['noise_multiplier'] = 0.771484375
#             elif args['mini_batch_size'] == 256:
#                 args['noise_multiplier'] = 0.89599609375
#             elif args['mini_batch_size'] == 512:
#                 args['noise_multiplier'] = 1.09375
#             print("==================",args['noise_multiplier'])
#             model, optimizer, train_loader = privacy_engine.make_private(#make_private方法中包含模型、优化器、数据loader；
#                 #模型也被包装以计算每个样本的梯度。
#                 #优化器现在负责梯度剪辑和添加噪音的梯度。
#                 #更新DataLoader以执行泊松采样。
#                 module=model,
#                 optimizer=optimizer,
#                 data_loader=train_loader,
#                 noise_multiplier=args['noise_multiplier'],#高斯噪声的标准差与加噪声的函数l2灵敏度之比(加多少噪声)
#                 max_grad_norm=args['max_grad_norm'],#每个样本梯度的最大范数。任何范数高于此值的梯度都将被剪切到此值。
#                 #clipping：每个样本的梯度裁剪机制(“平面”或“per_layer”或“自适应”)。平面裁剪计算所有参数的整个梯度的范数，每层裁剪为每个参数张量设置单独的范数，自适应裁剪在每次迭代中更新裁剪界。平面剪辑通常是首选，但是将每层剪辑与分布式训练结合使用可以提供显著的性能提升。
#                     )
#         for epoch in range(1, args['epoch'] + 1):
#             train(args, model, device, train_loader, optimizer, privacy_engine, epoch)
#             test_loss, test_acc = mm_test(model, device, test_loader)
#             nni.report_intermediate_result(test_acc)
#             # logger.debug('test accuracy %g', test_acc)
#             # logger.debug('Pipe send intermediate result done.')
#         #run_results.append(mm_test(model, device, test_loader))
#         # if len(run_results) > 1:
#         #         print(
#         #             "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
#         #                 len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
#         #             )
#         #         )
#         nni.report_final_result(test_acc)
#         # logger.debug('Final result is %g', test_acc)
#         # logger.debug('Send final result done.')
#
# def get_params():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--disable-dp", action="store_true", default=False,help="Disable privacy training and just train with vanilla SGD", )
#     parser.add_argument('--dataset', choices=['cifar10', 'fmnist', 'mnist'], default='cifar10')
#     parser.add_argument('--batch_size', type=int, default=512)
#     parser.add_argument('--mini_batch_size', type=int, default=512)  # 128-66.96%/256-66.42%/512-67.55%/如果不好的话可以把学习率调高一点
#     parser.add_argument('--lr', type=float, default=0.1)
#     parser.add_argument('--momentum', type=float, default=0.832434051172931, metavar='M', help='SGD momentum (default: 0.5)')
#     parser.add_argument('--noise_multiplier', type=float, default=1.09375)
#     parser.add_argument('--max_grad_norm', type=float, default=1.0)
#     parser.add_argument('--epoch', type=int, default=30)
#     parser.add_argument("-r", "--n-runs", type=int, default=1, metavar="R", help="number of runs to average on", )
#     parser.add_argument("--delta", type=float, default=1e-5, metavar="D", help="Target delta", )
#     parser.add_argument("--secure_rng", action="store_true", default=False,help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost", )
#     parser.add_argument("--device", type=str, default="cuda", help="GPU ID for this process", )
#     args, _ = parser.parse_known_args()
#     return args
#
# if __name__ == '__main__':
#     try:
#         # get parameters form tuner
#         tuner_params = nni.get_next_parameter()
#         #logger.debug(tuner_params)
#         params = vars(merge_parameter(get_params(), tuner_params))
#         print(params)
#         main(params)
#     except Exception as exception:
#         #logger.exception(exception)
#         raise
