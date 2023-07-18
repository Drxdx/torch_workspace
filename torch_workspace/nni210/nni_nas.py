import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
from opacus import PrivacyEngine
from torchvision import models
import argparse
import numpy as np
import torch
#import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import logging
import nni.retiarii.strategy as strategy
search_strategy = strategy.Random(dedup=True)  # dedup=False if deduplication is not wanted
from opacus.validators import ModuleValidator

from opacus.grad_sample import GradSampleModule
import nni
from copy import deepcopy
from collections import OrderedDict
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
#import torch.nn as nn
from nni.retiarii.hub.pytorch import DARTS,ProxylessNAS
from torchvision.datasets import CIFAR10
import os
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

P_model = ProxylessNAS.load_searched_model('acenas-m1', pretrained=True, download=True)
darts_v2_model = DARTS.load_searched_model('darts-v2', pretrained=True, download=True)
resnet181 = ModuleValidator.fix(darts_v2_model, strict=False)
# resnet18 = models.resnet18(num_classes=10)
# resnet181 = ModuleValidator.fix(resnet18, strict=False)

def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    #交叉熵损失函数用于分类
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct = 0
    #enumerate()将一个可遍历的数据对象组合为一个索引序列
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()  # 清空过往梯度，把loss关于weight的导数变成0
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # logger.info('Epoch:[{}]\t loss={:.5f}'.format(epoch, loss))
        losses.append(loss.item())

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()  # torch.eq()函数就是用来比较对应位置数字，相同则为1，否则为0;torch.eq().sum()就是将所有值相加;

        # if epoch == 5:
        #     print("epoch:",epoch)
        #     optimizer.step()
        #writer.add_scalar('train_loss',loss,epoch)

    if not args.disable_dp:
        # rdp = compute_rdp(sample_rate, noise_multiplier, steps, alphas)
        # eps, opt_alpha = get_privacy_spent(alphas, rdp, delta=delta)
        epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}, Accuracy:{100.0 * correct / len(train_loader.dataset)}"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} \t Accuracy:{100.0 * correct / len(train_loader.dataset)}")


def mm_test(model, device, test_loader):
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
    return correct / len(test_loader.dataset)


def main():
    # Training settings batch不一样导致不同的epsilon
    parser = argparse.ArgumentParser(description="Opacus CIFAR Example",formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("-b","--batch-size",type=int,default=128,metavar="B",help="Batch size")
    parser.add_argument("--batch-size-test",type=int,default=256,metavar="TB",help="input batch size for testing",)
    parser.add_argument("-n","--epochs",type=int,default=100,metavar="N",help="number of epochs to train",)
    parser.add_argument("-r","--n-runs",type=int,default=1,metavar="R",help="number of runs to average on",)
    parser.add_argument("--lr",type=float,default=0.1,metavar="LR",help="learning rate",)
    parser.add_argument("--sigma",type=float,default=0.642288589477539,metavar="S",help="Noise multiplier",)
    parser.add_argument("-c","--max-per-sample-grad_norm",type=float,default=0.1,metavar="C",help="Clip per-sample gradients to this norm",)
    parser.add_argument("--delta",type=float,default=1e-5,metavar="D",help="Target delta",)
    parser.add_argument("--device",type=str,default="cuda", help="GPU ID for this process",)
    #触发disable-dp，则为true，否则为false
    parser.add_argument("--save-model", action="store_true",default=False,help="Save the trained model",)
    parser.add_argument("--disable-dp",action="store_true",default=False,help="Disable privacy training and just train with vanilla SGD",)
    parser.add_argument("--secure-rng",action="store_true",default=False,help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",)
    #parser.add_argument("--data-root",type=str,default="../mnist",help="Where MNIST is/will be stored",)
    parser.add_argument("--data-root", type=str, default="/home/xuedaxuan/torch_workspace/pt.darts/data", help="Where CIFAR is/will be stored", )
    parser.add_argument("--sample-rate", default=0.04, type=float, metavar="SR",
                        help="sample rate used for batch construction (default: 0.005)", )
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "cos"], default="cos")
    args = parser.parse_args()
    device = torch.device(args.device)


    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    #train_transform = transforms.Compose(augmentations + normalize if args.disable_dp else normalize)
    train_transform = transforms.Compose(normalize)
    test_transform = transforms.Compose(normalize)


    train_dataset = CIFAR10(root=args.data_root, train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=1,pin_memory=True)

    test_dataset = CIFAR10(root=args.data_root, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=16,shuffle=False,num_workers=1,pin_memory=True)

   # logger.info('start training!')
    run_results = []

    #_作为一个标识符，当循环的值在下面不会用到，而你又不知道起啥变量名的时候可以用它。
    for _ in range(args.n_runs):
        model = resnet181.to(device)
        #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-4)
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
                noise_multiplier=args.sigma,#高斯噪声的标准差与加噪声的函数l2灵敏度之比(加多少噪声)
                max_grad_norm=args.max_per_sample_grad_norm,#每个样本梯度的最大范数。任何范数高于此值的梯度都将被剪切到此值。
                #clipping：每个样本的梯度裁剪机制(“平面”或“per_layer”或“自适应”)。平面裁剪计算所有参数的整个梯度的范数，每层裁剪为每个参数张量设置单独的范数，自适应裁剪在每次迭代中更新裁剪界。平面剪辑通常是首选，但是将每层剪辑与分布式训练结合使用可以提供显著的性能提升。
            )

        for epoch in range(1, args.epochs + 1):

            train(args, model, device, train_loader, optimizer, privacy_engine, epoch)

        run_results.append(mm_test(model, device, test_loader))

    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
            )
        )

if __name__ == "__main__":
    # 方式二：
    main()



