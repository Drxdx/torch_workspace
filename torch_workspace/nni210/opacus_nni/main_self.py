import nni
import argparse
import numpy as np
import opacus.validators
import torch
#import torch.nn as nn
import nni.retiarii.nn.pytorch as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
from nni.utils import merge_parameter
import time
from torchvision import transforms
from models import MODELS
from data import get_data, random_sample_loader
from nni.retiarii import fixed_arch
from nni.nas.hub.pytorch import ENAS as EnasSpace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def get_nas_model():
    model_space = EnasSpace(
        width=16,  # the initial filters (channel number) for the model
        num_cells=6,  # the number of stacked cells in total
        dataset='cifar'  # to give a hint about input resolution, here is 32x32
    )
    exported_arch = torch.load('/home/xuedaxuan/torch_workspace/nni210/exported_arch/exported_arch_enas_8OPS_50.pth')
    with fixed_arch(exported_arch):
        final_model = EnasSpace(width=16, num_cells=6, dataset='cifar')

    return final_model

def convnet(num_classes: object) -> object:
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128, num_classes, bias=True),
    )


def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    #交叉熵损失函数用于分类
    criterion = nn.CrossEntropyLoss()
    losses = []
    #enumerate()将一个可遍历的数据对象组合为一个索引序列
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()#清空过往梯度，把loss关于weight的导数变成0
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if not args.disable_dp:
        epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-dp", action="store_true", default=False,help="Disable privacy training and just train with vanilla SGD", )
    parser.add_argument('--dataset', choices=['cifar10', 'fmnist', 'mnist'], default='cifar10')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--mini_batch_size', type=int, default=512)  # 128-66.96%/256-66.42%/512-67.55%/如果不好的话可以把学习率调高一点
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--noise_multiplier', type=float, default=1.09375)
    parser.add_argument('--max_grad_norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument("-r", "--n-runs", type=int, default=1, metavar="R", help="number of runs to average on", )
    parser.add_argument("--delta", type=float, default=1e-5, metavar="D", help="Target delta", )
    parser.add_argument("--secure_rng", action="store_true", default=False,help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost", )
    parser.add_argument("--device", type=str, default="cuda", help="GPU ID for this process", )
    args = parser.parse_args()

    device = torch.device(args.device)
    train_data, test_data = get_data(args.dataset, augment=False)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.mini_batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.mini_batch_size, shuffle=False, num_workers=1, pin_memory=True)
    train_loader = random_sample_loader(train_loader, device, drop_last=True)
    test_loader = random_sample_loader(test_loader, device)

    run_results = []
    #model = convnet(num_classes=10)
    model = get_nas_model()
    #model = opacus.validators.ModuleValidator.fix(final_model)
    model.to(device)
    #_作为一个标识符，当循环的值在下面不会用到，而你又不知道起啥变量名的时候可以用它。
    for _ in range(args.n_runs):
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
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
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, privacy_engine, epoch)
        run_results.append(mm_test(model, device, test_loader))
        if len(run_results) > 1:
                print(
                    "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                        len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
                    )
                )
if __name__ == '__main__':
    main()
