import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
from opacus import PrivacyEngine


@model_wrapper      # this decorator should be put on the out most
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        #self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv2 = nn.Sequential([
            nn.Conv2d(32, 64, 3, 1),
            DepthwiseSeparableConv(32, 64)
        ])
        #self.dropout1 = nn.Dropout(0.25)
        self.dropout1 = nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75]))
        self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)
        feature = nn.ValueChoice([64, 128, 256])
        self.fc1 = nn.Linear(9216, feature)
        self.fc2 = nn.Linear(feature, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
        output = F.log_softmax(x, dim=1)
        return output


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


@model_wrapper
class ModelSpace(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # LayerChoice is used to select a layer between Conv2d and DwConv.
        self.conv2 = nn.LayerChoice([
            nn.Conv2d(32, 64, 3, 1),
            DepthwiseSeparableConv(32, 64)
        ])
        # ValueChoice is used to select a dropout rate.
        # ValueChoice can be used as parameter of modules wrapped in `nni.retiarii.nn.pytorch`
        # or customized modules wrapped with `@basic_unit`.
        self.dropout1 = nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75]))  # choose dropout rate from 0.25, 0.5 and 0.75
        self.dropout2 = nn.Dropout(0.5)
        feature = nn.ValueChoice([64, 128, 256])
        self.fc1 = nn.Linear(9216, feature)
        self.fc2 = nn.Linear(feature, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
        output = F.log_softmax(x, dim=1)
        return output


model_space = ModelSpace()
# print(model_space)

import nni.retiarii.strategy as strategy
search_strategy = strategy.Random(dedup=True)  # dedup=False if deduplication is not wanted


import nni

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from nni.retiarii.hub.pytorch import DARTS as DartsSpace
darts_v2_model = DartsSpace.load_searched_model('darts-v2', pretrained=True, download=True)
def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), accuracy))

    return accuracy


def evaluate_model(model_cls):
    # "model_cls" is a class, need to instantiate
    model = model_cls()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(MNIST('data/mnist', download=True, transform=transf), batch_size=64, shuffle=True)
    test_loader = DataLoader(MNIST('data/mnist', download=True, train=False, transform=transf), batch_size=64)

    privacy_engine= PrivacyEngine()
    model1, optimizer1, train_loader1 = privacy_engine.make_private(  # make_private方法中包含模型、优化器、数据loader；
        # 模型也被包装以计算每个样本的梯度。
        # 优化器现在负责梯度剪辑和添加噪音的梯度。
        # 更新DataLoader以执行泊松采样。
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1.03,  # 高斯噪声的标准差与加噪声的函数l2灵敏度之比(加多少噪声)
        max_grad_norm=0.1,  # 每个样本梯度的最大范数。任何范数高于此值的梯度都将被剪切到此值。
        # clipping：每个样本的梯度裁剪机制(“平面”或“per_layer”或“自适应”)。平面裁剪计算所有参数的整个梯度的范数，每层裁剪为每个参数张量设置单独的范数，自适应裁剪在每次迭代中更新裁剪界。平面剪辑通常是首选，但是将每层剪辑与分布式训练结合使用可以提供显著的性能提升。
    )

    for epoch in range(10):
        # train the model for one epoch
        train_epoch(model1, device, train_loader1, optimizer1, epoch)
        # test the model for one epoch
        accuracy = test_epoch(model, device, test_loader)
        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(accuracy)

    # report final test result
    nni.report_final_result(accuracy)

from nni.retiarii.evaluator import FunctionalEvaluator
evaluator = FunctionalEvaluator(evaluate_model)

from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'nni_dp'
exp_config.max_trial_number = 4   # 最多运行 4 个实验
exp_config.trial_concurrency = 2  # 最多同时运行 2 个试验
exp_config.trial_gpu_number = 1
exp_config.training_service.use_active_gpu = True
exp.run(exp_config, 8081)

