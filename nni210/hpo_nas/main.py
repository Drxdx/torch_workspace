import argparse
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
from data import get_data, random_sample_loader
import os
from nni.utils import merge_parameter
import nni
import logging
from nni.retiarii.hub.pytorch import ENAS as EnasSpace
from nni.retiarii import fixed_arch
logger = logging.getLogger('cifar_AutoML')
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

seed = 1234
cudnn.benchmark = True
cudnn.enabled = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def get_device():
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def train(args, model, train_loader, optimizer, epoch, n_acc_steps=1):
    device = next(model.parameters()).device
    model.train()
    num_examples = 0
    correct = 0
    train_loss = 0

    rem = len(train_loader) % n_acc_steps
    num_batches = len(train_loader)
    num_batches -= rem

    bs = train_loader.batch_size if train_loader.batch_size is not None else train_loader.batch_sampler.batch_size
    print(f"training on {num_batches} batches of size {bs}")

    for batch_idx, (data, target) in enumerate(train_loader):

        if batch_idx > num_batches - 1:
            break

        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                # accumulate per-example gradients but don't take a step yet
                optimizer.virtual_step()

        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += F.cross_entropy(output, target, reduction='sum').item()
        num_examples += len(data)

    train_loss /= num_examples
    train_acc = 100. * correct / num_examples

    # print(f'Train set: Average loss: {train_loss:.4f}, '
    #         f'Accuracy: {correct}/{num_examples} ({train_acc:.2f}%)')

    return train_loss, train_acc

def test(args, model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(data)

    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))

    # print(f'Test set: Average loss: {test_loss:.4f}, '
    #       f'Accuracy: {correct}/{num_examples} ({test_acc:.2f}%)')

    return test_loss, test_acc

def main(args):
    device = get_device()

    bs = args['batch_size']
    assert bs % args['mini_batch_size'] == 0
    n_acc_steps = bs // args['mini_batch_size']  # n_acc_steps=2048//256=8

    train_data, test_data = get_data(args['dataset'], augment=False)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args['mini_batch_size'], shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['mini_batch_size'], shuffle=False, num_workers=1, pin_memory=True)
    train_loader = random_sample_loader(train_loader, device, drop_last=True)
    test_loader = random_sample_loader(test_loader, device)

    results_acc = []
    end = []
    for _ in range(1):

        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'],momentum=args['momentum'])
        privacy_engine = PrivacyEngine(
            module=model,
            batch_size=bs,
            sample_size=len(train_data),
            alphas=ORDERS,
            noise_multiplier=args['noise_multiplier'],
            max_grad_norm=args['max_grad_norm'],
        )
        privacy_engine.attach(optimizer)

        results = []
        for epoch in range(0, args['epochs']):
            print(f"\nEpoch: {epoch}")
            train(args, model, train_loader, optimizer,epoch, n_acc_steps=n_acc_steps )
            test_loss, test_acc = test(args, model, test_loader)

            nni.report_intermediate_result(test_acc)
            logger.debug('test accuracy %g', test_acc)
            logger.debug('Pipe send intermediate result done.')

            if args['noise_multiplier'] > 0:
                rdp_sgd = get_renyi_divergence(
                    privacy_engine.sample_rate, privacy_engine.noise_multiplier
                ) * privacy_engine.steps
                epsilon, _ = get_privacy_spent(rdp_sgd)
                print(f"Privacy cost: ε = {epsilon:.3f}")
                # if 10 is not None and epsilon >= 10:
                #     return
            else:
                epsilon = None
            results.append([epsilon, test_acc])
            results_acc.append(test_acc)
        end.append(max(results_acc))
        results = np.array(results)

        nni.report_final_result(test_acc)
        logger.debug('Final result is %g', test_acc)
        logger.debug('Send final result done.')
        # print('\n' + '=' * 60)
        # print('Best test accuracy: %.2f for privacy budget ε = %.2f' % (results[:, 1].max(), results[-1, 0]))
        # print('=' * 60)
        # print('\n' + '=' * 60)
        # print(end)
        # print('=' * 60)

    if len(end) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(end), np.mean(end), np.std(end)
            )
        )


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'fmnist', 'mnist'], default='cifar10')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--mini_batch_size', type=int, default=512)  # 128-66.96%/256-66.42%/512-67.55%/如果不好的话可以把学习率调高一点
    parser.add_argument('--lr', type=float, default=2.1)
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N',help='how many batches to wait before logging training status')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--noise_multiplier', type=float, default=1.8469085693359375)
    parser.add_argument('--max_grad_norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=30)
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise