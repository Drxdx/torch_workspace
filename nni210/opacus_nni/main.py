import argparse
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from opacus import PrivacyEngine
from train_utils import get_device, train, test
from data import get_data, random_sample_loader
from nni.retiarii import fixed_arch
import os
from nni.nas.hub.pytorch import ENAS as EnasSpace

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

seed = 1234
cudnn.benchmark = True
cudnn.enabled = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model_space = EnasSpace(
    width=16,           # the initial filters (channel number) for the model
    num_cells=6,        # the number of stacked cells in total
    dataset='cifar'     # to give a hint about input resolution, here is 32x32
)

def main(dataset, batch_size=2048, mini_batch_size=256, lr=1,noise_multiplier=1, max_grad_norm=0.1, epochs=100, max_epsilon=10., delta=1e-5, secure_rng=False):
    exported_arch = torch.load('/home/xuedaxuan/torch_workspace/nni210/exported_arch/exported_arch_enas_8OPS_50.pth')
    with fixed_arch(exported_arch):
        final_model = EnasSpace(width=16, num_cells=6, dataset='cifar')
    #print(final_model)

    device = get_device()
    bs = batch_size
    assert bs % mini_batch_size == 0
    n_acc_steps = bs // mini_batch_size  #n_acc_steps=2048//256=8

    train_data, test_data = get_data(dataset, augment=False)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=mini_batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=mini_batch_size, shuffle=False, num_workers=1, pin_memory=True)
    train_loader = random_sample_loader(train_loader, device, drop_last=True)
    test_loader = random_sample_loader(test_loader, device)

    results_acc = []
    end = []
    for _ in range(2):
        model = final_model
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        # privacy_engine = PrivacyEngine(
        #     module=model,
        #     batch_size=bs,
        #     sample_size=len(train_data),
        #     alphas=ORDERS,
        #     noise_multiplier=noise_multiplier,
        #     max_grad_norm=max_grad_norm,
        # )
        # privacy_engine.attach(optimizer)
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
        )

        results = []
        for epoch in range(0, epochs):
            print(f"\nEpoch: {epoch}")
            train_loss, train_acc, epsilon = train(args, model, train_loader, optimizer, privacy_engine, epoch ,n_acc_steps=n_acc_steps)
            test_loss, test_acc = test(model, test_loader)

            # if noise_multiplier > 0:
            #     rdp_sgd = get_renyi_divergence(
            #         privacy_engine.sample_rate, privacy_engine.noise_multiplier
            #     ) * privacy_engine.steps
            #     epsilon, _ = get_privacy_spent(rdp_sgd)
            #     print(f"Privacy cost: ε = {epsilon:.3f}")
            #     if max_epsilon is not None and epsilon >= max_epsilon:
            #         return
            # else:
            #     epsilon = None
            results.append([epsilon, test_acc])
            results_acc.append(test_acc)
        end.append(max(results_acc))
        results = np.array(results)
        print('\n'+'='*60)
        print('Best test accuracy: %.2f for privacy budget ε = %.2f'%(results[:,1].max(), results[-1, 0]))
        print('='*60)
        print('\n' + '=' * 60)
        print(end)
        print('=' * 60)

    if len(end) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(end), np.mean(end), np.std(end)
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'fmnist', 'mnist'],default='cifar10')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--mini_batch_size', type=int, default=512)#128-66.96%/256-66.42%/512-67.55%/如果不好的话可以把学习率调高一点
    parser.add_argument('--lr', type=float, default=2)
    parser.add_argument('--noise_multiplier', type=float, default=1.8469085693359375)
    parser.add_argument('--max_grad_norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument("--delta",type=float,default=1e-5,metavar="D",help="Target delta",)
    parser.add_argument("--secure_rng",action="store_true",default=False,help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",)
    args = parser.parse_args()
    print(args)
    main(**vars(args))
