"""Module implementing the masked classification model"""
import torch
import torchvision.transforms as transforms
import torch.utils.data as torchdata
import numpy as np
import argparse
import collections
import time
import matplotlib.pyplot as plt
import os

import network as simclr_net
from dataset_toybox import ToyboxDataset
import utils


OUTPUT_PATH = "../neuron_ablation/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

TOYBOX_MEAN = (0.3496, 0.4370, 0.5197)
TOYBOX_STD = (0.1623, 0.1897, 0.1776)


class MaskLayer(torch.nn.Module):
    """Class implementing the mask layer for the forward run"""
    
    def __init__(self, fc_size, silenced_neurons):
        super().__init__()
        if silenced_neurons is None:
            silenced_neurons = torch.tensor([])
        self.fc_size = fc_size
        self.silenced_neurons = silenced_neurons
        self.mask = silenced_neurons
        self.mask = self.mask.unsqueeze(0)
        self.mask_multi_hot = torch.zeros(self.mask.size(0), self.fc_size).scatter_(1, self.mask, 1.)
        self.mask_multi_hot = torch.nn.Parameter(data=1.0 - self.mask_multi_hot, requires_grad=False)
        
    def forward(self, x):
        """Forward method"""
        y = x * self.mask_multi_hot
        return y
    
    def get_size(self):
        """Get the number of non-zero elements in layer"""
        return torch.sum(self.mask_multi_hot.data).cpu().numpy()


class MaskedClassifier:
    """Class implementing the masked classification"""
    def __init__(self, args):
        self.args = args
        self.network = simclr_net.SimClRNet(num_classes=12).cuda()
        self.network.load_state_dict(torch.load(self.args["backbone_file"]))
        self.mask_layer = MaskLayer(fc_size=self.network.feat_num, silenced_neurons=None)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=TOYBOX_MEAN, std=TOYBOX_STD)
        ])
        
        self.train_set = ToyboxDataset(root="../data", train=True, transform=transform, split="super", size=224,
                                       fraction=0.5, hypertune=False, rng=np.random.default_rng(0),
                                       interpolate=False)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=512, shuffle=False)

        self.test_set = ToyboxDataset(root="../data", train=False, transform=transform, split="super", size=224,
                                      fraction=0.5, hypertune=False, rng=np.random.default_rng(0),
                                      interpolate=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=512, shuffle=False)
        
        if self.args['debug']:
            print("train loader mean and std: ", utils.online_mean_and_sd(self.train_loader))
            print("test loader mean and std: ", utils.online_mean_and_sd(self.test_loader))

        self.network.freeze_all_params()
        # switch to evaluate mode
        self.network.cuda()
        self.network.eval()

    def get_predictions(self, loader, silenced_neurons):
        """Run data through network and get predictions"""
        targets = []
        preds = []
        
        self.mask_layer = MaskLayer(fc_size=self.network.feat_num, silenced_neurons=silenced_neurons)
        assert len(silenced_neurons) + self.mask_layer.get_size() == self.network.feat_num
        
        self.mask_layer = self.mask_layer.cuda()
        self.mask_layer.eval()
    
        with torch.no_grad():
            for i, (_, images, target) in enumerate(loader):
                images = images.cuda()
            
                # compute predictions
                act = self.network.backbone.forward(images)
                masked_act = self.mask_layer.forward(x=act)
                pred = self.network.classifier_fc(masked_act)
                
                targets.append(target.cpu().numpy())
                preds.append(pred.cpu().numpy())
    
        targets = np.concatenate(targets, axis=0)
        preds = np.concatenate(preds, axis=0)
        accuracy, _ = utils.calc_accuracy(output=torch.tensor(preds), target=torch.tensor(targets))
        return targets, preds, accuracy
    
    def do_forward_run(self, loader, seed):
        """Do forward runs for one seed"""
        random_generator = torch.Generator()
        random_generator.manual_seed(seed.item())
        
        neuron_silencing_order = torch.randperm(self.network.feat_num, generator=random_generator)
        
        accuracies = collections.defaultdict(float)
        
        for ssize in [0, 64, 128, 192, 256, 320, 384, 448, 512]:
            targets, preds, acc = self.get_predictions(loader=loader, silenced_neurons=neuron_silencing_order[:ssize])
            accuracies[ssize] = acc[0].item()
            
        return accuracies
            
    def do_forward_runs(self, seeds):
        """Do forward runs for multiple seeds"""
        all_runs_accuracies = collections.defaultdict(list)
        for seed in seeds:
            start_time = time.time()
            if self.args['train']:
                loader = self.train_loader
            else:
                loader = self.test_loader
            seed_acc = self.do_forward_run(seed=seed, loader=loader)
            for k, val in seed_acc.items():
                all_runs_accuracies[k].append(val)
            print("Time taken for seed {:d} is {:.2f}".format(seed, time.time()-start_time))
            
        if self.args['train']:
            save_name = OUTPUT_PATH + self.args['save_name'] + "_train_accs.pt"
        else:
            save_name = OUTPUT_PATH + self.args['save_name'] + "_test_accs.pt"
        print("Saving accuracies to {}".format(save_name))
        torch.save(all_runs_accuracies, save_name)
            
        for k in all_runs_accuracies.keys():
            accs = all_runs_accuracies[k]
            print(k, np.mean(accs), np.std(accs))
            
        return all_runs_accuracies


def get_parser():
    """Return parser for experiment"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--backbone_file", "-b", required=True, type=str, help="Enter file path for pretrained model..")
    parser.add_argument("--reps", "-reps", default=5, type=int, help="Number of evaluation repetitions")
    parser.add_argument("--debug", "-debug", default=False, action='store_true', help="Use this flag for debugging "
                                                                                      "code...")
    parser.add_argument("--train", "-t", default=False, action='store_true', help="Use flag to get curve for train "
                                                                                  "images...")
    parser.add_argument("--save-name", "-sv", required=True, type=str, help="Enter file path to save the accs...")
    return vars(parser.parse_args())


def plot(accuracies):
    """Plot the graphs"""
    xs = []
    ys = []
    es = []
    for k in accuracies.keys():
        accs = accuracies[k]
        xs.append(k)
        ys.append(np.mean(accs))
        es.append(np.std(accs))
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(xs, ys, yerr=es)
    ax.set_xlabel("Number of units ablated")
    ax.set_ylabel("Accuracy")
    plt.show()


def main():
    """Main method"""
    exp_args = get_parser()
    model = MaskedClassifier(args=exp_args)
    rng = np.random.default_rng(0)
    seeds = rng.integers(0, 65536, exp_args['reps'])
    all_accuracies = model.do_forward_runs(seeds=seeds)
    plot(all_accuracies)


if __name__ == "__main__":
    main()
