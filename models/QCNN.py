import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import torch
import sys
sys.path += ['.', '..', './', '../']


from hqm.circuits.angleencoding import StronglyEntangledCircuit, BasicEntangledCircuit
from hqm.layers.quanvolution import Quanvolution2D
import pennylane as qml


class HybridQCNN(pl.LightningModule):
    # https://discuss.pennylane.ai/t/issues-installing-and-running-pennylane-lightning-gpu/3612/11

    def __init__(self, in_channels=3, out_dim=10, epochs=0, dataset_size=0):
        super(HybridQCNN, self).__init__()

        self.epochs       = epochs
        self.dataset_size = dataset_size
        self.loss         = torch.nn.CrossEntropyLoss()

        NUM_QUBITS = 4
        NUM_LAYERS = 1
        dev               = qml.device("lightning.qubit", wires=NUM_QUBITS)
        circ              = BasicEntangledCircuit(n_qubits=NUM_QUBITS, n_layers=NUM_LAYERS, dev=dev)
        self.ql           = Quanvolution2D(qcircuit=circ, filters=NUM_QUBITS, kernelsize=2, stride=1, aiframework='torch')

        self.model = torch.nn.Sequential(
            #torch.nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=1, padding=2), 
            torch.nn.ReLU(),
            #torch.nn.AvgPool2d(kernel_size=2, stride=2),
            # Conv Layer 2
            torch.nn.Conv2d(in_channels=NUM_QUBITS, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            #torch.nn.AvgPool2d(kernel_size=2, stride=2),
            # Conv Layer 3
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            # Flatten
            torch.nn.Flatten(),
            # Fully Connected Layer 1
            torch.nn.LazyLinear(out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=64),
            torch.nn.ReLU(),
            # Fully Connected Layer 2
            torch.nn.Linear(in_features=64, out_features=out_dim),
        )
       
    def forward(self, x):

        xq = self.ql(x).to(x.device)
        #print(x.device, xq.device)
        #print(x.shape, xq.shape)
        x_output = self.model(xq)
        
        return x_output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        loss      = self.loss(outputs, labels)
        # For example, log accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = torch.sum(predicted == labels.data).item() / labels.size(0)

        # Logging info
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        
        loss      = self.loss(outputs, labels)
        # For example, log accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = torch.sum(predicted == labels.data).item() / labels.size(0)

        # Logging info
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        
        if batch_idx <=10 :
            bsize = inputs.shape[0]
            pairs = find_a_b_for_c(bsize)[-1]
            fig, axes = plt.subplots(nrows=pairs[0], ncols=pairs[1])
            axes = axes.flatten()
            for i, ax in enumerate(axes):
                img = inputs.cpu().detach().numpy()[i,...]
                img = np.moveaxis(img, 0, -1)
                ax.imshow(img)
                ax.set_title(f'Pr: {np.argmax(outputs.cpu().detach().numpy()[i,...])}\nGt: {labels.cpu().detach().numpy()[i,...]}')
                ax.axis(False)
            
            plt.tight_layout()
            self.logger.experiment.add_figure(f'Val-Prediction-{batch_idx}', plt.gcf(), global_step=self.global_step)
            plt.close()
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        
        loss      = self.loss(outputs, labels)
        # For example, log accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = torch.sum(predicted == labels.data).item() / labels.size(0)

        # Logging info
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)

        bsize = inputs.shape[0]
        pairs = find_a_b_for_c(bsize)[-1]
        fig, axes = plt.subplots(nrows=pairs[0], ncols=pairs[1])
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            img = inputs.cpu().detach().numpy()[i,...]
            img = np.moveaxis(img, 0, -1)
            ax.imshow(img)
            ax.set_title(f'Pr: {np.argmax(outputs.cpu().detach().numpy()[i,...])}\nGt: {labels.cpu().detach().numpy()[i,...]}', global_step=self.global_step)
            ax.axis(False)
        
        plt.tight_layout()
        self.logger.experiment.add_figure(f'Val-Prediction-{batch_idx}', plt.gcf())
        plt.close()

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002)
        num_steps = self.epochs * self.dataset_size
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "step", # step means "batch" here, default: epoch   # New!
                "frequency": 1, # default
            },
        }
    

def find_factors(c):
    factors = []
    for i in range(1, int(c**0.5) + 1):
        if c % i == 0:
            factors.append((i, c // i))
    return factors

def find_a_b_for_c(c):
    factors = find_factors(c)
    valid_pairs = [(a, b) for a, b in factors if a * b == c]
    return valid_pairs
