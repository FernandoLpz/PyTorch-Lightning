import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl

class NeuralNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(30, 16)
        self.layer_2 = nn.Linear(16, 1)

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.sigmoid(x)

        return x.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Gets "x" and "y" tensors for current batch
        x, y = batch
        
        # Feed the model and catch the prediction
        y_hat = self.forward(x)
        
        # Calculates loss
        loss = F.binary_cross_entropy(y_hat, y)
        
        # Calculates accuracy for current batch
        train_acc_batch = self.train_accuracy(y_hat, y)

        # Save metrics for current batch
        self.log('train_acc_batch', train_acc_batch)
        self.log('train_loss_batch', loss)
        
        return {'loss' : loss, 'preds' : y_hat, 'target' : y}
    
    def training_epoch_end(self, outputs):
        # This function recevies as parameters the output from "training_step()"
        # Outputs is a list which contains a dictionary like: 
        # [{'pred':x,'target':x,'loss':x}, {'pred':x,'target':x,'loss':x}, ...]

        # Accuracy per Epoch Option 1
        acc1 = 0
        for out in outputs:
            acc1 += self.train_accuracy(out['preds'], out['target'])
        acc1 = acc1 / len(outputs)
        print(f"\nOption (1): Train Acc: {acc1}")

        # Accuracy per Epoch Option 2
        acc2 = self.train_accuracy.compute()
        print(f"\nOption (2): Train Acc: {acc2}")

        self.log('Train_acc_epoch', acc2)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy(y_hat, y)

        self.log('val_loss', loss)
        val_acc = self.val_accuracy(y_hat, y)
        # print(f"Val acc: {val_acc}")
        return {'loss' : loss, 'preds' : y_hat, 'target' : y}

    def validation_epoch_end(self, outputs):
        outputs = outputs[0]
        acc = self.val_accuracy(outputs['preds'], outputs['target'])
        print(f"Validation Acc: {acc}")

        self.log('metric', acc)
        # self.log('val_acc_epoch', self.val_accuracy.compute())



if __name__ == "__main__":
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)

    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.FloatTensor)

    x_val = torch.from_numpy(x_val).type(torch.FloatTensor)
    y_val = torch.from_numpy(y_val).type(torch.FloatTensor)
    
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=5)

    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=5)

    nn = NeuralNet()
    trainer = pl.Trainer()
    trainer.fit(nn, train_dataloader, val_dataloader)