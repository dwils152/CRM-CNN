import pytorch_lightning as pl
from torch import nn, unsqueeze, squeeze
import torch.optim as optim
from torchmetrics import Accuracy, AUROC, FBetaScore

class LightningCrmCNN(pl.LightningModule):
    def __init__(self, crm_cnn):
        super().__init__()
        self.crm_cnn = crm_cnn
        self.criterion = nn.BCELoss()
        self.accuracy = Accuracy(task='binary')
        self.auroc = AUROC(task='binary')
        self.fbeta = FBetaScore(task='binary', beta=1)

    def _compute_metrics(self, batch, stage):
        inputs, labels = batch
        inputs = unsqueeze(inputs, 1).float()
        labels = squeeze(labels).float()
        predictions = self.crm_cnn(inputs)
        predictions = squeeze(predictions)
        
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        auroc = self.auroc(predictions, labels)
        fbeta = self.fbeta(predictions, labels)
        
        self.log(f'{stage}_loss', loss, sync_dist=True, on_step=True, on_epoch=True)
        self.log(f'{stage}_accuracy', accuracy, sync_dist=True, on_step=True, on_epoch=True)
        self.log(f'{stage}_auroc', auroc, sync_dist=True, on_step=True, on_epoch=True)
        self.log(f'{stage}_fbeta', fbeta, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_metrics(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        self._compute_metrics(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        self._compute_metrics(batch, 'test')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.trainer.model.parameters(), lr=1e-3)
        return optimizer
