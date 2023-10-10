import pytorch_lightning as pl
from torch import nn, unsqueeze, squeeze
import torch.optim as optim

class LightningCrmCNN(pl.LightningModule):
    def __init__(self, crm_cnn):
        super().__init__()
        self.crm_cnn = crm_cnn
        self.criterion = nn.BCELoss()

    def _shared_step(self, batch, stage):
        inputs, labels = batch
        inputs = unsqueeze(inputs, 1).float()
        labels = squeeze(labels).float()
        predictions = self.crm_cnn(inputs)
        predictions = squeeze(predictions)
        loss = self.criterion(predictions, labels)
        self.log(f'{stage}_loss', loss, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        self._shared_step(batch, 'test')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.trainer.model.parameters(), lr=1e-3)
        return optimizer
