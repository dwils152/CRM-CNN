import pytorch_lightning as pl
from torch import optim, nn, squeeze, unsqueeze

class LightningCrmCNN(pl.LightningModule):
    def __init__(self, crm_cnn):
        super().__init__()
        self.crm_cnn = crm_cnn

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # Make the inputs and labels the correct dimensions
        inputs = unsqueeze(inputs, 1).float()
        labels = squeeze(labels).float()
        predictions = self.crm_cnn(inputs)
        predictions = squeeze(predictions) 
        loss = nn.BCELoss(predictions, labels)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        # Make the inputs and labels the correct dimensions
        inputs = unsqueeze(inputs, 1).float()
        labels = squeeze(labels).float()
        predictions = self.crm_cnn(inputs)
        predictions = squeeze(predictions) 
        val_loss = nn.BCELoss(predictions, labels)
        self.log('val_loss', val_loss)
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        # Make the inputs and labels the correct dimensions
        inputs = unsqueeze(inputs, 1).float()
        labels = squeeze(labels).float()
        predictions = self.crm_cnn(inputs)
        predictions = squeeze(predictions) 
        test_loss = nn.BCELoss(predictions, labels)
        self.log('test_loss', test_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer