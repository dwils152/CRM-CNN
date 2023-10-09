import pytorch_lightning as pl
from torch import optim, nn

class LightningCrmCNN(pl.LightningModule):
    def __init__(self, crm_cnn):
        super().__init__()
        self.crm_cnn = crm_cnn

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        # Make the inputs and labels the correct dimensions
        inputs = torch.unsqueeze(inputs, 1).float().to(device)
        labels = torch.squeeze(labels).float().to(device)

        predictions = self.crm_cnn(inputs)
        predictions = torch.squeeze(predictions) 

        loss = nn.BCELoss(predictions, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer