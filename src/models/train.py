import LightningCrmCNN as LightningCrmCNN 
import CrmCNN as CrmCNN
import MMapDataset as MMapDataset
import pytorch_lightning as pl

data = 

model = LightningCrmCNN(CrmCNN())
trainer = pl.Trainer()
trainer.fit(model, train_dataloaders=train_loader)