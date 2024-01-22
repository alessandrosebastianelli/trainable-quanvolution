from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
import sys
import os

sys.path += ['.', './']

from models.QCNN import HybridQCNN
from dataio.loader import EuroSATDataModule



if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    data_module = EuroSATDataModule(num_workers=8, batch_size=1)

    tb_logger = pl.loggers.TensorBoardLogger(os.path.join('lightning_logs','classifiers'), name='EuroSATClassifier')

    # Instantiate ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('saved_models','classifiers'),
        filename='EuroSATClassifier',
        monitor='val_loss',
        save_top_k=1,
        mode='min',
    )

    # Instantiate LightningModule and DataModule
    model = HybridQCNN(in_channels=3, out_dim=10, epochs=30, dataset_size=21600)

    # Instantiate Trainer
    trainer = pl.Trainer(max_epochs=30, callbacks=[checkpoint_callback], logger=tb_logger)

    # Train the model
    trainer.fit(model, data_module)
