import pytorch_lightning as pl
from typing import Optional
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torchmetrics import Accuracy


class VisionModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        scheduler: Optional[str] = None,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.model = model
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss()

        self.train_top1_acc = Accuracy(task="multiclass", top_k=1, num_classes=num_classes)
        self.train_top5_acc = Accuracy(task="multiclass", top_k=5, num_classes=num_classes)
        self.val_top1_acc = Accuracy(task="multiclass", top_k=1, num_classes=num_classes)
        self.val_top5_acc = Accuracy(task="multiclass", top_k=5, num_classes=num_classes)
        self.test_top1_acc = Accuracy(task="multiclass", top_k=1, num_classes=num_classes)
        self.test_top5_acc = Accuracy(task="multiclass", top_k=5, num_classes=num_classes)


    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.learning_rate)

        if self.scheduler == "step":
            scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
        elif self.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min")
        elif self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=200)
        elif self.scheduler == "one_cycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=1,
                three_phase=True,
            )
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "name": "learning_rate",
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
            },
        }


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.train_top1_acc(y_hat, y)
        self.train_top5_acc(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_top1_acc", self.train_top1_acc, prog_bar=True)
        self.log("train_top5_acc", self.train_top5_acc)

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.val_top1_acc(y_hat, y)
        self.val_top5_acc(y_hat, y)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_top1_acc", self.val_top1_acc, prog_bar=True, sync_dist=True)
        self.log("val_top5_acc", self.val_top5_acc, sync_dist=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.test_top1_acc(y_hat, y)
        self.test_top5_acc(y_hat, y)

        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        self.log("test_top1_acc", self.test_top1_acc, prog_bar=True, sync_dist=True)
        self.log("test_top5_acc", self.test_top5_acc, sync_dist=True)
