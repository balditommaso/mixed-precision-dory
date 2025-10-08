import os
import torch
import json
import pytorch_lightning as pl
import arch as ARCHS
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import ArgumentParser
from datamodule import DATALOADERS
from training_module import VisionModel
from quant_util import extract_modules

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument("--save_dir", default="./checkpoint", type=str, help="Path to the saving directory.")
    parser.add_argument("--dataset", default="cifar-10", choices=["cifar-10", "MNIST"], type=str, help="Dataset for training.")
    parser.add_argument("--model", default="dummy_cnn", choices=["dummy_cnn", "mobilenet_v1"], type=str, help="Model to be trained.")
    parser.add_argument("--file_name", default="mobilenetV1", type=str, help="Name of the files.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers in the dataloader.")
    parser.add_argument("--seed", type=int, default=32, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for training.")
    parser.add_argument("--scheduler", type=str, default=None, help="Learning rate scheduler.")
    parser.add_argument("--save_json", action="store_true", default=False, help="store the JSON file to edit quantization config.")
    args = parser.parse_args()
    
    return args


def main(args):
    # activate the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    
    # get the data module
    datamodule = DATALOADERS[args.dataset]("./data", args.batch_size, 0.0, args.num_workers, args.seed)

    # get the model
    model = getattr(ARCHS, args.model)(10)
    
    # log the training
    tb_logger = pl_loggers.TensorBoardLogger(args.save_dir, name=f"{args.file_name}_logs")
    
    # monitor the learning rate and the scheduler
    callbacks = [LearningRateMonitor("epoch")]
    
    # save top checkpoints based on val_loss
    top_checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename=args.file_name,
        monitor="train_loss",
        mode="min",
        auto_insert_metric_name=False
    )
    callbacks.append(top_checkpoint_callback)
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices="auto",
        logger=tb_logger,
        callbacks=callbacks,
    )
    pl_model = VisionModel(model, 10, args.lr, args.scheduler)
    
    trainer.fit(model=pl_model, datamodule=datamodule)        
        
    pl_model = VisionModel.load_from_checkpoint(
        top_checkpoint_callback.best_model_path,
        model=model,
        num_classes=10,
        strict=True
    )

    perf = trainer.test(pl_model, datamodule=datamodule)
    print(perf)
    
    if args.save_json:
        quant_config = extract_modules(pl_model.model)
        path = os.path.join(args.save_dir, f"{args.file_name}_config.json")
        with open(path, "w") as f:
            json.dump(quant_config, f, indent=4)
        print(f"[*]\t Config JSON file stored in {path}")
        


if __name__ == "__main__":
    args = argument_parser()
    main(args)