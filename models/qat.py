import os
import json
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import ArgumentParser
from datamodule import CIFAR10DataModule
from mobilenet import mobilenet_v1
from training_module import VisionModel
from quant_util import quantize_model
from brevitas import config
from brevitas.export import export_qonnx


config.IGNORE_MISSING_KEYS = True



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument("--save_dir", default="./checkpoint", type=str, help="Path to the saving directory.")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint.")
    parser.add_argument("--config_path", type=str, help="Path to the quant config file.")
    parser.add_argument("--file_name", type=str, help="Name of the files.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers in the dataloader.")
    parser.add_argument("--seed", type=int, default=32, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for training.")
    parser.add_argument("--scheduler", type=str, default=None, help="Learning rate scheduler.")
    parser.add_argument("--save_onnx", action="store_true", default=False, help="store the JSON file to edit quantization config.")
    args = parser.parse_args()
    
    return args


def main(args):    
    with open(args.config_path, "r") as f:
        quant_config = json.load(f)
        
    # get the data module
    datamodule = CIFAR10DataModule("./data", args.batch_size, 0.0, args.num_workers, args.seed)

    # load the model from checkpoint
    model = mobilenet_v1(10)
    pl_model = VisionModel.load_from_checkpoint(
        args.model_path, 
        map_location="cpu",
        model=model,
        num_classes=10,
        strict=True
    )
    
    # apply quantization
    pl_model.model = quantize_model(pl_model.model, quant_config)
    
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
    
    trainer.fit(model=pl_model, datamodule=datamodule)        
        
    pl_model = VisionModel.load_from_checkpoint(
        top_checkpoint_callback.best_model_path,
        model=pl_model.model,
        num_classes=10,
    )

    perf = trainer.test(pl_model, datamodule=datamodule)
    print(perf)
    
    if args.save_onnx:
        onnx_path = os.path.join(args.save_dir, f"{args.file_name}.onnx")
        x, _ = next(iter(datamodule.test_dataloader()))   
        export_qonnx(
            pl_model.model.cpu(),        
            input_t=x.cpu(),
            export_params=True,
            export_path=onnx_path,
            opset_version=13
        )
        print(f"[*]\tQONNX format exported in {onnx_path}")
        


if __name__ == "__main__":
    args = argument_parser()
    main(args)