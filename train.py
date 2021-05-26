#!/usr/bin/env python3

from model import ErrorChecker, ErrorCheckerDataModule

import logging
import argparse
import os
import warnings

import pytorch_lightning as pl

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ErrorCheckerDataModule.add_argparse_args(parser)
    parser = ErrorChecker.add_model_specific_args(parser)

    parser.add_argument("--model_name", type=str, default="roberta-base",
        help="Name of the model from the Huggingface Transformers library.")
    parser.add_argument("--model_path", type=str, default=None,
        help="Path to the saved checkpoint to be loaded.")
    parser.add_argument("--dataset", type=str, required=True,
        help="Dataset name.")
    parser.add_argument("--batch_size", type=int, default=16,
        help="Batch size for finetuning the model")
    parser.add_argument("--output_dir", type=str, default="experiments",
        help="Output directory")
    parser.add_argument("--experiment", type=str, required=True,
        help="Experiment name used for naming the experiment directory")
    parser.add_argument("--max_length", type=int, default=512,
        help="Maximum number of tokens per example")
    parser.add_argument("--seed", default=42, type=int,
        help="Random seed.")
    parser.add_argument("--max_threads", default=8, type=int,
        help="Maximum number of CPU threads.")
    
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    logger.info("Initializing...")
    logger.info(args)

    pl.seed_everything(args.seed)
    dm = ErrorCheckerDataModule(args)
    dm.prepare_data()
    dm.setup('fit')

    model = ErrorChecker(args)

    ckpt_output_dir = os.path.join(args.output_dir,
        args.experiment
    )

    monitor = "loss/val"
    mode = "min"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_output_dir,
        save_top_k=1,
        verbose=True,
        monitor=monitor,
        mode=mode
    )

    trainer = pl.Trainer.from_argparse_args(args, 
        callbacks=[checkpoint_callback], accelerator='dp')
    trainer.fit(model, dm)
