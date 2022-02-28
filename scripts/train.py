import argparse
import glob
import json
import os
import random
import subprocess

import numpy as np
import pytorch_lightning as pl
import torch
import yaml

from vortx import collate, data, lightningmodel, utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # if training needs to be resumed from checkpoint,
    # it is helpful to change the seed so that
    # the same data augmentations are not re-used
    pl.seed_everything(config["seed"])
    
    logger = pl.loggers.WandbLogger(project=config["wandb_project_name"], config=config)
    
    subprocess.call(
        [
            "zip",
            "-q",
            os.path.join(logger.experiment.dir, "code.zip"),
            "config.yml",
            *glob.glob("*.py"),
        ]
    )
    
    ckpt_dir = os.path.join(logger.experiment.dir, "ckpts")
    checkpointer = pl.callbacks.ModelCheckpoint(
        save_last=True,
        dirpath=ckpt_dir,
        verbose=True,
        save_top_k=2,
        monitor="val/loss",
    )
    callbacks = [checkpointer, lightningmodel.FineTuning(config["initial_epochs"])]
    
    if config["use_amp"]:
        amp_kwargs = {"precision": 16}
    else:
        amp_kwargs = {}
    
    model = lightningmodel.LightningModel(config)
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        benchmark=True,
        max_epochs=config["initial_epochs"] + config["finetune_epochs"],
        check_val_every_n_epoch=5,
        detect_anomaly=True,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,  # a hack so batch size can be adjusted for fine tuning
        **amp_kwargs,
    )
    trainer.fit(model, ckpt_path=config["ckpt"])
