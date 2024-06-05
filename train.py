"""
 ------------------------------------------------------------------------
 Modified from HumanSD (https://github.com/IDEA-Research/HumanSD/tree/main)
 ------------------------------------------------------------------------
"""

import json
import cv2
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import Dataset
import os
import argparse
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from PIL import Image
from ldm.util import instantiate_from_config, load_model_from_config
from cldm.utils import ImageLogger, CUDACallback, save_configs, load_state_dict
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
        
def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", 
                        type=str, 
                        help="path to config which constructs model")

    parser.add_argument("--max_epochs",
                        type=int, 
                        default=10,
                        help="how many samples to produce for each given prompt. A.k.a. batch size")
    
    parser.add_argument("--devices",
                        type=int, 
                        default=1,
                        help="how many gpus to train on")
    
    parser.add_argument("-r",
                        "--resume",
                        type=str,
                        nargs="?",
                        const=True,
                        help="resume from checkpoint")
    
    parser.add_argument("-s",
                        "--seed",
                        type=int,
                        default=23,
                        help="seed for seed_everything")
    
    parser.add_argument("--log_frequency",
                        type=int,
                        default=300,
                        help="log images every certain steps")
    
    parser.add_argument("--scale_lr",
                        type=str2bool,
                        nargs="?",
                        const=True,
                        default=True,
                        help="scale base-lr by ngpu * batch_size * n_accumulate")
    
    # argument for ControlNet only
    parser.add_argument("--control_ckpt", 
                        type=str, 
                        default=None,
                        help="path to the pre-generated model, please see tool_add_control.py in https://github.com/lllyasviel/ControlNet/tree/main")
    parser.add_argument("--sd_locked", 
                        default=True,
                        type=str2bool, 
                        help="freeze SD decoder layers")
    parser.add_argument("--only_mid_control", 
                        default=False,
                        type=str2bool, 
                        help="output of controlnet is only added to middle SD block")
    parser.add_argument("--config_metrics", 
                        type=str, 
                        default="utils/metrics/mini_metrics.yaml",
                        help="path to config evaluation metrics, used in validation step")
    
    
    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(opt.config)
    run_name, model_name = config.name, opt.config.split('/')[-2]
    print(f'training model {model_name}')
    if not os.path.exists(os.path.join('experiments', model_name)):
        os.mkdir(os.path.join('experiments', model_name))
    
    # Configs  
    max_epochs = opt.max_epochs
    logger_freq = opt.log_frequency
    batch_size = config.data.params.batch_size
    learning_rate = config.model.learning_rate
    lightning_config = config.pop("lightning", OmegaConf.create())
    
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["accelerator"] = "gpu"
    trainer_config["max_epochs"] = max_epochs
    trainer_config["devices"] = opt.devices
    # check if resume
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        trainer_config["resume_from_checkpoint"] = opt.resume
    trainer_opt = argparse.Namespace(**trainer_config)

    # define model
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = instantiate_from_config(config.model).cpu()
    assert opt.control_ckpt is not None, 'please specify the control_ckpt argument, see tool_add_control.py in https://github.com/lllyasviel/ControlNet/tree/main'
    m, u = model.load_state_dict(load_state_dict(opt.control_ckpt, location='cpu'), strict=False)
    if len(m) > 0:
        print("missing keys:")
        print(m)
    if len(u) > 0:
        print("unexpected keys:")
        print(u)
    model.sd_locked = opt.sd_locked
    model.only_mid_control = opt.only_mid_control
    model.learning_rate = learning_rate
    
    # define dataset
    train_set = instantiate_from_config(config.data.params.train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # the save directory already exists
    if os.path.exists(os.path.join('experiments', model_name, run_name)) and not opt.resume:
        import time
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
        run_name = run_name + '_' + current_time
        print(f'Warnning: Run name already exists in experiments! Add time: {current_time} to run name.')
    output_path = os.path.join('experiments', model_name, run_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    # save missing/unexpected keys
    unloaded_keys = {'missing_keys':m, 'unexpected_keys':u}
    save_json_path = os.path.join(output_path, 'unloaded_keys.json')
    with open(save_json_path, "w") as outfile: 
        json.dump(unloaded_keys, outfile)
            
    # define callbacks
    # checkpoints callback
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": output_path,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        }
    }
    
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3
    
    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg =  OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    
    # other callbacks
    default_callbacks_cfg = {
        "img_logger": {
            "target": "train.ImageLogger",
            "params": {
                "batch_frequency": logger_freq,
                "run_name": run_name
            }
        },
        "cuda_callback": {
            "target": "train.CUDACallback"
        },
        "learning_rate_logger": {
            "target": "train.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            }
        },
    }
    default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})
    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    
    if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
        print(
            'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint':
                {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                    'params': {
                        "dirpath": os.path.join(output_path, 'trainstep_checkpoints'),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        'save_top_k': -1,
                        'every_n_train_steps': 50000,
                        'save_weights_only': True
                    }
                    }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)
        
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    callbacks = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    
    # define trainer 
    trainer_kwargs = dict()
    tb_logger = TensorBoardLogger(os.path.join("experiments", model_name), name=run_name)
    trainer_kwargs["logger"] = tb_logger
    trainer_kwargs["callbacks"] = callbacks
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    
    
    # configure learning rate
    if opt.scale_lr:
        bs, base_lr = config.data.params.batch_size, config.model.learning_rate
        ngpu = opt.devices
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        model.learning_rate = config.model.learning_rate
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")
        
    # Train!
    save_configs([config, callbacks_cfg], output_path, opt.config)
    
    trainer.fit(model, train_loader)
    print('training done.')
    
    # save dict
    print(f'saving model to {output_path}')
    torch.save(model.state_dict(), os.path.join(output_path, 'final.pth'))

if __name__ == "__main__":
    main()