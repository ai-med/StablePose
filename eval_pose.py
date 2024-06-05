"""
 ------------------------------------------------------------------------
 Modified from HumanSD (https://github.com/IDEA-Research/HumanSD/tree/main)
 ------------------------------------------------------------------------
"""

import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from einops import rearrange

from ldm.util import instantiate_from_config, load_model_from_config, log_txt_as_img
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from cldm.utils import load_state_dict

from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
import einops
import cv2
import json

def load_model_from_ckpt(config, ckpt):
    print(f"Loading model from {ckpt}")
    sd = load_state_dict(ckpt)
    model = instantiate_from_config(config)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0:
        print("missing keys:")
        print(m)
    if len(u) > 0:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model

def draw_humansd_skeleton(image, present_pose, keypoint_thresh):
    humansd_skeleton=[
              [0,0,1],
              [1,0,2],
              [2,1,3],
              [3,2,4],
              [4,3,5],
              [5,4,6],
              [6,5,7],
              [7,6,8],
              [8,7,9],
              [9,8,10],
              [10,5,11],
              [11,6,12],
              [12,11,13],
              [13,12,14],
              [14,13,15],
              [15,14,16],
          ]
    humansd_skeleton_width=10
    humansd_color=sns.color_palette("hls", len(humansd_skeleton)) 
    
    def plot_kpts(img_draw, kpts, color, edgs, width):
        for idx, kpta, kptb in edgs:
            if kpts[kpta,2]>keypoint_thresh and \
                kpts[kptb,2]>keypoint_thresh :
                line_color = tuple([int(255*color_i) for color_i in color[idx]])
                cv2.line(img_draw, (int(kpts[kpta,0]),int(kpts[kpta,1])), (int(kpts[kptb,0]),int(kpts[kptb,1])), line_color,width)
                cv2.circle(img_draw, (int(kpts[kpta,0]),int(kpts[kpta,1])), width//2, line_color, -1)
                cv2.circle(img_draw, (int(kpts[kptb,0]),int(kpts[kptb,1])), width//2, line_color, -1)
    
    pose_image = np.zeros_like(image)
    for person_i in range(present_pose.shape[0]):
        if torch.sum(present_pose[person_i,:,:])>0:
            plot_kpts(pose_image, present_pose[person_i,:,:],humansd_color,humansd_skeleton,humansd_skeleton_width)

    return pose_image


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images



def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x
    
def gen_pose_mask(gaussian_kernels, batch, batch_size):
    '''
    Input: a list of Gaussian kernel sizes
    Return: pose masks generated from the Gaussian kernels
    ------------------------------------------------------
    Generate pose-mask for our proposed ViT in evaluation. The input specifies a list of Gaussian kernel sizes, 
    e.g. [23, 13], that are used to generate pose masks.
    '''
    from torchvision.transforms import GaussianBlur
    import torch.nn.functional as F
    pose_condition = batch['hint']
    pose_condition = einops.rearrange(pose_condition, 'b h w c -> b c h w')
    pose_condition = pose_condition.to(memory_format=torch.contiguous_format).float()

    pose_image = torch.cat([pose_condition], 1)
    pose_masks = []
    for k in gaussian_kernels:
        masks = torch.zeros((batch_size, 64,64))
        masks.requires_grad = False
        blur = GaussianBlur(kernel_size=(k, k), sigma=3)
        pose_image_blured = blur(pose_image)
        for i, pose in enumerate(pose_image_blured):
            _, h_idx, w_idx = torch.where(pose>-0.99)
            h_idx, w_idx = h_idx//8, w_idx//8
            masks[i][h_idx, w_idx] = 1
        # no pose mask for unconditional inputs
        pose_masks.append(torch.cat((torch.ones_like(masks), masks), dim=0))
    return torch.cat(tuple(pose_masks), dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", 
                        default=["pose","quality","text"])
    parser.add_argument("--ddim_steps", 
                        type=int, 
                        default=50,
                        help="number of ddim sampling steps")
    parser.add_argument( "--plms", 
                        action='store_true',
                        help="use plms sampling")
    parser.add_argument("--dpm_solver", 
                        action='store_true',
                        help="use dpm_solver sampling")
    parser.add_argument("--ddim_eta", 
                        type=float, 
                        default=0.0,
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--H", 
                        type=int, 
                        default=512,
                        help="image height, in pixel space")
    parser.add_argument("--W", 
                        type=int, 
                        default=512,
                        help="image width, in pixel space")
    parser.add_argument("--C", 
                        type=int, 
                        default=4,
                        help="latent channels")
    parser.add_argument( "--f", 
                        type=int, 
                        default=8,
                        help="downsampling factor")
    parser.add_argument("--batch_size",
                        type=int, 
                        default=4,
                        help="how many samples to produce for each given prompt. A.k.a. batch size")
    parser.add_argument("--scale", 
                        type=float, 
                        default=9,
                        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    parser.add_argument("--config_model", 
                        type=str, 
                        help="path to config which constructs model")
    parser.add_argument("--config_metrics", 
                        type=str, 
                        default="utils/metrics/metrics.yaml",
                        help="path to config evaluation metrics")
    parser.add_argument("--ckpt", 
                        type=str, 
                        help="path to checkpoint of trained controlnet")
    parser.add_argument("--seed", 
                        type=int, 
                        default=42,
                        help="the seed (for reproducible sampling)")
    parser.add_argument("--device", 
                        type=str, 
                        default="cuda")
    parser.add_argument("--device_ids", 
                        type=str, 
                        default=[0, 1]) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    opt = parser.parse_args()
    
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config_model}")
    metrics_config = OmegaConf.load(f"{opt.config_metrics}")
    run_name, model_name = config.name, opt.config_model.split('/')[-2]
    model = load_model_from_ckpt(config.model, f"{opt.ckpt}").cpu()
    model = model.to(opt.device)
    
    metrics_config.metrics.params.pose.run_name = run_name + '_eval'
    metrics_calculator=instantiate_from_config(metrics_config.metrics)
    
    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
        
    sampler=torch.nn.DataParallel(sampler,device_ids=opt.device_ids)
    os.makedirs('outputs', exist_ok=True)
    outpath = os.path.join('outputs', model_name, run_name)
    # output dir already exists
    if os.path.exists(outpath):
        import time
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
        outpath = outpath + '_' + current_time
    metrics_outpath = os.path.join(outpath, 'metrics')
    os.makedirs(metrics_outpath, exist_ok=True)
    images_outpath = os.path.join(outpath, 'images')
    os.makedirs(images_outpath, exist_ok=True)

    batch_size = opt.batch_size
    start_code = None
    
    # define test dataset
    dataset = instantiate_from_config(config.data.params.test)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    all_metrics_results={metric:{} for metric in opt.metrics}
    
    with torch.no_grad():
        for batch_idx, (data) in enumerate(test_loader):
            with model.ema_scope():
                input_data=model.get_input(data,0) 
                all_conds = input_data[1]
                # used for t2i-adapter only
                features_adapter, append_to_context = None, None
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * ["longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"])
                    pose_condition = rearrange(data['hint'], 'b h w c -> b c h w')
                    all_conds = {'c_crossattn': input_data[1]['c_crossattn'], 'pose_condition':[pose_condition]}
                    uc = {'c_crossattn': [uc], 'pose_condition':[pose_condition]}
                    
                    # generate pose mask
                    gaussian_kernels = config.model.params.gaussian_kernels
                    pose_masks = gen_pose_mask(gaussian_kernels, data, batch_size)
                    all_conds.update({'pose_mask': pose_masks[len(gaussian_kernels) * batch_size:]})
                    uc.update({'pose_mask': pose_masks[:len(gaussian_kernels) * batch_size]})
                
                samples_ddim, _ = sampler.module.sample(S=opt.ddim_steps,
                                                conditioning=all_conds,
                                                batch_size=batch_size,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc,
                                                eta=opt.ddim_eta,
                                                x_T=start_code,
                                                features_adapter=features_adapter,
                                                append_to_context=append_to_context,
                                                cond_tau=1.0,)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1)
                metrics = metrics_calculator.calc_metrics(data, x_samples_ddim,metrics=opt.metrics)
                
                for key in all_metrics_results.keys():
                    if key == 'quality': 
                        # won't calculate here, pls refer to eval_quality.py
                        continue
                    if len(all_metrics_results[key].keys()):
                        for metric_key in all_metrics_results[key].keys():
                            all_metrics_results[key][metric_key].append(metrics[key][metric_key])
                        with open(os.path.join(metrics_outpath,key+".csv"),"a") as f:
                            f.write(",".join(str(v[-1]) for v in all_metrics_results[key].values())+"\n")
                    else:
                        for metric_key in metrics[key].keys():
                            all_metrics_results[key][metric_key]=[metrics[key][metric_key]]
                        with open(os.path.join(metrics_outpath,key+".csv"),"w") as f:
                            f.write(",".join(str(v) for v in all_metrics_results[key].keys())+"\n")
                        with open(os.path.join(metrics_outpath,key+".csv"),"a") as f:
                            f.write(",".join(str(v[-1]) for v in all_metrics_results[key].values())+"\n")
                
                x_samples_ddim = x_samples_ddim.cpu().detach().numpy()
                x_samples=(x_samples_ddim*255).astype(np.uint8)
                text_images = log_txt_as_img((x_samples.shape[1], x_samples.shape[2]), \
                                            [prompt_i+"\n"+",".join(str(v[-1]) for v in all_metrics_results["pose"].values()) for prompt_i in data["txt"]], size=x_samples.shape[2] // 25)  #<<<<<<< data["prompt"]
                text_images = (einops.rearrange(text_images, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                
                original_images=torch.clamp((data["jpg"]+ 1.0) / 2.0, min=0.0, max=1.0)
                original_images = original_images.cpu().numpy()* 255
                
                # save images
                for batch_i in range(batch_size):
                    present_generated_img=x_samples[batch_i,...][:,:,[2,1,0]]
                    present_text_image=text_images[batch_i,...]
                    original_image=original_images[batch_i,...]
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    present_pose_image = draw_humansd_skeleton(original_image, data["pose"][batch_i], keypoint_thresh=0.05)
                    
                    save_image=np.concatenate([present_generated_img,present_pose_image,present_text_image,original_image],1)
                    save_folder, save_name = data["img_path"][batch_i].split('/')[-2:]
                    save_path=os.path.join(images_outpath,save_folder)
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    
                    cv2.imwrite(os.path.join(save_path, save_name),save_image)
                    print(f'saved image {os.path.join(save_path, save_name)}')
                
            # print
            print(f"======================== batch:{batch_idx} ==========================")
            print("Present Metrics:")
            for key in all_metrics_results.keys():
                print(f"\t{key}:")
            
                for metric_key in all_metrics_results[key].keys():
                    print(f"\t{metric_key}: {list(all_metrics_results[key][metric_key])[-1]}")
                        
            print("==================================================") 
            
            
if __name__ == "__main__":
    main()

