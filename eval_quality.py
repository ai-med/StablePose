"""
 ------------------------------------------------------------------------
 Modified from HumanSD (https://github.com/IDEA-Research/HumanSD/tree/main)
 ------------------------------------------------------------------------
"""

import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.utilities.data import dim_zero_cat

from PIL import Image
import torchvision.transforms as TF
from tqdm import tqdm
import json
import os
import cv2

class QualityMetrics():
    def __init__(self,
                 device,
                 refer_dataset_base_dir,
                 refer_dataset_json_path,
                 fid_model_feature,
                 kid_subset_size):
        
        # FID
        self.refer_dataset_base_dir=refer_dataset_base_dir
        self.refer_dataset_json_path=refer_dataset_json_path
        self.device=device
        
        self.fid_image_transforms=TF.Compose([
                TF.Resize(299),
                TF.CenterCrop(299),
                TF.ToTensor(),
                TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        self.fid_model_feature=fid_model_feature
        self.kid_subset_size=kid_subset_size
        self.fid_model = self.kid_model = None
        # IS
        self.inception_model = InceptionScore(normalize=True).to(self.device)

        dataset_imgs=[]
        with open(refer_dataset_json_path,"r") as f:
            dataset_json=json.load(f)
        
        print(f"initialize dataset {refer_dataset_json_path.split('/')[-1].split('.')[0]}")

        key_list = list(dataset_json.keys())
        for image_i in tqdm(range(len(dataset_json))):
            present_image_path = os.path.join(refer_dataset_base_dir,dataset_json[key_list[image_i]]["img_path"])
            img = Image.open(present_image_path).convert('RGB')
            dataset_imgs.append(self.fid_image_transforms(img).unsqueeze(0))
            
        dataset_imgs=torch.concat(dataset_imgs).to(self.device)
        
        self.fid_model_feature=fid_model_feature
        self.fid_model=FrechetInceptionDistance(feature=self.fid_model_feature,normalize=True).to(self.device)
        self.fid_model.update(dataset_imgs,real=True)
        
        # IS
        self.inception_model = InceptionScore(normalize=True).to(self.device)
        
        # KID
        self.kid_subset_size=kid_subset_size
        self.kid_model = KernelInceptionDistance(subset_size=self.kid_subset_size,normalize=True).to(self.device)
        self.kid_model.update(dataset_imgs, real=True)


    def calculate_fid(self, img):
        self.fid_model.update(img, real=False)
        return self.fid_model.compute()
    
    def calculate_kid(self, img):
        self.kid_model.update(img, real=False)
        # self.kid_model.fake_features becomes longer batch by batch
        if dim_zero_cat(self.kid_model.fake_features).shape[0]<=self.kid_model.subset_size:
            return None
        return self.kid_model.compute()

    def calculate_is(self,img):
        self.inception_model.update(img)
        return self.inception_model.compute()

    def compute(self, batch, output_images):    
        if  type(output_images) is np.ndarray:
            output_images=torch.tensor(output_images)
            
        if  output_images.shape[-1]==3:
            output_images=output_images.permute(0,3,1,2)
            
        with torch.no_grad():
            fid_value=self.calculate_fid(output_images)
            is_value=self.calculate_is(output_images)
            kid_value=self.calculate_kid(output_images)
        
        fid_result={
                "Fréchet Inception Distance    (FID)                                               ": fid_value.item(),
            }
        
        is_result={
                "Inception Score (IS)                                                              ": is_value[0].item(), # (mean, std)
            }
        if kid_value is None:
            kid_result={
                    "Kernel Inception Distance (KID)                                                    ": kid_value,
                }
        else:
            kid_result={
                    "Kernel Inception Distance (KID)                                                    ": kid_value[0].item(), #(mean, std)
                }
        
        results={**fid_result,**is_result,**kid_result}
        return results
    
    def __call__(self,batch, output_images):
        quality_result=self.compute(batch, output_images)
        return quality_result

if __name__ == "__main__":
    img_root = 'outputs/stable_pose/run_name/images'# please specify path to generated images
    run_name = img_root.split('/')[-2]
    results = {}

    for sub_folder in os.listdir('val_jsons'):
        img_dir = os.path.join(img_root, sub_folder.split('.')[0])
        if not os.path.exists(img_dir):
            print(f'{img_dir} is not generated yet.')
            continue
        refer_dataset_json_path = os.path.join('val_jsons', sub_folder)
        quality_metrics = QualityMetrics(device='cuda',
                         refer_dataset_base_dir='path_to_HumanArt', # please specify path to HumanArt here
                         refer_dataset_json_path=refer_dataset_json_path,
                         fid_model_feature=64,
                         kid_subset_size=200)

        fid_image_transforms=TF.Compose([
            TF.Resize(299),
            TF.ToTensor(),
            TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image_list = []
        for img_path in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_path)
            img = Image.open(img_path).convert('RGB')
            # extract generated images from saved ones
            img = np.array(img).reshape(512,4,512,3) # (512, 2048, 3)
            img = cv2.cvtColor(img[:,0,:,:], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            image_list.append(fid_image_transforms(img).unsqueeze(0))

        images=torch.concat(image_list).to('cuda')
        metrics = quality_metrics(batch=None, output_images=images)
        renamed_metrics = {}
        for k,v in metrics.items():
            if k.startswith('F'):
                renamed_metrics['FID'] = v
            elif k.startswith('K'):
                renamed_metrics['KID'] = v
            elif k.startswith('I'):
                renamed_metrics['IS'] = v
            else:
                raise NotImplementedError(f"unknown key {k}")
        results[sub_folder.split('.')[0]] = renamed_metrics
        print(renamed_metrics)

        with open(os.path.join('quality_results_' + run_name + '.json'),"w") as f:
            json.dump(results, f)

    results['overall'] = {}
    fid_list, kid_list, is_list = [], [], []
    for k, v in results.items():
        if k == 'overall':
            continue
        if not v['KID'] is None:
            kid_list.append(v['KID'])
        fid_list.append(v['FID'])
        is_list.append(v['IS'])

    results['overall']['FID'] = sum(fid_list)/len(fid_list)
    results['overall']['KID'] = sum(kid_list)/len(kid_list)
    results['overall']['IS'] = sum(is_list)/len(is_list)
    
    with open(os.path.join('quality_results_' + run_name + '.json'),"w") as f:
        json.dump(results, f)
    print('done')
    

