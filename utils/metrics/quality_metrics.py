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
        
        fid_image_transforms=TF.Compose([
                TF.Resize(299),
                TF.CenterCrop(299),
                TF.ToTensor(),
                TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        dataset_imgs=[]
        with open(refer_dataset_json_path,"r") as f:
            dataset_json=json.load(f)
        
        print("initialize contrast dataset")
        '''
        for image_i in tqdm(range(len(dataset_json["images"]))):
            present_image_path = os.path.join(refer_dataset_base_dir,dataset_json["images"][image_i]["file_name"])
            img = Image.open(present_image_path).convert('RGB')
            dataset_imgs.append(fid_image_transforms(img).unsqueeze(0))
        '''
        key_list = list(dataset_json.keys())
        for image_i in tqdm(range(len(dataset_json))):
            present_image_path = os.path.join(refer_dataset_base_dir,dataset_json[key_list[image_i]]["img_path"])
            img = Image.open(present_image_path).convert('RGB')
            dataset_imgs.append(fid_image_transforms(img).unsqueeze(0))
            
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
        
