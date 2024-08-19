# Title: Unpaired Volumetric Harmonization of Brain MRI with Conditional Latent Diffusion
# Author: Mengqi Wu, Minhui Yu, Shuaiming Jing, Pew-Thian Yap, Zhengwu Zhang, Mingxia Liu
# Date: August 2024

# Copyright (c) 2024 Mengqi Wu, mengqiw@unc.edu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tr
from pathlib import Path
import pandas as pd

class MRIdata_3d(Dataset):
	def __init__(self,image_dir:str,annotation_file=None): 
		assert Path(image_dir).is_dir(), f'{image_dir} is not a valid directory'
		
		self.img_dir = image_dir
		self.img_labels = annotation_file
		self.length = len(self.img_labels)
		self.transform = tr.Compose([
					tr.ToTensor()
				])

	def __len__(self):
		return self.length

	def __getitem__(self,idx):
		img_path_base = Path(self.img_dir)
		fn  = str(self.img_labels.iloc[idx,0])
		img_path_full = str(next(img_path_base.glob(f'*{fn}*.npy')))
		site = self.img_labels.iloc[idx,1]

		img_volume = np.load(img_path_full).astype(np.float32)

		img_volume = torch.from_numpy(img_volume)
		if len(img_volume.shape) != 4:
			img_volume = img_volume.unsqueeze(0)
		# print('min, max, mean after normalization: ',[img_volume.min(),img_volume.max(),img_volume.mean()])
		# print(img_volume.shape)
		# print(type(img_volume))
		# print(img_volume.dtype)
		if 'age' in self.img_labels.columns:
			age = self.img_labels.iloc[idx,2]
			example = {'image':img_volume,'fn':fn,'age':torch.tensor(age).float(), 'site':torch.tensor(site)}
		else:
			example = {'image':img_volume,'fn':fn}

		return example

class MRIdata_3d_p(Dataset):
	def __init__(self,image_dir:str,annotation_file=None): 
		assert Path(image_dir).is_dir(), f'{image_dir} is not a valid directory'
		assert isinstance(annotation_file,list), 'Annotation file should be a list of 2 dataframes'
		
		self.img_dir = image_dir
		# self.img_labels = pd.read_csv(annotation_file,sep='\t')
		self.img_labels_src = annotation_file[0]
		self.img_labels_tar = annotation_file[1]
		self.length = len(self.img_labels_src)

	def __len__(self):
		return self.length

	def __getitem__(self,idx):
		img_path_base = Path(self.img_dir)

		# load src image
		fn_src  = str(self.img_labels_src.iloc[idx,0])
		img_path_full_src = str(next(img_path_base.glob(f'*{fn_src}*.npy')))
		img_volume_src = np.load(img_path_full_src).astype(np.float32)
		img_volume_src = torch.from_numpy(img_volume_src)
		if len(img_volume_src.shape) != 4:
			img_volume_src = img_volume_src.unsqueeze(0)

		site = self.img_labels_src.iloc[idx,1]
		sub_id = self.img_labels_src.iloc[idx,2]

		# load tar image
		fn_tar = list(self.img_labels_tar.loc[self.img_labels_tar['subject_id']==sub_id]['filename'])[0]
		img_path_full_tar = str(next(img_path_base.glob(f'*{fn_tar}*.npy')))
		img_volume_tar = np.load(img_path_full_tar).astype(np.float32)
		img_volume_tar = torch.from_numpy(img_volume_tar)
		if len(img_volume_tar.shape) != 4:
			img_volume_tar = img_volume_tar.unsqueeze(0)


		# print('min, max, mean after normalization: ',[img_volume.min(),img_volume.max(),img_volume.mean()])
		# print(img_volume.shape)
		# print(type(img_volume))
		# print(img_volume.dtype)
		if 'age' in self.img_labels_src.columns:
			age = self.img_labels_src.iloc[idx,2]
			example = {'image':img_volume_src,'fn':fn_src,'age':torch.tensor(age), 'site':torch.tensor(site)}
			return example
		else:
			return {'image':img_volume_src,'fn':fn_src}, {'image':img_volume_tar,'fn':fn_tar}

class MRIdata_3d_cv(Dataset):
	def __init__(self,image_dir:list,annotation_file=None): 
		# assert Path(image_dir).is_dir(), f'{image_dir} is not a valid directory'
		# if annotation_file is not None: assert Path(annotation_file).is_file(), f'{annotation_file} is not a valid file'
		
		self.img_dir = image_dir
		# self.img_labels = pd.read_csv(annotation_file,sep='\t')
		self.img_labels = annotation_file
		self.length = len(self.img_labels)
		self.transform = tr.Compose([
					tr.ToTensor()
					# tr.Resize((input_size,input_size))
				])

	def __len__(self):
		return self.length

	def __getitem__(self,idx):
		fn  = str(self.img_labels.iloc[idx,0])
		# img_path_full = str(next(img_path_base.glob(f'*{fn}*.npy')))
		img_path_full = next(self.img_dir[idx].parent.glob(f'*{self.img_dir[idx].name}*.npy')) # get coarse path and search for full fn in that path
		site = self.img_labels.iloc[idx,1]

		img_volume = np.load(img_path_full).astype(np.float32)
		# img_volume = self.transform(img_volume)
		# img_volume = torch.cat([self.transform(img_volume[i]) for i in range(img_volume.shape[0])])
		# img_volume = torch.tensor(img_volume,device='cpu') # transform already contain tr.toTensor()
		img_volume = torch.from_numpy(img_volume)
		if len(img_volume.shape) != 4:
			img_volume = img_volume.unsqueeze(0)
		# print('min, max, mean after normalization: ',[img_volume.min(),img_volume.max(),img_volume.mean()])
		# print(img_volume.shape)
		# print(type(img_volume))
		# print(img_volume.dtype)
		if 'age' in self.img_labels.columns:
			age = self.img_labels.iloc[idx,2]
			example = {'image':img_volume,'fn':fn,'age':torch.tensor(age).float(), 'site':torch.tensor(site)}
		else:
			example = {'image':img_volume,'fn':fn, 'site':site}

		return example


class OpenBHB_train_3d(MRIdata_3d):
	def __init__(self):
		image_dir = Path('F:/OpenBHB/train/train_192_64/')
		annotation_file = pd.read_csv('F:/OpenBHB/train/train_labels/official_site_class_labels.tsv',sep='\t')
		super().__init__(image_dir, annotation_file)
  
class OpenBHB_val_3d(MRIdata_3d):
	def __init__(self):
		image_dir = Path('F:/OpenBHB/val/val_192_64/')
		annotation_file = pd.read_csv('F:/OpenBHB/val/val_labels/official_site_class_labels.tsv',sep='\t')
		super().__init__(image_dir, annotation_file)
  
class OpenBHB_test_3d(MRIdata_3d):
	def __init__(self):
		image_dir = Path('F:/OpenBHB/val/val_192_64/')
		annotation_file = pd.read_csv('F:/OpenBHB/val/val_labels/official_site_class_labels.tsv',sep='\t').head(24)
		super().__init__(image_dir, annotation_file)

### LDM training

class OpenBHB_train_3d_src(MRIdata_3d):
	def __init__(self):
		image_dir = Path('F:/OpenBHB/train/train_192_64/')
		annotation_file = pd.read_csv('F:/OpenBHB/train/train_labels/site_cl_source_~17.tsv',sep='\t')
		super().__init__(image_dir, annotation_file)

class OpenBHB_train_3d_tar(MRIdata_3d):
	def __init__(self):
		image_dir = Path('F:/OpenBHB/train/train_192_64/')
		annotation_file = pd.read_csv('F:/OpenBHB/train/train_labels/site_17.tsv',sep='\t')
		super().__init__(image_dir, annotation_file)
  
class OpenBHB_val_3d_src(MRIdata_3d):
	def __init__(self):
		image_dir = Path('F:/OpenBHB/val/val_192_64/')
		annotation_file = pd.read_csv('F:/OpenBHB/val/val_labels/site_cl_source_~17.tsv',sep='\t')
		super().__init__(image_dir, annotation_file)

class OpenBHB_val_3d_tar(MRIdata_3d):
	def __init__(self):
		image_dir = Path('F:/OpenBHB/val/val_192_64/')
		annotation_file = pd.read_csv('F:/OpenBHB/val/val_labels/site_17.tsv',sep='\t')
		super().__init__(image_dir, annotation_file)

######## IXI
class IXI_train_3d_src(MRIdata_3d):
	def __init__(self):
		image_dir = Path('F:/IXI_T1_FSL/train_192_64/')
		annotation_file = pd.read_csv('F:/IXI_T1_FSL/IXI_train_src.tsv',sep='\t')
		super().__init__(image_dir, annotation_file)

class IXI_train_3d_tar(MRIdata_3d):
	def __init__(self):
		image_dir = Path('F:/IXI_T1_FSL/train_192_64/')
		annotation_file = pd.read_csv('F:/IXI_T1_FSL/IXI_train_tar.tsv',sep='\t')
		super().__init__(image_dir, annotation_file)
  
class IXI_val_3d_src(MRIdata_3d):
	def __init__(self):
		image_dir = Path('F:/IXI_T1_FSL/train_192_64/')
		annotation_file = pd.read_csv('F:/IXI_T1_FSL/IXI_val_src.tsv',sep='\t')
		super().__init__(image_dir, annotation_file)

class IXI_val_3d_tar(MRIdata_3d):
	def __init__(self):
		image_dir = Path('F:/IXI_T1_FSL/train_192_64/')
		annotation_file = pd.read_csv('F:/IXI_T1_FSL/IXI_val_tar.tsv',sep='\t')
		super().__init__(image_dir, annotation_file)


class SRPBS_train_3d_src(MRIdata_3d):
	def __init__(self):
		image_dir = Path('F:/SRPBS_fsl/diffusion_project/SRPBS_train_192_64')
		annotation_file = pd.read_csv('F:/SRPBS_fsl/diffusion_project/SRPBS_train_src.tsv',sep='\t')
		super().__init__(image_dir, annotation_file)

class SRPBS_train_3d_tar(MRIdata_3d):
	def __init__(self):
		image_dir = Path('F:/SRPBS_fsl/diffusion_project/SRPBS_train_192_64')
		annotation_file = pd.read_csv('F:/SRPBS_fsl/diffusion_project/SRPBS_train_tar.tsv',sep='\t')
		super().__init__(image_dir, annotation_file)

class SRPBS_val_3d_src(MRIdata_3d):
	def __init__(self):
		image_dir = Path('F:/SRPBS_fsl/diffusion_project/SRPBS_val_192_64')
		annotation_file = pd.read_csv('F:/SRPBS_fsl/diffusion_project/SRPBS_val_src.tsv',sep='\t')
		super().__init__(image_dir, annotation_file)
  
class SRPBS_val_3d_tar(MRIdata_3d):
	def __init__(self):
		image_dir = Path('F:/SRPBS_fsl/diffusion_project/SRPBS_val_192_64')
		annotation_file = pd.read_csv('F:/SRPBS_fsl/diffusion_project/SRPBS_train_tar.tsv',sep='\t')
		super().__init__(image_dir, annotation_file)
