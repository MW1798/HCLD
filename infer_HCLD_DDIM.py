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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import shutil
import tempfile
from pathlib import Path
import datetime

import data.MRIdata as MRI

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
import torchvision
from PIL import Image


from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.networks.layers import Act
from monai.utils import first, set_determinism
from torch.cuda.amp import autocast
from tqdm import tqdm
from itertools import cycle


from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler



def inference():
	count = 0
	with torch.no_grad():
		for ((val_step, batch_src_val), batch_tar_val) in progress_bar: # source only
			results = {}
			fn = batch_src_val['fn']
			
			if resume and all(item in resume_fn_list for item in fn):
				continue
			
			conditions = autoencoder.encode_stage_2_inputs(batch_tar_val['image'].to(device))*scale_factor
			images = batch_src_val["image"][:len(conditions)].to(device)
			results['orig_src']=batch_src_val['image'].float()

			
			z_x = autoencoder.encode_stage_2_inputs(images) * scale_factor
   
			# reverse DDIM sampling to add noise
			scheduler_ddim.set_timesteps(num_inference_steps=num_inference_fdp)
			img_noisy, latent_noisy = inferer.reverse_sample(
				input_noise=z_x, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler_ddim,
				conditioning=conditions,mode='concat', verbose=False, return_latent=True
			)

			# DDIM RDP
			scheduler_ddim.set_timesteps(num_inference_steps=num_inference_rdp)
			recon_images = inferer.sample(
				input_noise=latent_noisy, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler_ddim,
				conditioning=conditions,mode='concat', verbose=False
			)
			if clip_neg:
				recon_images[recon_images<0]=0
			recon_images = (recon_images-recon_images.min())/(recon_images.max()-recon_images.min()) # -> [0,1]
			# print([recon_images.min(),recon_images.max(),recon_images.mean(),recon_images.std()])
			recon_images = recon_images * brain_mask
			images = images * brain_mask
			results['recon']=recon_images.detach().cpu().float()
   
			del recon_images, z_x, img_noisy
			torch.cuda.empty_cache()
			

			# DDIM RDP for no GN
			scheduler_ddim.set_timesteps(num_inference_steps=num_inference_rdp)
			recon_images = inferer.sample(
				input_noise=latent_noisy, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler_ddim,
				conditioning=conditions,mode='concat', GN=False, verbose=False
			)
			if clip_neg:
				recon_images[recon_images<0]=0
			recon_images = (recon_images-recon_images.min())/(recon_images.max()-recon_images.min()) # -> [0,1]
			# print([recon_images.min(),recon_images.max(),recon_images.mean(),recon_images.std()])
			recon_images = recon_images * brain_mask
			results['noGN']=recon_images.detach().cpu().float()

	  
   
			del recon_images, latent_noisy
			torch.cuda.empty_cache()
		   
			if save_results:
				for b_idx in range(len(fn)):
					for k in results:
						if k not in ['noGN','recon']:
							continue
						img_volume = results[k][b_idx].detach().cpu() # [1,W,H,Z], eg: torch.Size([1, 192, 192, 64])
						img_volume[img_volume<0]=0
						img_volume = (img_volume-img_volume.min())/(img_volume.max()-img_volume.min()) # -> [0,1] 
						save_fn = f'{fn[b_idx]}_{k}.npy'
						full_save_pt = save_dir/k/save_fn
						if not full_save_pt.parent.exists():
							os.makedirs(full_save_pt.parent)
						np.save(full_save_pt,img_volume.squeeze().float())
						count += 1
						
			if save_samples:
				root = save_dir / 'samples'
				if not root.exists():
					os.makedirs(root)
				for k in results:
					img_volume = results[k].detach().cpu() # [B,1,W,H,Z], eg: torch.Size([2, 1, 192, 192, 64])
					img_volume[img_volume<0]=0
					img_volume = (img_volume-img_volume.min())/(img_volume.max()-img_volume.min()) # -> [0,1]

					grid_a = torchvision.utils.make_grid(img_volume[:,:,:,:,img_volume.shape[4]//2], nrow=1,normalize=True) # axial middle slices
					grid_a = grid_a.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
					grid_a = (grid_a * 255).astype(np.uint8)
					filename = "{}_{}_{}.png".format(fn[0],k,'a')
					save_path = root / filename
					Image.fromarray(grid_a).save(save_path)


					grid_c = torchvision.utils.make_grid(img_volume[:,:,:,img_volume.shape[3]//2,:], nrow=4,normalize=True) # coronal middle slice
					grid_c = grid_c.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
					grid_c = (grid_c * 255).astype(np.uint8)
					filename = "{}_{}_{}.png".format(fn[0],k,'c')
					save_path = root / filename
					Image.fromarray(grid_c).save(save_path)

					grid_s = torchvision.utils.make_grid(img_volume[:,:,img_volume.shape[2]//2,:,:], nrow=4,normalize=True) # saggital middle slice
					grid_s = grid_s.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
					grid_s = (grid_s * 255).astype(np.uint8)
					filename = "{}_{}_{}.png".format(fn[0],k,'s')
					save_path = root / filename
					Image.fromarray(grid_s).save(save_path)
			 
		if include_target:       
			for batch_tar_val in tqdm(target_loader): # target only
				results = {}
				fn = batch_tar_val['fn']
				
				if resume and all(item in resume_fn_list for item in fn):
					continue
				
				conditions = autoencoder.encode_stage_2_inputs(batch_tar_val['image'].to(device))*scale_factor
				images = batch_tar_val["image"][:len(conditions)].to(device)
				
				z_x = autoencoder.encode_stage_2_inputs(images) * scale_factor
	
				# reverse DDIM sampling to add noise
				scheduler_ddim.set_timesteps(num_inference_steps=num_inference_fdp)
				img_noisy, latent_noisy = inferer.reverse_sample(
					input_noise=z_x, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler_ddim,
					conditioning=conditions,mode='concat', verbose=False, return_latent=True
				)

				# DDIM RDP
				scheduler_ddim.set_timesteps(num_inference_steps=num_inference_rdp)
				recon_images = inferer.sample(
					input_noise=latent_noisy, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler_ddim,
					conditioning=conditions,mode='concat', verbose=False
				)
				if clip_neg:
					recon_images[recon_images<0]=0
				recon_images = (recon_images-recon_images.min())/(recon_images.max()-recon_images.min()) # -> [0,1]
				recon_images = recon_images * brain_mask
				images = images * brain_mask
				results['recon']=recon_images.detach().cpu().float()
	
				del recon_images, z_x, img_noisy
				torch.cuda.empty_cache()
				

				# DDIM RDP for no GN
				scheduler_ddim.set_timesteps(num_inference_steps=num_inference_rdp)
				recon_images = inferer.sample(
					input_noise=latent_noisy, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler_ddim,
					conditioning=conditions,mode='concat', GN=False, verbose=False
				)
				if clip_neg:
					recon_images[recon_images<0]=0
				recon_images = (recon_images-recon_images.min())/(recon_images.max()-recon_images.min()) # -> [0,1]
				# print([recon_images.min(),recon_images.max(),recon_images.mean(),recon_images.std()])
				recon_images = recon_images * brain_mask
				results['noGN']=recon_images.detach().cpu().float()
				
				if save_results:
					for b_idx in range(len(fn)):
						for k in results:
							if k not in ['noGN','recon']:
								continue
							img_volume = results[k][b_idx].detach().cpu() # [1,W,H,Z]
							img_volume[img_volume<0]=0

							img_volume = (img_volume-img_volume.min())/(img_volume.max()-img_volume.min()) # -> [0,1] 
							save_fn = f'{fn[b_idx]}_{k}.npy'
							full_save_pt = save_dir/k/save_fn
							if not full_save_pt.parent.exists():
								os.makedirs(full_save_pt.parent)
							np.save(full_save_pt,img_volume.squeeze().float())
							count += 1
	   
				if save_samples:
					root = save_dir / 'samples'
					if not root.exists():
						os.makedirs(root)
					for k in results:
						img_volume = results[k].detach().cpu() # [B,1,W,H,Z]
						img_volume[img_volume<0]=0
						img_volume = (img_volume-img_volume.min())/(img_volume.max()-img_volume.min()) # -> [0,1]

						grid_a = torchvision.utils.make_grid(img_volume[:,:,:,:,img_volume.shape[4]//2], nrow=1,normalize=True) # axial middle slices
						grid_a = grid_a.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
						grid_a = (grid_a * 255).astype(np.uint8)
						filename = "{}_{}_{}.png".format(fn[0],k,'a')
						save_path = root / filename
						Image.fromarray(grid_a).save(save_path)


						grid_c = torchvision.utils.make_grid(img_volume[:,:,:,img_volume.shape[3]//2,:], nrow=4,normalize=True) # coronal middle slice
						grid_c = grid_c.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
						grid_c = (grid_c * 255).astype(np.uint8)
						filename = "{}_{}_{}.png".format(fn[0],k,'c')
						save_path = root / filename
						Image.fromarray(grid_c).save(save_path)

						grid_s = torchvision.utils.make_grid(img_volume[:,:,img_volume.shape[2]//2,:,:], nrow=4,normalize=True) # saggital middle slice
						grid_s = grid_s.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
						grid_s = (grid_s * 255).astype(np.uint8)
						filename = "{}_{}_{}.png".format(fn[0],k,'s')
						save_path = root / filename
						Image.fromarray(grid_s).save(save_path)
      
		print(f"Saved {count} files to {save_dir}")
					

if __name__ == '__main__':
	bs = 1
	# source_dataset = MRI.OpenBHB_val_3d_src()
	source_dataset = MRI.OpenBHB_train_3d_src()
	# source_dataset = MRI.IXI_train_3d_src()
	# source_dataset = MRI.IXI_val_3d_src()

	# target_dataset = MRI.OpenBHB_val_3d_tar()
	target_dataset = MRI.OpenBHB_train_3d_tar()
	# target_dataset = MRI.IXI_train_3d_tar()
	# target_dataset = MRI.IXI_val_3d_tar()
 

	source_loader = DataLoader(source_dataset, batch_size=bs, shuffle=True, num_workers=3, persistent_workers=True,drop_last=False)
	target_loader = DataLoader(target_dataset, batch_size=bs, shuffle=True, num_workers=3, persistent_workers=True,drop_last=False)
	
		# Check if CUDA is available
	if torch.cuda.is_available():
		# Get the current default CUDA device
		device = torch.cuda.current_device()
		# Print the name of the current default CUDA device
		print('Using CUDA: ',torch.cuda.get_device_name(device))
	else:
		print("CUDA is not available.")
	torch.cuda.empty_cache()

	now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")

	run_name = 'sample_run'

	include_target = True
	save_results = True
	save_samples = False
	adain = True
	clip_neg = True
 
	# define inference parameters
	num_train_ddim = 50
	num_inference_fdp = 30
	num_inference_rdp = 10


	print(run_name)

	save_dir = Path(f'sample_path/{now}_{run_name}')

	resume = False
	if resume:
		resume_dir = Path('sample_path/HCLD_ckp.pth')
  
		save_dir = resume_dir

		if len(os.listdir(save_dir))>5:
			print(f'Resuem inference, find {len(os.listdir(save_dir))} files')
			resume_fn_list = [f.split('_')[0] for f in os.listdir(save_dir)] # list of IDs
		else:
			subdir = save_dir/os.listdir(save_dir)[0]
			print(f'Resuem inference, find {len(os.listdir(subdir))} files')
			resume_fn_list = [f.split('_')[0] for f in os.listdir(subdir)] # list of IDs

	elif not save_dir.exists():
		os.makedirs(save_dir)
	elif not resume:
		assert len(os.listdir(save_dir))==0,'Log dir exist!'
  
	parameters = f"""AdaIN = {adain},
	num_inference_fdp = {num_inference_fdp},
	num_inference_rdp = {num_inference_rdp},
	num_train_ddim = {num_train_ddim}"""
 
	print(parameters)
	with open(save_dir / 'parameters.txt', 'w') as f:
		f.write(run_name+'\n')
		f.write(parameters)
		

	#### load pth
	ae_pt = torch.load(Path('sample_path/auteoencoder.pth'))

	LDM_pt = torch.load('sample_path/ldm_ckp.pth') 

	brain_mask = torch.from_numpy(np.load('sample_path/mean_brain_mask.npy')).unsqueeze(0).float().to(device) 

	autoencoder = AutoencoderKL(
		spatial_dims=3,
		in_channels=1,
		out_channels=1,
		num_channels=(32, 64, 64),
		latent_channels=6,
		num_res_blocks=2,
		norm_num_groups=8,
		attention_levels=(False, False, True),
  		use_flash_attention = True

	)
	autoencoder.to(device)
	
	autoencoder.load_state_dict(ae_pt['model_state_dict'])
	print('AE weighted loaded!')


	unet = DiffusionModelUNet(
		spatial_dims=3,
		in_channels=12,
		out_channels=6,
		num_res_blocks=1,
		num_channels=(32, 64, 64),
		attention_levels=(False, True, True),
		num_head_channels=(0, 64, 64),
		use_flash_attention=True
	)
	unet.to(device)
	
	unet.load_state_dict(LDM_pt['unet_state_dict'])
	print('LDM weighted loaded!')

	autoencoder.eval()
	unet.eval()

	# define scalar
	check_data = first(source_loader)
	with torch.no_grad():
		with autocast(enabled=True):
			z_ = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))

	print(f"Scaling factor set to {1/torch.std(z_)}")
	scale_factor = 1 / torch.std(z_)
	z = torch.zeros_like(z_).cpu()
	del z_, check_data
	torch.cuda.empty_cache()

	adain = True
	scheduler_ddim = DDIMScheduler(num_train_timesteps=num_train_ddim, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
	inferer = LatentDiffusionInferer(scheduler_ddim, scale_factor=scale_factor,Adain='AdaIN_reverse')
	
	progress_bar = tqdm(zip(enumerate(source_loader),cycle(target_loader)),total=len(source_loader), ncols=150) # progress bar need to be after check_data

	
	progress_bar.set_description(f"Inference Source")
	
	inference()