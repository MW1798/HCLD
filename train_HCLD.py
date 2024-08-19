# Title: Unpaired Volumetric Harmonization of Brain MRI with Conditional Latent Diffusion
# Author: Mengqi Wu, Minhui Yu, Shuaiming Jing, Pew-Thian Yap, Zhengwu Zhang, Mingxia Liu
# Date: August 2024

# Modified by Mengqi Wu from MONAI example
# Copyright (c) MONAI Consortium
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
from monai.losses.ssim_loss import SSIMLoss
from torch.cuda.amp import autocast
from tqdm import tqdm
from itertools import cycle


from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

import util

S_loss_type = 'GS' # use gram-based style loss
# S_loss_type = 'S' # use statistic-based style loss

if S_loss_type == 'GS':
	from util import style_loss_gram as Style_loss, IN # Gs
else:
	from util import Style_loss, IN # S


from torch.utils.tensorboard import SummaryWriter
# torch.set_float32_matmul_precision('medium')


def train():
	print(f'Train on {len(train_loader_src.dataset)}, validate on {len(val_loader_src.dataset)}')

	# define autoencoder
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

	# load autoencoder checkpoint
	state_dict = Path('sample_path/autoencoder_ckp.pth')
	if not state_dict is None:
		state_dict = torch.load(str(state_dict))
		autoencoder.load_state_dict(state_dict['model_state_dict'])
		print('AE weighted loaded!')
	autoencoder.eval()

	# define cLDM
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
 

	# define noise scheduler
	scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
	scheduler_ddim = DDIMScheduler(num_train_timesteps=num_train_ddim, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)

	# define scalar
	check_data = first(train_loader_src)
	with torch.no_grad():
		with autocast(enabled=True):
			z_ = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))

	print(f"Scaling factor set to {1/torch.std(z_)}")
	scale_factor = 1 / torch.std(z_)
	z = torch.zeros_like(z_).cpu()
	del z_
	torch.cuda.empty_cache()

	s_D = util.Style_Discriminator_3d().to(device)  # discriminator on full 3D latent
	optimizer_s_D = torch.optim.Adam(s_D.parameters(), lr=1e-5, betas=(0.5, 0.999))
	style_criterion = torch.nn.BCEWithLogitsLoss()

	# define infer
	inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor,Adain=adain)
	optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-4)

	lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_diff, 'min', patience=2, verbose=True)
	lr_scheduler_s_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s_D, 'min', patience=2, verbose=True)


	content_loss = 'l2'

	if content_loss == 'l2': 
		latent_content_loss = F.mse_loss

	if content_loss:
		print(f'Using {content_loss} content loss')



	n_epochs = 80
	val_interval = 5
	accumulation_steps = 4 # effective batch size = batch size * accumulation_steps

	scaler = GradScaler()
	gs = epoch_start * (len(train_loader_src)//bs)

	if Resume:
		state_dict_ldm = torch.load(resume_from)
		unet.load_state_dict(state_dict_ldm['unet_state_dict'])
		s_D.load_state_dict(state_dict_ldm['s_D_state_dict'])
		optimizer_diff.load_state_dict(state_dict_ldm['optimizer_diff'])
		optimizer_s_D.load_state_dict(state_dict_ldm['optimizer_s_D'])
		scaler.load_state_dict(state_dict_ldm['scaler_state_dict'])
		lr_scheduler.load_state_dict(state_dict_ldm['lr_scheduler_state_dict'])
		lr_scheduler_s_D.load_state_dict(state_dict_ldm['lr_scheduler_s_D_state_dict'])
		tqdm.write('All cLDM state dict loaded!')



	for epoch in range(epoch_start,(epoch_start+n_epochs)):
		unet.train()
		epoch_loss = 0
		epoch_diff_loss = 0
		epoch_content_loss = 0
		epoch_style_loss = 0
		epoch_Adv_style_loss = 0
		epoch_D_style_loss = 0
		epoch_gradient_loss = 0
		progress_bar = tqdm(zip(enumerate(train_loader_src),cycle(train_loader_tar)),total=len(train_loader_src), ncols=150)
		progress_bar.set_description(f"Epoch {epoch}")
		for ((step, batch_src), batch_tar) in progress_bar:
			gs +=1

			images = batch_src["image"].to(device)  
			conditions_img = batch_tar['image'].to(device) 
			conditions = autoencoder.encode_stage_2_inputs(conditions_img) * scale_factor

			optimizer_diff.zero_grad(set_to_none=True)
			optimizer_s_D.zero_grad(set_to_none=True)


			with autocast(enabled=True):
				# Generate random noise
				noise = torch.randn_like(z).to(device)

				# Create timesteps
				timesteps = torch.randint(
					0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
				).long()

				# Get model prediction
				noise_pred, noisy_image, latent, _ = inferer(
					inputs=images, autoencoder_model=autoencoder, diffusion_model=unet, noise=noise, timesteps=timesteps,
					condition=conditions, mode='concat'
				) 

				del conditions_img, _
				torch.cuda.empty_cache()

				if diff_loss_type == 'L2':
					loss_diff = F.mse_loss(noise_pred.float(), noise.float())
				else:
					loss_diff = F.l1_loss(noise_pred.float(), noise.float())
				noisy_image = noisy_image[:,:noisy_image.shape[1]//2,:,:,:] # since noisy_image returned are concat of noisy_image and condition
			

				if content_loss == 'l2':
					x0_pred = torch.zeros_like(z).to(device)

					for n in range (len(noise_pred)):
						_, x0_pred[n] = scheduler.step(torch.unsqueeze(noise_pred[n,:,:,:,:], 0), timesteps[n], torch.unsqueeze(noisy_image[n,:,:,:,:], 0))
	 

					## latent level loss
					if use_IN:
						loss_content = latent_content_loss(IN(latent.float()),IN(x0_pred.float()))
					else:
						loss_content = latent_content_loss(latent.float(),x0_pred.float()) 

					# gradient_map loss
					grad_map_org = util.torch_gradmap(images.float()*brain_mask)
					dec = autoencoder.decode_stage_2_outputs((x0_pred/scale_factor).float())
					grad_map_rec = util.torch_gradmap(dec.float()*brain_mask)
					loss_grad = util.grad_loss(grad_map_org,grad_map_rec,loss_type=grad_loss_type)

					## style loss
					if style_loss:
						loss_style = Style_loss(x0_pred.float(),conditions.float()) # gram/mean style loss
	
						s_real = s_D(conditions.float().detach()).view(-1)				# discriminate  3D latent
						errD_real = style_criterion(s_real, torch.ones(s_real.size()).to(device))
						s_fake = s_D(latent.float().detach()).view(-1) 	# discriminate 3D real source latent
						errD_fake = style_criterion(s_fake, torch.zeros(s_fake.size()).to(device))
						# update discriminator
						errD = errD_fake + errD_real
						errD.backward()
	  
						if (step+1) % (accumulation_steps*1) == 0:
							optimizer_s_D.step()
							optimizer_s_D.zero_grad(set_to_none=True)
	  
						if epoch > burn_in:
							s_recon = s_D(x0_pred.float()).view(-1) 	# discriminate 3D latent
							loss_style_adv = style_criterion(s_recon, torch.ones(s_recon.size()).to(device))

							loss = diff_loss_weight * loss_diff + content_weight * loss_content + style_weight * loss_style + adv_style_weight * loss_style_adv + gradient_weight * loss_grad
						else:
							loss = diff_loss_weight * loss_diff + content_weight * loss_content + style_weight * loss_style + gradient_weight * loss_grad
							loss_style_adv = torch.tensor([0.0])
					else:
						loss_style = torch.tensor([0.0])
						loss = diff_loss_weight * loss_diff + content_weight * loss_content + gradient_weight * loss_grad

				else:
					loss_content = torch.tensor([0.0])
					loss = loss_diff



			scaler.scale(loss).backward()
			del noise_pred, noisy_image, noise, latent, conditions, grad_map_org, grad_map_rec, dec
			torch.cuda.empty_cache()

			# implement gradient accumulation
			if (step+1) % accumulation_steps == 0:
				scaler.step(optimizer_diff)
				scaler.update()
				optimizer_diff.zero_grad(set_to_none=True)

			epoch_loss += loss.item()
			epoch_diff_loss += loss_diff.item()
			epoch_content_loss += loss_content.item()
			epoch_style_loss += loss_style.item()
			epoch_Adv_style_loss += loss_style_adv.item()
			epoch_D_style_loss += errD.item()
			epoch_gradient_loss += loss_grad.item()

			progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
			
		writer.add_scalar('diffusion loss',epoch_diff_loss / (step + 1),epoch)
		writer.add_scalar('content loss',epoch_content_loss / (step + 1),epoch)
		writer.add_scalar('style loss',epoch_style_loss / (step + 1),epoch)
		writer.add_scalar('Adversarial style loss',epoch_Adv_style_loss / (step + 1),epoch)
		writer.add_scalar('style D loss',epoch_D_style_loss / (step + 1),epoch)
		writer.add_scalar('gradient loss',epoch_gradient_loss / (step + 1),epoch)
		writer.add_scalar('total loss',epoch_loss / (step + 1),epoch)



		if (epoch + 1) % val_interval == 0:
			now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")

			torch.save({
			'unet_state_dict': unet.state_dict(),
			's_D_state_dict': s_D.state_dict(),
			'optimizer_diff': optimizer_diff.state_dict(),
			'optimizer_s_D': optimizer_s_D.state_dict(),
			'scaler_state_dict': scaler.state_dict(),
			'lr_scheduler_state_dict': lr_scheduler.state_dict(),
			'lr_scheduler_s_D_state_dict': lr_scheduler_s_D.state_dict()
				}, save_dir/f'ep_{epoch}_ckp_{now}.pth')
			
			autoencoder.eval()
			unet.eval()
			s_D.eval()
			val_loss_total = 0
			val_loss_diff_total = 0
			val_loss_content_total = 0
			val_loss_style_total = 0
			val_loss_style_total_adv = 0
			val_loss_s_D_total = 0
			val_loss_gradient_total = 0
			val_ssim_total = 0

			with torch.no_grad():
				for ((val_step, batch_src_val), batch_tar_val) in zip(enumerate(val_loader_src, start=1),cycle(val_loader_tar)):
					results = {}
					images = batch_src_val["image"].to(device)
					conditions_img = batch_tar_val['image'].to(device)
					conditions = autoencoder.encode_stage_2_inputs(conditions_img)*scale_factor

					# Generate random noise
					noise = torch.randn_like(z).to(device)

					# Create timesteps
					timesteps = torch.randint(
						0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
					).long()

					# Get model prediction
					noise_pred, noisy_image, latent, _ = inferer(
						inputs=images, autoencoder_model=autoencoder, diffusion_model=unet, noise=noise, timesteps=timesteps,
						condition=conditions, mode='concat'
					)
     
					del conditions_img, _
					torch.cuda.empty_cache()
     
					if diff_loss_type == 'L2':
						val_loss_diff = F.mse_loss(noise_pred.float(), noise.float())
					else:
						val_loss_diff = F.l1_loss(noise_pred.float(), noise.float())
					noisy_image = noisy_image[:,:noisy_image.shape[1]//2,:,:,:] # since noisy_image returned are concat of noisy_image and condition

					
					if content_loss == 'l2':
						x0_pred = torch.zeros_like(z).to(device)
						for n in range (len(noise_pred)):
							_, x0_pred[n] = scheduler.step(torch.unsqueeze(noise_pred[n,:,:,:,:], 0), timesteps[n], torch.unsqueeze(noisy_image[n,:,:,:,:], 0))
	  
					
						## latent level loss
						if use_IN:
							val_loss_content = latent_content_loss(IN(latent.float()), IN(x0_pred.float()))
						else:
							val_loss_content = latent_content_loss(latent.float(), x0_pred.float())
	  
						# gradient_map loss
						grad_map_org = util.torch_gradmap(images.float()*brain_mask)
						dec = autoencoder.decode_stage_2_outputs((x0_pred/scale_factor).float())
						grad_map_rec = util.torch_gradmap(dec.float()*brain_mask)
						val_loss_grad = util.grad_loss(grad_map_org,grad_map_rec,loss_type=grad_loss_type)
	
						## style loss
						if style_loss:
							val_loss_style = Style_loss(x0_pred.float(),conditions.float())

							s_real = s_D(conditions.float()).view(-1) 				# discriminate  3D latent
							errD_real = style_criterion(s_real, torch.ones(s_real.size()).to(device))

							s_fake = s_D(latent.float().detach()).view(-1)	# discriminate 3D real source latent
							errD_fake = style_criterion(s_fake, torch.zeros(s_fake.size()).to(device))
							val_errD = errD_fake + errD_real

							s_recon = s_D(x0_pred.float()).view(-1) 			# discriminate full 3D latent
							val_loss_style_adv = style_criterion(s_recon, torch.ones(s_recon.size()).to(device))
							val_loss = diff_loss_weight * val_loss_diff + content_weight * val_loss_content + style_weight * val_loss_style + adv_style_weight * val_loss_style_adv+ gradient_weight * val_loss_grad

						else:
							val_loss_style = torch.tensor([0.0])
							val_loss = diff_loss_weight * val_loss_diff + content_weight * val_loss_content + gradient_weight * val_loss_grad

						del noise_pred, noisy_image, noise, latent, _, grad_map_org, grad_map_rec, dec, s_real, s_fake
						torch.cuda.empty_cache()
					else:
						val_loss_content = torch.tensor([0.0])
						val_loss = diff_loss_weight * val_loss_diff

						del noise_pred, noisy_image, noise, latent, conditions, conditions_img
						torch.cuda.empty_cache()


					# get the first sammple from the first validation batch for visualisation
					# purposes
					if val_step == 1:
						tqdm.write('generating samples...')
						conditions = autoencoder.encode_stage_2_inputs(batch_tar_val['image'].to(device))*scale_factor


						results['inputs_src']=images.detach().cpu().float()
						results['cond_tar']= batch_tar_val['image']

						z_x = autoencoder.encode_stage_2_inputs(images) * scale_factor
						del images
						torch.cuda.empty_cache()

						# reverse DDIM sampling to add noise
						scheduler_ddim.set_timesteps(num_inference_steps=num_inference_fdp)
						img_noisy, latent_noisy = inferer.reverse_sample(
							input_noise=z_x, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler_ddim,
							conditioning=conditions,mode='concat', verbose=False, return_latent=True
						)
						results['noisy']=img_noisy.detach().cpu().float()
	  
						scheduler_ddim.set_timesteps(num_inference_steps=num_inference_rdp)
						recon_images = inferer.sample(
							input_noise=latent_noisy, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler_ddim,
							conditioning=conditions,mode='concat'
						)
						results['recon']=recon_images.detach().cpu().float()
						del recon_images
						torch.cuda.empty_cache()

						scheduler_ddim.set_timesteps(num_inference_steps=num_inference_rdp)
						recon_images = inferer.sample(
							input_noise=latent_noisy, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler_ddim,
							conditioning=conditions,mode='concat', GN=False
						)
						results['recon_noGN']=recon_images.detach().cpu().float()
						del recon_images, z_x
						torch.cuda.empty_cache()

						noise = torch.randn_like(z)[0].unsqueeze(0)
						noise = noise.to(device)
						scheduler_ddim.set_timesteps(num_inference_steps=num_inference_rdp)
						synthetic_images = inferer.sample(
							input_noise=noise, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler_ddim,
							conditioning=conditions,mode='concat'
						)
						results['syn_src2tar']=synthetic_images.detach().cpu().float()
						del synthetic_images, noise
						torch.cuda.empty_cache()

						results['error_recon']=torch.abs(results['inputs_src']-results['recon'])
						results['error_nGN']=torch.abs(results['inputs_src']-results['recon_noGN'])

					
						root = save_dir / 'images'/'val'
						if not root.exists():
							os.makedirs(root)
						for k in results:
							img_volume = results[k].detach().cpu() # [B,1,W,H,Z]
							img_volume[img_volume<0]=0


							img_volume = (img_volume-img_volume.min())/(img_volume.max()-img_volume.min()) # -> [0,1]

							grid_a = torchvision.utils.make_grid(img_volume[:,:,:,:,img_volume.shape[4]//2], nrow=1,normalize=True) # axial middle slices
							grid_a = grid_a.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
							grid_a = (grid_a * 255).astype(np.uint8)
							filename = "{}_gs-{:06}_e-{:04}_{}.png".format(k,gs,epoch,'a')
							save_path = root / filename
							Image.fromarray(grid_a).save(save_path)


							grid_c = torchvision.utils.make_grid(img_volume[:,:,:,img_volume.shape[3]//2,:], nrow=4,normalize=True) # coronal middle slice
							grid_c = grid_c.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
							grid_c = (grid_c * 255).astype(np.uint8)
							filename = "{}_gs-{:06}_e-{:04}_{}.png".format(k,gs,epoch,'c')
							save_path = root / filename
							Image.fromarray(grid_c).save(save_path)

							grid_s = torchvision.utils.make_grid(img_volume[:,:,img_volume.shape[2]//2,:,:], nrow=4,normalize=True) # saggital middle slice
							grid_s = grid_s.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
							grid_s = (grid_s * 255).astype(np.uint8)
							filename = "{}_gs-{:06}_e-{:04}_{}.png".format(k,gs,epoch,'s')
							save_path = root / filename
							Image.fromarray(grid_s).save(save_path)

					val_loss_total += val_loss.item()
					val_loss_diff_total += val_loss_diff.item()
					val_loss_content_total += val_loss_content.item()
					val_loss_style_total += val_loss_style.item()
					val_loss_s_D_total += val_errD.item()
					val_loss_style_total_adv += val_loss_style_adv.item()
					val_loss_gradient_total += val_loss_grad.item()
     
					

			val_loss_total /= val_step
			val_loss_diff_total /= val_step
			val_loss_content_total /= val_step
			val_loss_style_total /= val_step
			val_loss_style_total_adv /= val_step
			val_loss_s_D_total /= val_step
			val_loss_gradient_total /= val_step

			lr_scheduler.step(val_loss_total)
			lr_scheduler_s_D.step(val_loss_s_D_total)
			writer.add_scalar('val total loss',val_loss_total,epoch)
			writer.add_scalar('val diffusion loss',val_loss_diff_total,epoch)
			writer.add_scalar('val content loss',val_loss_content_total,epoch)
			writer.add_scalar('val style loss',val_loss_style_total,epoch)
			writer.add_scalar('val adversarial style loss',val_loss_style_total_adv,epoch)
			writer.add_scalar('val style D loss',val_loss_s_D_total,epoch)
			writer.add_scalar('val gradient loss',val_loss_gradient_total,epoch)


			tqdm.write(f'val_loss = {val_loss_total}')





if __name__ == "__main__":
	
	# load dataloader
	bs = 1

	train_dataset_src = MRI.IXI_train_3d_src() # train source dataset
	val_dataset_src = MRI.IXI_val_3d_src() # val source dataset
	train_dataset_tar = MRI.IXI_train_3d_tar() # train target dataset
	val_dataset_tar = MRI.IXI_val_3d_tar() # val target dataset

	train_loader_src = DataLoader(train_dataset_src, batch_size=bs, shuffle=True, num_workers=3, persistent_workers=True,drop_last=True)
	val_loader_src = DataLoader(val_dataset_src, batch_size=bs, shuffle=True, num_workers=3, persistent_workers=True,drop_last=True)
	# tar
	train_loader_tar = DataLoader(train_dataset_tar, batch_size=bs, shuffle=True, num_workers=3, persistent_workers=True,drop_last=True)
	val_loader_tar = DataLoader(val_dataset_tar, batch_size=bs, shuffle=True, num_workers=3, persistent_workers=True,drop_last=True)

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



	adain = 'AdaIN_reverse'
	style_loss = True
	style_weight = 10
	adv_style_weight = 1
	content_weight = 10
	gradient_weight =10000
	diff_loss_weight = 1
	grad_loss_type = 'l2'
	use_IN = True
	burn_in = 5
	num_train_ddim = 50
	num_inference_fdp = 30
	num_inference_rdp = 10
	diff_loss_type = 'L2'
	use_mask = False
 
	run_name = 'sample_run'


	print(f'{now}_{run_name}')

	Resume = False
	if Resume:
		resume_from = Path('sample_path/cLDM_ckp.pth')
		epoch_start = 40
		print(f'Resume from: {resume_from}, epoch {epoch_start}')
		save_dir = resume_from.parent
		writer=SummaryWriter(f'runs/{save_dir.name}')

	else:
		epoch_start = 0
		writer=SummaryWriter(f'runs/{now}_{run_name}')
		save_dir = Path(f'log/aekl_train/{now}_{run_name}')


	if not save_dir.exists():
		os.makedirs(save_dir)
	elif not Resume:
		assert len(os.listdir(save_dir))==0,'Log dir exist!'

	# define model
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using {device}")
 
	if use_mask:
		brain_mask = torch.from_numpy(np.load('sample_path/mean_brain_mask.npy')).unsqueeze(0).float().to(device)
	else:
		brain_mask = torch.ones_like(brain_mask).to(device)


	parameters = f"""AdaIN = {adain},
	Style_loss = {S_loss_type},
	style_weight = {style_weight},
	adv_style_weight = {adv_style_weight},
	useIN (in Closs) = {use_IN},
	content_weight = {content_weight},
	{grad_loss_type} gradient loss with weight = {gradient_weight},
	diffusion loss weight = {diff_loss_weight},
	burn_in = {burn_in},
	num_inference_fdp = {num_inference_fdp},
	num_inference_rdp = {num_inference_rdp},
	num_train_ddim = {num_train_ddim}"""
  
	print(parameters)
	with open(save_dir / 'parameters.txt', 'w') as f:
		f.write(run_name+'\n')
		f.write(parameters)

	train()
 
