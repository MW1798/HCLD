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
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

import shutil
import tempfile
from pathlib import Path
import datetime

import data.MRIdata as MRI

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator

from torch.utils.tensorboard import SummaryWriter


torch.set_float32_matmul_precision('medium')


def train():
	# training
	n_epochs = 30
	val_interval = 2
	accumulation_steps = 1 # effective batch size = batch size * accumulation_steps

 
	epoch_recon_loss_list = []
	epoch_gen_loss_list = []
	epoch_disc_loss_list = []
	val_recon_epoch_loss_list = []
	intermediary_images = []
	n_example_images = 4
	gs = 0



	model = AutoencoderKL(
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
	model.to(device)


	discriminator = PatchDiscriminator(
		spatial_dims=3,
		num_layers_d=3,
		num_channels=32,
		in_channels=1,
		out_channels=1,
		kernel_size=4,
		activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
		norm="BATCH",
		bias=False,
		padding=1,
	)
	discriminator.to(device)

	# define loss and optimizers
	perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", fake_3d_ratio=0.25)
	perceptual_loss.to(device)

	adv_loss = PatchAdversarialLoss(criterion="least_squares")
	adv_weight = 0.01
	# perceptual_weight = 0.01
	perceptual_weight = 0.1
	kl_weight = 1e-6

	# create optimizers
	optimizer_g = torch.optim.Adam(model.parameters(), 1e-4)
	optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-4)

	# Create schedulers
	scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, 'min', patience=10, verbose=True)
	scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, 'min', patience=10, verbose=True)

	# create gradient scaler
	scaler_g = torch.cuda.amp.GradScaler()
	scaler_d = torch.cuda.amp.GradScaler()

	if not state_dict is None:
		model.load_state_dict(state_dict['model_state_dict'])
		discriminator.load_state_dict(state_dict['discriminator_state_dict'])
		optimizer_g.load_state_dict(state_dict['optimizer_g_state_dict'])	
		optimizer_d.load_state_dict(state_dict['optimizer_d_state_dict'])
		scheduler_g.load_state_dict(state_dict['scheduler_g_state_dict'])
		scheduler_d.load_state_dict(state_dict['scheduler_d_state_dict'])
		scaler_g.load_state_dict(state_dict['scaler_g_state_dict'])
		scaler_d.load_state_dict(state_dict['scaler_d_state_dict'])
		print('All state dict loaded!')
 
 
	try:
		for epoch in range(epoch_start,n_epochs):
			model.train()
			discriminator.train()
			epoch_loss = 0
			gen_epoch_loss = 0
			disc_epoch_loss = 0
			progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=150)
			progress_bar.set_description(f"Epoch {epoch}")
			for step, batch in progress_bar:
				gs +=1
				images = batch["image"].to(device)
				optimizer_g.zero_grad(set_to_none=True)

				# Generator part
				with autocast(enabled=True):
					reconstruction, z_mu, z_sigma = model(images)
					logits_fake = discriminator(reconstruction.contiguous().float())[-1]

					recons_loss = F.l1_loss(reconstruction.float(), images.float())
					p_loss = perceptual_loss(reconstruction.float(), images.float())
					generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

					kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
					kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

					loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss) + (adv_weight * generator_loss)

				scaler_g.scale(loss_g).backward()
				del z_mu, z_sigma
				torch.cuda.empty_cache()


    
				# implement gradient accumulation
				if (step+1) % accumulation_steps == 0:
					scaler_g.step(optimizer_g)
					scaler_g.update()

				# Discriminator part
				optimizer_d.zero_grad(set_to_none=True)

				with autocast(enabled=True):
					logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
					loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
					logits_real = discriminator(images.contiguous().detach())[-1]
					loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
					discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

					loss_d = adv_weight * discriminator_loss

				scaler_d.scale(loss_d).backward()
				del images, reconstruction
				torch.cuda.empty_cache()
    
				if (step+1) % accumulation_steps == 0:
					scaler_d.step(optimizer_d)
					scaler_d.update()
	

				epoch_loss += recons_loss.item()
				gen_epoch_loss += generator_loss.item()
				disc_epoch_loss += discriminator_loss.item()

				progress_bar.set_postfix(
					{
						"recons_loss": epoch_loss / (step + 1),
						"gen_loss": gen_epoch_loss / (step + 1),
						"disc_loss": disc_epoch_loss / (step + 1),
					}
				)

				writer.add_scalar('recons_loss',recons_loss.item(),gs)
				writer.add_scalar('preceptual_loss',p_loss.item(),gs)
				writer.add_scalar('kl_loss',kl_loss.item(),gs)
				writer.add_scalar('loss_g',loss_g.item(),gs)
				writer.add_scalar('loss_d',loss_d.item(),gs)

						 
			scheduler_g.step(loss_g)
			scheduler_d.step(loss_d)


			epoch_recon_loss_list.append(epoch_loss / (step + 1))
			epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
			epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))
   
			now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")

			torch.save({
			'model_state_dict': model.state_dict(),
			'discriminator_state_dict': discriminator.state_dict(),
			'optimizer_g_state_dict': optimizer_g.state_dict(),
			'optimizer_d_state_dict': optimizer_d.state_dict(),
			'scheduler_g_state_dict': scheduler_g.state_dict(),
			'scheduler_d_state_dict': scheduler_d.state_dict(),
			'scaler_g_state_dict': scaler_g.state_dict(),
			'scaler_d_state_dict': scaler_d.state_dict(),
				}, save_dir/f'ep_{epoch}_ckp_{now}.pth')

			if (epoch + 1) % val_interval == 0:
				model.eval()
				val_loss = 0
				with torch.no_grad():
					for val_step, batch in enumerate(val_loader, start=1):
						results = {}
						images = batch["image"].to(device)
						optimizer_g.zero_grad(set_to_none=True)

						reconstruction, z_mu, z_sigma = model(images)
						# get the first sammple from the first validation batch for visualisation
						# purposes
						if val_step == 1:
							# intermediary_images.append(reconstruction[:n_example_images, 0])
							results['inputs']=images.detach().cpu().float()
							results['recon']=reconstruction.detach().cpu().float()
							results['error']=torch.abs(results['inputs']-results['recon'])
							
							# root = os.path.join(save_dir, "images", 'val')
							root = save_dir / 'images'/'val'
							if not root.exists():
								os.makedirs(root)
							for k in results:
								img_volume = results[k].detach().cpu() # [B,1,W,H,Z], eg: torch.Size([2, 1, 192, 192, 64])
								img_volume[img_volume<0]=0

								img_volume = (img_volume-img_volume.min())/(img_volume.max()-img_volume.min()) # -> [0,1]

								grid_a = torchvision.utils.make_grid(img_volume[:,:,:,:,img_volume.shape[4]//2], nrow=1,normalize=True) # axial middle slices
								grid_a = grid_a.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
								grid_a = (grid_a * 255).astype(np.uint8)
								filename = "{}_gs-{:06}_e-{:04}_{}.png".format(k,gs,epoch,'a')
								save_path = root / filename
								Image.fromarray(grid_a).save(save_path)


								grid_c = torchvision.utils.make_grid(img_volume[:,:,:,img_volume.shape[3]//2,:], nrow=1,normalize=True) # coronal middle slice
								grid_c = grid_c.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
								grid_c = (grid_c * 255).astype(np.uint8)
								filename = "{}_gs-{:06}_e-{:04}_{}.png".format(k,gs,epoch,'c')
								save_path = root / filename
								Image.fromarray(grid_c).save(save_path)

								grid_s = torchvision.utils.make_grid(img_volume[:,:,img_volume.shape[2]//2,:,:], nrow=1,normalize=True) # saggital middle slice
								grid_s = grid_s.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
								grid_s = (grid_s * 255).astype(np.uint8)
								filename = "{}_gs-{:06}_e-{:04}_{}.png".format(k,gs,epoch,'s')
								save_path = root / filename
								Image.fromarray(grid_s).save(save_path)

								# grid = torchvision.utils.make_grid(results[k].detach().cpu(), nrow=4)
								# # if self.rescale:
								# grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
								# grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
								# grid = grid.numpy()
								# grid = (grid * 255).astype(np.uint8)
							
						recons_loss = F.l1_loss(reconstruction.float(), images.float())

						val_loss += recons_loss.item()
						del recons_loss, reconstruction, images, z_mu, z_sigma
						torch.cuda.empty_cache()

				val_loss /= val_step
				val_recon_epoch_loss_list.append(val_loss)
				writer.add_scalar('val recon loss',val_loss,epoch)
				del val_loss
				torch.cuda.empty_cache()


		progress_bar.close()
  
	except Exception as e:
		now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")
		print(now)
		print("error occured!",e)
		# Save state dictionaries
		torch.save({
			'model_state_dict': model.state_dict(),
			'discriminator_state_dict': discriminator.state_dict(),
			'optimizer_g_state_dict': optimizer_g.state_dict(),
			'optimizer_d_state_dict': optimizer_d.state_dict(),
			'scheduler_g_state_dict': scheduler_g.state_dict(),
			'scheduler_d_state_dict': scheduler_d.state_dict(),
			'scaler_g_state_dict': scaler_g.state_dict(),
			'scaler_d_state_dict': scaler_d.state_dict(),
		}, save_dir/f'ckp_{now}.pth')

	except KeyboardInterrupt:
		now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")
		print('Interrupted by user!')
		torch.save({
			'model_state_dict': model.state_dict(),
			'discriminator_state_dict': discriminator.state_dict(),
			'optimizer_g_state_dict': optimizer_g.state_dict(),
			'optimizer_d_state_dict': optimizer_d.state_dict(),
			'scheduler_g_state_dict': scheduler_g.state_dict(),
			'scheduler_d_state_dict': scheduler_d.state_dict(),
			'scaler_g_state_dict': scaler_g.state_dict(),
			'scaler_d_state_dict': scaler_d.state_dict(),
		}, save_dir/f'ckp_{now}.pth')


	
if __name__ == "__main__":
	# print_config()
	now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")

	run_name = 'sample_run'


	# load dataloader
	train_dataset = MRI.OpenBHB_train_3d()
	val_dataset = MRI.OpenBHB_val_3d()
	train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=3, persistent_workers=True)
	val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=3, persistent_workers=True)
	print(f'Train on {len(train_loader.dataset)}, validate on {len(val_loader.dataset)}')

	# define model
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using {device}")
 
	Resume = False
	if Resume:
		resume_from = Path('sample_path/ckp.pth')
		epoch_start = 4
		print(f'Resume from: {resume_from}, epoch {epoch_start}')
		save_dir = resume_from.parent
		writer=SummaryWriter(f'runs/{save_dir.name}')
		state_dict = torch.load(resume_from)

	else:
		epoch_start = 0
		writer=SummaryWriter(f'runs/{now}_{run_name}')
		save_dir = Path(f'log/aekl_train/{now}_{run_name}')
		state_dict = None
  
	if not save_dir.exists():
		os.makedirs(save_dir)
	else:
		assert len(os.listdir(save_dir))==0,'Log dir exist!'

	train()