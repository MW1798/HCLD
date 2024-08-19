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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance


def get_mean_std(input, eps=1e-6):
	B, C = input.shape[:2]
	mean = torch.mean(input.view(B,C,-1),dim=2).view(B,C,1,1,1) # mean shape (B, C, 1, 1)
	std = torch.sqrt(torch.var(input.view(B,C,-1), dim=2)+eps).view(B, C, 1, 1,1)
	
	return mean, std

def AdaIN(content, style):
	assert content.shape[:2] == style.shape[:2]
	c_mean, c_std = get_mean_std(content)
	s_mean, s_std = get_mean_std(style)

	
	normalized = s_std*((content-c_mean)/c_std) + s_mean
	
	return normalized

def IN(content):
	c_mean, c_std = get_mean_std(content)

	
	normalized = (content-c_mean)/c_std
	return normalized

def Style_loss(input, target):
	mean_loss, std_loss = 0, 0

	for input_layer, target_layer in zip(input, target): 
		mean_input_layer, std_input_layer = get_mean_std(input_layer)
		mean_target_layer, std_target_layer = get_mean_std(target_layer)

		mean_loss += F.mse_loss(mean_input_layer, mean_target_layer)
		std_loss += F.mse_loss(std_input_layer, std_target_layer)

	return mean_loss+std_loss

# Gram matrix and Style loss
def gram_matrix(input):
	b, c, h, w, d = input.size()  # a=batch size(=1)


	features = input.view(b * c, h * w * d)  # resise F_XL into \hat F_XL

	G = torch.mm(features, features.t())  # compute the gram product

	# we 'normalize' the values of the gram matrix
	# by dividing by the number of element in each feature maps.
	return G.div(b * c * h * w * d)

def style_loss_gram(input,target):
	target_G = gram_matrix(target)
	input_G = gram_matrix(input)
	loss = F.mse_loss(input_G, target_G)
	
	return loss

def torch_gradmap(img):
	dh = img[:,:,:,:,1:] - img[:,:,:,:,:-1]
	dw = img[:,:,:,1:,:] - img[:,:,:,:-1,:]
	dz = img[:,:,1:,:,:] - img[:,:,:-1,:,:]
	gra_map = (dh[:,:,1:, 1:, :] + dw[:,:,1:,:,1:] + dz[:,:,:,1:,1:])/3.
	gra_map=torch.nn.functional.pad(gra_map, (1,0,1,0,1,0), "constant", 0) 
	return gra_map

def grad_loss(grad_map1, grad_map2, loss_type):
	if loss_type == 'l1':
		return F.l1_loss(grad_map1, grad_map2)
	elif loss_type == 'l2':
		return F.mse_loss(grad_map1, grad_map2)
	else:
		raise ValueError("Wrong loss type, either l1 or l2")


# Define the MLP-based style discriminator
class Style_Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Style_Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input.view(input.size(0), -1))
    

class Style_Discriminator_3d(nn.Module):
	def __init__(self):
		super(Style_Discriminator_3d, self).__init__()
		self.main = nn.Sequential(
			nn.Conv3d(6, 64, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv3d(64, 128, 4, 2, 1, bias=False),
			nn.InstanceNorm3d(128),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv3d(128, 256, 4, 2, 1, bias=False),
			nn.InstanceNorm3d(256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Flatten(),
			nn.Linear(256*6*6*2, 1)
		)

	def forward(self, input):
		return self.main(input)


def calculate_intensity_correlation(image1, image2):
    # Ensure the images are numpy arrays
    image1 = np.array(image1)
    image2 = np.array(image2)

    # Calculate the mean of the images
    mean1 = np.mean(image1)
    mean2 = np.mean(image2)

    # Subtract the mean from the images
    image1 -= mean1
    image2 -= mean2

    # Calculate the product of the images
    product = image1 * image2

    # Calculate the sum of the product
    sum_product = np.sum(product)

    # Calculate the sum of the squares of the images
    sum_square1 = np.sum(image1 ** 2)
    sum_square2 = np.sum(image2 ** 2)

    # Calculate the Pearson correlation coefficient
    correlation = sum_product / np.sqrt(sum_square1 * sum_square2)

    return correlation.mean()

def mean_wasserstein_distance(list1, list2):
    # Flatten the images and compute the Wasserstein distance for each pair
    distances = [wasserstein_distance(img1.flatten(), img2.flatten()) for img1, img2 in zip(list1, list2)]
    
    # Return the mean distance
    return np.mean(distances)