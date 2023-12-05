import os
import glob
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


class ImageFolder(data.Dataset):
	def __init__(self, root,image_size=512,mode='train'):
		"""Initializes image paths and preprocessing module."""
		self.root = root

		# GT : Ground Truth
		self.image_paths = glob.glob(os.path.join(root, "*.png"))
		self.image_size = image_size
		self.mode = mode
		self.RotationDegree = [0,90,180,270]

		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[index]
		image = Image.open(image_path)
		filename = image_path.split('_')[-1][:-len(".png")].split('/')[-1] + '.png'

		Transform = []
		Transform.append(T.Resize((512,512)))
		Transform.append(T.ToTensor())
		Transform = T.Compose(Transform)

		#image = np.array(image)/65535.0
		image = Transform(image)
		image = image.float()
		Norm_ = T.Normalize((0.5), (0.5))
		image = Norm_(image)
		#image = image.repeat(3, 1, 1)
		sample = {'img': image, 'img_name': filename}
		return sample

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train'):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode)
	#train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
	data_loader = data.DataLoader(dataset=dataset,
								  shuffle=False,
								  num_workers=1,
								  pin_memory=False
								  )
	return data_loader
