'''Imports CIFAR-10 data.'''
import os
import urllib.request
import zipfile
import tarfile
import numpy as np
import pickle
import sys

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
data_dir = 'data/'

def maybe_download_and_extract():
	filename = url.split('/')[-1]
	file_path = os.path.join(data_dir, filename)
	
	if not os.path.exists(file_path):
		if not os.path.exists(data_dir):
			os.makedirs(data_dir)

		file_path, _ = urllib.request.urlretrieve(url=url, filename=file_path)

		print()
		print("Download finished. Extracting files.")

		if file_path.endswith(".zip"):
			zipfile.Zipfile(file=file_path, mode="r").extractall(data_dir)
		elif file_path.endswith(".tar.gz"):
			tarfile.open(name=file_path, mode="r:gz").extractall(data_dir)

		print ("Done.")
	else:
		print("Data already exists")
	
def load_CIFAR10_batch(filename):
  '''load data from single CIFAR-10 file'''

  with open(filename, 'rb') as f:
    if sys.version_info[0] < 3:
      dict = pickle.load(f)
    else:
      dict = pickle.load(f, encoding='latin1')
    x = dict['data']
    y = dict['labels']
    x = x.astype(float)
    y = np.array(y)
  return x, y

def load_data():
  '''load all CIFAR-10 data and merge training batches'''
  maybe_download_and_extract()
  xs = []
  ys = []
  for i in range(1, 6):
    filename = 'data/cifar-10-batches-py/data_batch_' + str(i)
    X, Y = load_CIFAR10_batch(filename)
    xs.append(X)
    ys.append(Y)

  x_train = np.concatenate(xs)
  y_train = np.concatenate(ys)
  del xs, ys

  classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck']

  # Normalize Data
  mean_image = np.mean(x_train, axis=0)
  x_train -= mean_image

  data_dict = {
    'images_train': x_train,
    'labels_train': y_train,
    'classes': classes
  }
  im_tr = np.array(data_dict['images_train'])
  im_tr = np.reshape(im_tr, (-1, 3, 32, 32))
  im_tr = np.transpose(im_tr, (0,2,3,1))
  data_dict['images_train'] = im_tr
  return data_dict

#def generate_random_batch(images, labels, batch_size):
  # Generate batch
  #indices = np.random.choice(images.shape[0], batch_size)
  #images_batch = images[indices]
  #labels_batch = labels[indices]
  #return images_batch, labels_batch

def gen_batch(data, batch_size, num_iter):
  data = np.array(data)
  index = len(data)
  for i in range(num_iter):
    index += batch_size
    if (index + batch_size > len(data)):
      index = 0
      shuffled_indices = np.random.permutation(np.arange(len(data)))
      data = data[shuffled_indices]
    yield data[index:index + batch_size]

def main():
  data_sets = load_data()
  print(data_sets['images_train'].shape)
  print(data_sets['labels_train'].shape)

if __name__ == '__main__':
  main()
