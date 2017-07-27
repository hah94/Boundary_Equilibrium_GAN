import tensorflow as tf
import numpy as np
from data_loader import *
from models import *
from PIL import Image


class BEGAN():
	def __init__(self, data):
		self.data = data
		self.z_dim = 100
		self.size = 32
		self.channel = 3
		self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
		self.k = tf.placeholder(tf.float32, shape=[])
		gamma = 0.5 # diversity / quality parameter
		k_lr = 0.001 # learning rate of k

		self.Gen_sample = G_conv(self.z)
		self.D_real = D_AE(self.X)
		self.D_fake = D_AE(self.Gen_sample, reuse=True)

		Loss_real = tf.reduce_mean(tf.abs(self.X - self.D_real))
		Loss_fake = tf.reduce_mean(tf.abs(self.Gen_sample - self.D_fake))
		self.G_loss = Loss_fake
		self.D_loss = Loss_real - Loss_fake * self.k
		self.kn = self.k + k_lr * (gamma * Loss_real - Loss_fake)
		self.M = Loss_real + tf.abs(gamma * Loss_real - Loss_fake)

		self.lr = tf.placeholder(tf.float32, shape=[])
		opt = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.D_optimizer = opt.minimize(self.D_loss)
		self.G_optimizer = opt.minimize(self.G_loss)
		self.sess = tf.Session()

	def train(self, sample_dir, total_epoch=500000, batch_size=16):
		self.sess.run(tf.global_variables_initializer())
		init_lr = 1e-4
		kn = 0

		for epoch in range(total_epoch):
			lr = init_lr * pow(0.5, epoch // 50000)
			idxs = np.random.permutation(self.data.shape[0])
			x = self.data[idxs]
			for i in range(50000 // batch_size):
				batch_X = x[i * batch_size : (i+1) * batch_size]
				batch_z = np.random.uniform(-1., 1., size=[batch_size, self.z_dim])
				_, _, kn = self.sess.run([self.D_optimizer, self.G_optimizer, self.kn], 
										 feed_dict = {self.X: batch_X, self.z: batch_z, self.k: min(max(kn,0.), 1.), self.lr: lr})
				
			if (epoch % 1000 == 0) :
				D, G, M = self.sess.run([self.D_loss, self.G_loss, self.M_global],
										feed_dict = {self.X: batch_X, self.z: batch_z, self.k: min(max(kn, 0.), 1.), self.lr: lr})
				print('Epoch: {}, Dloss: {:.4}, Gloss: {:.4}, M_global: {:.4}, k_value: {:.6}, learning_rate: {:.8}'.
					  format(epoch, D, G, M, min(max(kn, 0.), 1.), lr))

				sample_X, D_real, samples = self.sess.run([self.X, self.D_real, self.Gen_sample],
														  feed_dict={self.X: batch_X[:16,:,:,:], 
														  	         self.z: np.random.uniform(-1., 1., size=[16, self.z_dim])})
				im1 = Image.fromarray(sample_X)
				im1.save('{}/{}_X.png'.format(sample_dir, epoch))
				im2 = Image.fromarray(D_real)
				im2.save('{}/{}_D_real.png'.format(sample_dir, epoch))
				im3 = Image.fromarray(samples)					
				im3.save('{}/{}_G.png'.format(sample_dir, epoch))

			if (epoch % 10000 == 0) :
				check_dir = 'checkpoint/'
				if not os.path.exists(check_dir):
					os.makedirs(check_dir)
				tf.train.saver().save(self.sess, 'checkpoint/began_epoch_{}.ckpt'.format(epoch))

if __name__ == '__main__' :
	os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5,6'
	sample_dir = 'samples/'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)
	
	data_all = load_data()
	data = data_all['images_train']

	began = BEGAN(data)
	began.train(sample_dir)

										 
				
