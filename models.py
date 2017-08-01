import tensorflow as tf
import tensorflow.contrib.slim as slim

def G_conv(z, size=4, channel=3, name='G_conv'): #default for cifar-10 data
	with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
		with slim.arg_scope([slim.conv2d_transpose], padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02)):
			net = slim.fully_connected(z, size * size * 512, scope='fc1')
			net = tf.reshape(net, (-1, size, size, 512))
			net = slim.conv2d_transpose(net, 256, 3, stride=2, scope='deconv1')
			net = slim.conv2d_transpose(net, 128, 3, stride=2, scope='deconv2')
			net = slim.conv2d_transpose(net, 64, 3, stride=2, scope='deconv3')
			net = slim.conv2d_transpose(net, channel, 3, stride=2, activation_fn=tf.nn.sigmoid, scope='deconv4')

			return net

def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		l1 = 0.5 * (1 + leak)
		l2 = 0.5 * (1 - leak)
		l = l1 * x + l2 * abs(x)
		return l

def D_AE(x, size=32, hidden_num=256, reuse=False, name='D_AE'):
	with tf.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()
		#Encode
		with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=lrelu, weights_initializer=tf.random_normal_initializer(0, 0.02)):
			with slim.arg_scope([slim.conv2d], kernel_size=3, stride=2, normalizer_fn=slim.batch_norm, padding='SAME', data_format=DATA_FORMAT_NHWC):
				e = slim.conv2d(x, size, scope='conv1')
				e = slim.conv2d(e, size * 2, scope='conv2')
				e = slim.conv2d(e, size * 4, scope='conv3')
				e = slim.conv2d(e, size * 8, scope='conv4')
				#e = slim.flatten(e, scope='flatten1')
				e = slim.fully_connected(e, hidden_num, scope='fc1')
		
		#Decode
		with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
			with slim.arg_scope([slim.conv2d_transpose], stride=2, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02)):
				d = slim.fully_connected(e, 4 * 4 * 512, scope='fc1-decode')
				d = tf.reshape(d, (-1, 4, 4, 512))
				d = slim.conv2d_transpose(d, 256, 3, scope='deconv1')
				d = slim.conv2d_transpose(d, 128, 3, scope='deconv2')
				d = slim.conv2d_transpose(d, 32, 3, scope='deconv3')
				d = slim.conv2d_transpose(d, 3, 3, activation_fn=tf.nn.sigmoid, scope='deconv4')

		return d
				





