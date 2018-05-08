"""
SEQ2SEQ IMAGE CAPTIONING
Tom Sherborne 8/5/18
"""
import tensorflow as tf

class ShapeWorldEncoder(object):
    def __init__(self, train_cnn=False):
        self.train_cnn = train_cnn
        
        # Input placeholder
        self.input_feed = None      # [batch, 64, 64, 3]
        self.phase = None           # [batch,]
    
    def build_model(self, image_feed, phase):
        self.input_feed = image_feed
        self.phase = phase
        return self.__build_model()
        
    def __build_model(self):
        """
        Build a conv net model
        """
        tf.logging.info("Building CNN graph...")
        
        # Cell Factory
        def create_conv_cell(inputs, kernel_size, stride, cell_name, is_trainable):
            """Create a single convolutional operator cell"""
            assert len(kernel_size) == 4, "Kernel size %s is incorrectly specified" % kernel_size
            assert len(stride) == 4, "Stride %s is incorrectly specified" % stride
            
            kernel = tf.get_variable(name=cell_name + "_kernel", shape=kernel_size, trainable=is_trainable)
            bias = tf.get_variable(name=cell_name + "_bias", shape=(kernel_size[-1]), trainable=is_trainable)
            
            conv = tf.nn.conv2d(inputs, kernel, stride, padding="SAME", name=cell_name + "_conv")
            conv = tf.nn.bias_add(conv, bias, name=cell_name + "_bias")
            conv = tf.contrib.layers.batch_norm(conv, center=True, scale=True, trainable=is_trainable, is_training=self.phase)
            conv = tf.nn.relu(conv, name=cell_name + "_relu")
            
            return conv
        
        # 3-deep layer factory
        def create_conv_layer(inputs, filter_size, prev_channels, num_filters, stride_len, pool, layer_name, is_trainable):
            """Create a set of 3 conv_cells all with the same stride and filter size"""
            initial_kernel = [filter_size, filter_size, prev_channels, num_filters]
            later_kernel = [filter_size, filter_size, num_filters, num_filters]
            
            stride = [1, stride_len, stride_len, 1]
            cell1 = create_conv_cell(inputs, initial_kernel, stride, layer_name + "_c1", is_trainable)
            cell2 = create_conv_cell(cell1, later_kernel, stride, layer_name + "_c2", is_trainable)
            cell3 = create_conv_cell(cell2, later_kernel, stride, layer_name + "_c3", is_trainable)
            
            if pool:
                cell3 = tf.nn.max_pool(cell3, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                                       padding="SAME", name=layer_name + "_maxpool")
            return cell3
        
        # Number of channels for filter input and output
        nfilt = (3, 32, 64, 128)
        
        with tf.variable_scope('cnn', initializer=tf.contrib.layers.xavier_initializer()) as scope:
            layer1 = create_conv_layer(self.input_feed, filter_size=5, prev_channels=nfilt[0], num_filters=nfilt[1],
                                       stride_len=1, pool=True, layer_name="conv1", is_trainable=self.train_cnn)
            
            layer2 = create_conv_layer(layer1, filter_size=3, prev_channels=nfilt[1], num_filters=nfilt[2],
                                       stride_len=1, pool=True, layer_name="conv2", is_trainable=self.train_cnn)
            
            layer3 = create_conv_layer(layer2, filter_size=3, prev_channels=nfilt[2], num_filters=nfilt[3],
                                       stride_len=1, pool=False, layer_name="conv3", is_trainable=self.train_cnn)
            
            # Global max pooling over the final conv cells
            gmp0 = tf.reduce_max(layer3, [1, 2], name="cnn_end")

        return gmp0
