"""
Configuration files for data importing, models and training/evaluation

Based on the im2txt implementation on GitHub from the Tensorflow Authors

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import tensorflow as tf

class Config(object):
    def __init__(self, mode, sw_specification):
        """Load the ShapeWorld JSON file and build config file"""
        # If filename then load file else keep dict
        if isinstance(sw_specification, str):
            with open(sw_specification, 'r') as fh:
                self.src_config = json.load(fh)
        else:
            self.src_config = sw_specification

        # Add objects in src_config as attributes [non recursive so inner layer dicts are still dicts]
        for k, v in self.src_config.items():
            setattr(self, k, v)

        # Mode
        assert mode in ["train", "eval", "test"], "Mode not recognised"
        self.mode = mode
        
        # Shapeworld shard file feats
        self.instances_per_shard = 10000
        self.num_shards = 10 if self.mode == "train" else 1
        self.num_epochs = 100 if self.mode == "train" else 1
        self.batch_size = 128
        self.num_steps_per_epoch = int((np.floor(self.instances_per_shard) * self.num_shards)/self.batch_size)
        self.num_total_steps = int(np.floor(self.num_epochs * self.num_steps_per_epoch))
        
        # Data config -----------------------------
        self.pixel_noise_stddev = 0.1
        self.noise_axis = 3
        
        # Model config -----------------------------
        self.train_cnn = False
        self.cnn_checkpoint = None
        self.initializer = tf.contrib.layers.xavier_initializer
        self.initializer_scale = 0.1
        self.embedding_size = 128
        self.img_embedding_size = self.embedding_size
        self.joint_embedding_size = self.img_embedding_size + self.embedding_size
        # self.num_lstm_units = self.joint_embedding_size
        self.num_lstm_units = 128  # same as image embedding size for one-shot
        self.output_classes = 56
        
        # Training config -----------------------------
        self.optimizer = "RMSProp"
        self.initial_learning_rate = 0.001        # todo: check this lr is right
        self.clip_gradients = 5.0
        self.train_cnn_learning_rate = 0.001     # todo: check this lr is right
        self.max_checkpoints_to_keep = 10
        self.save_every_n_steps = 100
        self.eval_interval_seconds = 5700