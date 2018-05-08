"""
SEQ2SEQ IMAGE CAPTIONING
Tom Sherborne 8/5/18
"""
import tensorflow as tf
from image_network import ShapeWorldEncoder
seq2seq = tf.contrib.seq2seq


class CaptioningModel(object):
    """
    Image Captioning on Shapeworld data with tf.contrib.seq2seq
    """
    
    def __init__(self, config):
        #  Keep input config
        self.config = config
        self.global_step = None                     # Global step Tensor.

        # Input feats from Shapeworld data loading
        self.images = None                          # float32 with shape [batch, 64, 64, 3]
        self.logits = None                          # float32 with shape [batch, 56]
        self.phase = None                           # bool with shape [1]
        self.input_seqs = None                      # int32 with shape [batch, config.max_input_len]
        self.target_seqs = None                     # int32 with shape [batch, config.max_input_len]
        self.input_seqs_len = None                  # int32 with shape [batch,]
        
        # Embeddings
        self.img_embedding = None  # float32 with shape [batch, config.embedding_size]
        self.seq_embeddings = None  # float32 with shape [batch, config.max_input_len, config.embedding_size]

        # CNN encoder
        self.cnn_encoder = None                     # Keep CNN as an attribute
        self.cnn_variables = []                     # Collection of variables to restore
        self.cnn_checkpoint = config.cnn_checkpoint # Checkpoint directory to load variables from
        
        # LSTM Decoder
        self.lstm = tf.contrib.rnn.BasicLSTMCell(num_units=config.num_lstm_units)
        self.initializer = config.initializer()
        
        # Loss terms
        self.batch_loss = None                      # A float32 scalar Tensor
        self.init_fn = None                         # Function to restore the CNN submodel from checkpoint.

        # todo: get eos and sos vocab ids
        

    def is_training(self):
        return self.config.mode == "train"
    
    def is_testing(self):
        return self.config.mode == "test"
    
    def build_model(self, batch, embedding_init):
        """Construct the TF computation graph for the IC model"""
        self.build_inputs(batch)
        
        # setup img and seq embeddings
        self.build_img_embeddings()
        self.build_seq_embeddings(embedding_init)
        # conditional on mode: setup decoder models
    
    def build_inputs(self, batch):
        """Setup input sequences for training"""
        assert 'world' in batch
        assert 'input_seqs' in batch
        assert 'target_seqs' in batch
        assert 'seqs_len' in batch

        # set class input objects to batch features
        self.images = batch['world']
        self.input_seqs = batch['input_seqs']
        self.target_seqs = batch['target_seqs']
        self.input_seqs_len = batch['seqs_len']
        

