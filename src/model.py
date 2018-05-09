"""
SEQ2SEQ IMAGE CAPTIONING
Borrows heavily from the im2txt model in tf.models
Tom Sherborne 8/5/18
"""

import tensorflow as tf
import numpy as np
from src.image_network import ShapeWorldEncoder
seq2seq = tf.contrib.seq2seq

def parse_batch(batch):
    """
    Transform a generic ShapeWorld agreement batch to a model appropriate format
    PREVIOUSLY IN TRAINING SCRIPTS
    """
    # todo use tf.sequence mask with caption len for mask (https://www.tensorflow.org/api_docs/python/tf/sequence_mask)
    pass


class CaptioningModel(object):
    """
    Image Captioning on Shapeworld data with tf.contrib.seq2seq
    Model is based on LRCN_1u model from (Donahue et al. 2016)[arxiv.org/abs/1411.4389]
    """
    
    def __init__(self, config):
        assert config.mode in ['train', 'test', 'validation']   # validation currently unused
        
        #  Keep input config
        self.config = config
        self.global_step = None                     # Global step Tensor.

        # Input feats from Shapeworld data loading
        self.images = None                          # float32 with shape [batch, 64, 64, 3]
        self.logits = None                          # float32 with shape [batch, 56]
        self.phase = None                           # bool with shape [1]
        self.input_seqs = None                      # int32 with shape [batch, config.max_input_len]
        self.input_mask = None                      # int32 with shape [batch, config.max_input_len]
        self.target_seqs = None                     # int32 with shape [batch, config.max_input_len]
        self.input_seqs_len = None                  # int32 with shape [batch,]
        
        # Embeddings
        self.img_embedding = None  # float32 with shape [batch, config.embedding_size]
        self.seq_embeddings = None  # float32 with shape [batch, config.max_input_len, config.embedding_size]
        self.embedding_map = None   # float32 with shape (self.vocab_size, self.config.embedding_size)
        
        # CNN encoder [optionally trainable]
        self.cnn_encoder = None                     # Keep CNN as an attribute
        self.cnn_variables = []                     # Collection of variables to restore
        self.cnn_checkpoint = config.cnn_checkpoint # Checkpoint directory to load variables from
        
        # LSTM Decoder [trainable decoder]
        self.lstm = tf.contrib.rnn.BasicLSTMCell(num_units=config.num_lstm_units)
        
        # Projection layer
        self.projection_layer = tf.layers.Dense(units=self.vocab_size,
                                                activation=None,
                                                kernel_initializer=self.initializer,
                                                use_bias=False)
        
        self.initializer = config.initializer()
        self.embedding_initializer = config.embedding_initializer(minval=-self.config.initializer_scale,
                                                                  maxval=self.config.initializer_scale)
        
        # Loss terms
        self.batch_loss = None                      # A float32 scalar Tensor
        self.init_fn = None                         # Function to restore the CNN submodel from checkpoint.

        # seq2seq specific feats todo: fill these in from the vocab
        self.vocab_size = None
        self.target_start_tok = None
        self.target_end_tok = None
        self.max_decoding_iters = None

    def is_training(self):
        return self.config.mode == "train"
    
    def is_testing(self):
        return self.config.mode == "test"
    
    def build_model(self, batch, embedding_init=None):
        """Construct the TF computation graph for the IC model"""
        assert self.target_start_tok is not None # todo: fix this catch to stop me running bad models
        
        self.__build_inputs(batch)
        
        # setup img and seq embeddings
        self.__build_img_embeddings()
        self.__build_seq_embeddings(embedding_init)
        
        # Conditional on mode: setup decoder models
        if self.is_testing():
            self.__build_inference_graph()
        else:
            self.__build_training_graph()
        self.__build_cnn_initializer()
        self.__build_global_step()
    
    def __build_inputs(self, batch):
        """Setup input sequences for training"""
        batch = parse_batch(batch)
        assert 'world' in batch
        assert 'input_seqs' in batch
        assert 'target_seqs' in batch
        assert 'seqs_len' in batch

        # set class input objects to batch features
        self.images = batch['world']
        self.input_seqs = batch['input_seqs']
        self.target_seqs = batch['target_seqs']
        self.input_seqs_len = batch['seqs_len']
        
        # Batch normalisation phase is high for training, low for testing
        self.phase = tf.placeholder_with_default(input=tf.constant(True, dtype=tf.bool), shape=None, name='phase')

    def __build_img_embeddings(self):
        """Build the CNN computation graph for encoding an image to embedding"""
        self.cnn_encoder = ShapeWorldEncoder(train_cnn=self.config.train_cnn)
        
        # Build model and get output [batch_size, 128]
        self.img_embedding = self.cnn_encoder.build_model(self.images, self.phase)
        
        # Get the Collection of CNN variables for restoration
        self.cnn_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="cnn")
        
    def __build_seq_embeddings(self, embedding_init):
        """Build the 1-hot to embedding encoding with optional initialization"""
        with tf.variable_scope("seq_embeddings"), tf.device("/gpu:0"):
            embed_shape = (self.vocab_size, self.config.embedding_size)
            
            # Initialise with some pretrained embeddings
            if self.is_training() and embedding_init:
                assert np.shape(embedding_init) == embed_shape, \
                    "Embedding initialisation is shape %s, expecting %s" % (np.shape(embedding_init), embed_shape)
                
                embedding_map = tf.get_variable(name="seq_map",
                                                shape=embed_shape,
                                                initializer=tf.constant_initializer(embedding_init),
                                                trainable=self.config.train_embeddings)
                del embedding_init
            else:
                embedding_map = tf.get_variable(name="seq_map",
                                                shape=embed_shape,
                                                initializer=self.embedding_initializer,
                                                trainable=True)
            
            # Lookup on GPU
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)
        
        self.embedding_map = embedding_map      # for recurrent decoding
        self.seq_embeddings = seq_embeddings    # for input sequences
        
            
    def __build_inference_graph(self):
        """
        Build the inference graph for evaluating captioning
        
        Order of proceedings:
        1. Get LSTM and Projection Layer feats as the trainable objects
        2. Declare decoding Greedy/Sample EmbeddingHelper, initial state and BasicDecoder
        3. Concatenate image and word embeds together for inputs
        4. Run decoding with a maximum number of iterations
        5. Return captions
        """
        assert self.config.batch_size == 1, "Batch size must be 1 for inference!"
        
        with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
            # Image embedding batch size sets the LSTM initial state size
            lstm_batch_size = tf.shape(self.img_embedding)[0]
    
            zero_state = self.lstm.zero_state(batch_size=lstm_batch_size,
                                              dtype=tf.float32
                                              )
    
            #   Initial input tokens are concatenated [<S>;<img_embedding] size: [batch, 1, [<S>;<img_embedding]]
            #   This is accomplished by augmenting the embedding matrix with the image embedding.
            #   This limits the batch size to be 1 as the img_embed concat op has to renew for each new image
            
            # # Mask image sequences
            # masked_img = tf.multiply(tf.expand_dims(self.img_embedding, axis=-2),
            #                          tf.expand_dims(self.input_mask, axis=-1))
            #
            # # Join images and word embeddings together and transpose for time major
            # joint_embedding = tf.transpose(tf.concat([self.seq_embeddings, masked_img], axis=-1),
            #                                perm=[1, 0, 2])
            #
            # tiled_end_tok = tf.fill((self.config.batch_size), self.target_end_tok)
            #
            # Greedy embedding helper for the output
            decoding_helper = seq2seq.GreedyEmbeddingHelper(embedding=self.embedding_map,
                                                            start_tokens=None,
                                                            end_token=self.target_end_tok)
    
            training_decoder = seq2seq.BasicDecoder(cell=self.lstm,
                                                    helper=decoding_helper,
                                                    initial_state=zero_state,
                                                    output_layer=self.projection_layer)
    
            lstm_outputs, _, output_seq_lens = seq2seq.dynamic_decode(training_decoder)
            import pdb;
            pdb.set_trace()

        # outputs is a vector of pre-softmax distributions over the vocab size [batch_size,max_seq_len, vocab_size]
        lstm_outputs = lstm_outputs.rnn_output
        assert tf.shape(lstm_outputs)[0] == self.config.batch_size

    # todo: can both build_X_graph fns be combined?
    def __build_training_graph(self):
        """
        Build the training graph for learning
        
        Model is based on LRCN_1u model from (Donahue et al. 2016)[arxiv.org/abs/1411.4389]
        Order of proceedings:
        1. Get LSTM and Projection Layer feats as the trainable objects
        2. Declare decoding TrainingHelper, initial state and BasicDecoder
        3. Concatenate image and word embeds together for inputs
        4. Run decoding
        5. Compute Loss op
        """
        
        with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
            # Image embedding batch size sets the LSTM initial state size
            lstm_batch_size = tf.shape(self.input_seqs)[0]
            
            zero_state = self.lstm.zero_state(batch_size=lstm_batch_size,
                                              dtype=tf.float32
                                              )
            
            # Mask image sequences
            masked_img = tf.multiply(tf.expand_dims(self.img_embedding, axis=-2),
                                     tf.expand_dims(self.input_mask, axis=-1))
            
            # Join images and word embeddings together and transpose for time major
            joint_embedding = tf.transpose(tf.concat([self.seq_embeddings, masked_img], axis=-1),
                                           perm=[1, 0, 2])



            # Standard fully Teacher forced training helper
            decoding_helper = seq2seq.TrainingHelper(inputs=joint_embedding,
                                                     sequence_length=self.input_seqs_len,
                                                     time_major=True,
                                                     name="dec_helper"
                                                     )

            training_decoder = seq2seq.BasicDecoder(cell=self.lstm,
                                                    helper=decoding_helper,
                                                    initial_state=zero_state,
                                                    output_layer=self.projection_layer)

            lstm_outputs, _, output_seq_lens = seq2seq.dynamic_decode(training_decoder)
        import pdb;pdb.set_trace()

        # outputs is a vector of pre-softmax distributions over the vocab size [batch_size,max_seq_len, vocab_size]
        lstm_outputs = lstm_outputs.rnn_output
        assert tf.shape(lstm_outputs)[0] == self.config.batch_size
        
        import pdb;pdb.set_trace()
        
        # Weighted softmax cross entropy todo: look at cosine-dist minimisation loss
        loss_ = seq2seq.sequence_loss(logits=lstm_outputs,
                                      targets=self.target_seqs,
                                      weights=self.input_mask,
                                      average_across_timesteps=True,
                                      average_across_batch=True)

        tf.losses.add_loss(loss_)
        self.batch_loss = tf.losses.get_total_loss()
        
        # Summaries for Tensorboard
        tf.summary.scalar("losses/batch_loss", self.batch_loss)
        for var in tf.trainable_variables():
            tf.summary.histogram("parameters/" + var.op.name, var)
            
    def __build_cnn_initializer(self):
        """Restore the CNN collection from a checkpoint prior to training the decoder"""
        if self.is_training():
            # Restore CNN collection
            saver = tf.train.Saver(self.cnn_variables)
            ckpt = tf.train.latest_checkpoint(self.cnn_checkpoint)

            def restore_fn(sess):
                tf.logging.info("Restoring OD CNN variables from checkpoint file %s",
                                ckpt)
                saver.restore(sess, ckpt)

            self.init_fn = restore_fn
    
    def __build_global_step(self):
        """Global step for training tracking is a model attribute"""
        self.global_step = tf.Variable(initial_value=0,
                                       name="global_step",
                                       trainable=False,
                                       collections=(tf.GraphKeys.GLOBAL_STEP,
                                                    tf.GraphKeys.GLOBAL_VARIABLES))
