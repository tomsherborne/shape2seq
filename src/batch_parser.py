"""
SEQ2SEQ IMAGE CAPTIONING
Borrows heavily from the im2txt model in tf.models
Tom Sherborne 8/5/18
"""

import tensorflow as tf
seq2seq = tf.contrib.seq2seq
tfl = tf.contrib.lookup

# SRC VOCAB FROM SHAPEWORLD API
SIMPLE_SRC_VOCAB = ['', '.', 'a', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'gray', 'green', 'is', 'magenta',
            'pentagon', 'rectangle', 'red', 'semicircle', 'shape', 'square', 'there', 'triangle', 'yellow', '[UNKNOWN]']

# Aux words to useful vocabulary
AUX_VOCAB = ["", '[UNKNOWN]', "<S>", "</S>"]

SHAPE_COLOR_VOCAB = AUX_VOCAB + ['blue', 'circle', 'cross', 'cyan', 'ellipse', 'gray', 'green', 'magenta', 'pentagon',
                                 'rectangle', 'red', 'semicircle', 'square', 'triangle', 'yellow']
SHAPE_VOCAB = AUX_VOCAB + ['circle', 'cross', 'ellipse', 'pentagon', 'rectangle', 'semicircle', 'square', 'triangle']
COLOR_VOCAB = AUX_VOCAB + ['blue', 'cyan', 'gray', 'green', 'magenta', 'red', 'yellow']
STANDARD_VOCAB = AUX_VOCAB + ['.', 'a', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'gray', 'green', 'is', 'magenta',
                        'pentagon', 'rectangle', 'red', 'semicircle', 'shape', 'square', 'there', 'triangle', 'yellow']

TGT_VOCAB_ = {"shape": SHAPE_VOCAB, "color": COLOR_VOCAB, "shape_color": SHAPE_COLOR_VOCAB, "standard": STANDARD_VOCAB}


class SimpleBatchParser(object):
    """
    Initialise a batch parsing function to return the correct sequences and vocabulary for training a specific model
    Modes are:
        standard:  "there is a red square ." -> "<S> there is a red square </S>"
        shape_color: "there is a red square ." -> "<S> red square </S>"
        shape: "there is a red square ." -> "<S> square </S>"
        color: "there is a red square ." -> "<S> red </S>"
    
    The get_batch_parser() fn returns function to perform this functionality
    """
    def __init__(self, batch_type, sos_token="<S>", eos_token="</S>", max_seq_len=12):
        """
        Initialise the batch parser based upon the desired return object
        """
        
        assert batch_type in ['shape', 'color', 'shape_color', 'standard'],  "Must specify a valid batch parser type"
        self.batch_type = batch_type    # Controls which kind of batch object is returned
        
        self.src_vocab = {v: i for i, v in enumerate(SIMPLE_SRC_VOCAB)}
        self.tgt_vocab = {v: i for i, v in enumerate(TGT_VOCAB_[batch_type])}
        self.rev_vocab = {v: i for i, v in self.tgt_vocab.items()}
        self.vocab_transfer_dict = {self.src_vocab[k]: self.tgt_vocab[k]
                                    for k in self.src_vocab.keys()
                                    if k in self.tgt_vocab.keys()}
        # Key mapping for original vocabulary
        self.vocab_map = self.generate_map()
        
        self.sos_token_id = self.tgt_vocab[sos_token]
        self.eos_token_id = self.tgt_vocab[eos_token]
        self.token_filter = [self.sos_token_id, self.eos_token_id]
        self.max_seq_len = max_seq_len
        
    def generate_map(self):
        """Generate Tensorflow Hashmaps for index remapping"""
        keys = tf.cast(list(self.vocab_transfer_dict.keys()), dtype=tf.int32)
        values = tf.cast(list(self.vocab_transfer_dict.values()), dtype=tf.int32)

        return tfl.HashTable(tfl.KeyValueTensorInitializer(keys, values), -1)
        
    def get_vocab(self):
        """Return vocabulary and reverse vocabulary objects"""
        return self.tgt_vocab, self.rev_vocab
    
    def crop_color(self, row):
        return tf.concat([[self.sos_token_id],
                          [self.vocab_map.lookup(row[3])],
                          [self.eos_token_id]],
                         axis=0)
   
    def crop_shape(self, row):
        return tf.concat([[self.sos_token_id],
                          [self.vocab_map.lookup(row[4])],
                          [self.eos_token_id]],
                         axis=0)
    
    def crop_shape_color(self, row):
        return tf.concat([[self.sos_token_id],
                          [self.vocab_map.lookup(row[3])],
                          [self.vocab_map.lookup(row[4])],
                          [self.eos_token_id]],
                         axis=0)
    
    def crop_standard(self, row):
        return tf.concat([[self.sos_token_id],
                          tf.map_fn(lambda elem: self.vocab_map.lookup(elem), row[:-1]),
                          [self.eos_token_id]],
                         axis=0)

    def split_seqs(self, row):
        # Cast row of vocab indices
        row = tf.cast(row, dtype=tf.int32)
        # Caption length
        caption_length = tf.shape(row)[0]
        # Input sequence length
        input_length = tf.cast(tf.expand_dims(tf.subtract(caption_length, 1), 0), dtype=tf.int32)
        # Input sequence
        input_seq = tf.slice(row, begin=[0], size=input_length)
        # Target sequence
        target_seq = tf.slice(row, begin=[1], size=input_length)
        # Mask for input
        input_mask = tf.ones(input_length, dtype=tf.int32)
        
        return row, input_seq, target_seq, input_mask, input_length
    
    def get_batch_parser(self):
        """
        Transform a ShapeWorld "simple" batch to a model appropriate format
        i.e. "there is a red square" to "red square" or "red", or "square" based upon the vocab
        """
        if self.batch_type == "shape":
            crop_fn = self.crop_shape
            
        elif self.batch_type == "color":
            crop_fn = self.crop_color
            
        elif self.batch_type == "shape_color":
            crop_fn = self.crop_shape_color
        else:
            crop_fn = self.crop_standard

        split_fn = self.split_seqs
        
        def batch_parser(batch):
            # Map the fn to correct vocab
            captions_ = tf.map_fn(crop_fn, batch['caption'], dtype=tf.int32)
            
            # Split sequences
            batch['complete_seqs'],batch['input_seqs'],batch['target_seqs'],batch['input_mask'],batch['seqs_len'] = \
                tf.map_fn(split_fn, captions_, dtype=(tf.int32, tf.int32, tf.int32, tf.int32, tf.int32))
            return batch

        return batch_parser
