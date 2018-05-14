"""
SEQ2SEQ IMAGE CAPTIONING
Borrows heavily from the im2txt model in tf.models
Tom Sherborne 8/5/18

Batch parser base class
"""

import tensorflow as tf
tfl = tf.contrib.lookup

class ParserBase(object):
    def __init__(self, src_vocab,  tgt_vocab, sos_token="<S>", eos_token="</S>", max_seq_len=12):
        self.src_vocab = src_vocab  # Src Shapeworld vocab
        self.tgt_vocab = tgt_vocab  # Vocab : {"word": idx}
        self.vocab_transfer_dict = {self.src_vocab[k]: self.tgt_vocab[k]
                                    for k in self.src_vocab.keys()
                                    if k in self.tgt_vocab.keys()}
        
        self.sos_token_id = self.tgt_vocab[sos_token]
        self.eos_token_id = self.tgt_vocab[eos_token]
        self.max_seq_len = max_seq_len
        
        # Reverse vocab : {idx: "word"}
        self.rev_vocab = {v: i for i, v in self.tgt_vocab.items()}

        # Key mapping for original vocabulary
        self.vocab_map = self.generate_map()

    def generate_map(self):
        """Generate Tensorflow Hashmaps for index remapping"""
        keys = tf.cast(list(self.vocab_transfer_dict.keys()), dtype=tf.int32)
        values = tf.cast(list(self.vocab_transfer_dict.values()), dtype=tf.int32)
    
        return tfl.HashTable(tfl.KeyValueTensorInitializer(keys, values), -1)

    def get_vocab(self):
        """Return vocabulary and reverse vocabulary objects"""
        return self.tgt_vocab, self.rev_vocab

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
        print("get_batch_parser needs implementing!")
        return