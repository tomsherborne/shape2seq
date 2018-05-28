"""
SEQ2SEQ IMAGE CAPTIONING
Borrows heavily from the im2txt model in tf.models
Tom Sherborne 8/5/18
"""
import copy
from collections import namedtuple
import tensorflow as tf
tfl = tf.contrib.lookup

from src.parser_base import ParserBase

# Src Vocab from Agreement-Simple dataset
SIMPLE_SRC_VOCAB = ['', '.', 'a', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'gray', 'green', 'is', 'magenta',
            'pentagon', 'rectangle', 'red', 'semicircle', 'shape', 'square', 'there', 'triangle', 'yellow', '[UNKNOWN]']

# Shape and colours
SHAPES = ['circle', 'cross', 'ellipse', 'pentagon', 'rectangle', 'semicircle', 'square', 'triangle']   # Specific shapes
SHAPES_HYPERNYMS = ['shape']      # Abstract words for shapes
COLORS = ['blue', 'cyan', 'gray', 'green', 'magenta', 'red', 'yellow']  # Color words
STOPS = ['a', 'an', 'there', 'is', "."]      # Stop words
AUX_VOCAB = ['[UNKNOWN]', "<S>", "</S>"]    # Aux words to useful vocabulary
SPATIAL_AUX_VOCAB = ["above", "below", "left", "right", "of", "to"]


AGREEMENT_ONESHAPE_VOCAB = AUX_VOCAB + SHAPES + SHAPES_HYPERNYMS + COLORS + STOPS + [""]
AGREEMENT_SPATIAL_VOCAB = AGREEMENT_ONESHAPE_VOCAB + SPATIAL_AUX_VOCAB+ [""]


AUX_VOCAB = ["", '[UNKNOWN]', "<S>", "</S>"]

SHAPE_COLOR_VOCAB = AUX_VOCAB + ['blue', 'circle', 'cross', 'cyan', 'ellipse', 'gray', 'green', 'magenta', 'pentagon',
                                 'rectangle', 'red', 'semicircle', 'square', 'triangle', 'yellow']
SHAPE_VOCAB = AUX_VOCAB + ['circle', 'cross', 'ellipse', 'pentagon', 'rectangle', 'semicircle', 'square', 'triangle']
COLOR_VOCAB = AUX_VOCAB + ['blue', 'cyan', 'gray', 'green', 'magenta', 'red', 'yellow']
STANDARD_VOCAB = AUX_VOCAB + ['.', 'a', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'gray', 'green', 'is', 'magenta',
                        'pentagon', 'rectangle', 'red', 'semicircle', 'shape', 'square', 'there', 'triangle', 'yellow']

TGT_VOCAB_ = {"shape": SHAPE_VOCAB, "color": COLOR_VOCAB, "shape_color": SHAPE_COLOR_VOCAB, "standard": STANDARD_VOCAB}
SIMPLE_TGT_VOCAB_ = {"shape": SHAPE_VOCAB, "color": COLOR_VOCAB, "shape_color": SHAPE_COLOR_VOCAB, "standard": STANDARD_VOCAB}


class SimpleBatchParser(ParserBase):
    """
    Initialise a batch parsing function to return the correct sequences and vocabulary for training a specific model
    Modes are:
        standard:  "there is a red square ." -> "<S> there is a red square </S>"
        shape_color: "there is a red square ." -> "<S> red square </S>"
        shape: "there is a red square ." -> "<S> square </S>"
        color: "there is a red square ." -> "<S> red </S>"
    
    The get_batch_parser() fn returns function to perform this functionality
    """
    def __init__(self, src_vocab, batch_type, sos_token="<S>", eos_token="</S>", max_seq_len=5):
        """
        Initialise the batch parser based upon the desired return object
        """
        tgt_vocab = {v: i for i, v in enumerate(SIMPLE_TGT_VOCAB_[batch_type])}
        super().__init__(src_vocab=src_vocab,
                         tgt_vocab=tgt_vocab,
                         sos_token=sos_token,
                         eos_token=eos_token,
                         max_seq_len=max_seq_len)
        
        assert batch_type in ['shape', 'color', 'shape_color', 'standard'],  "Must specify a valid batch parser type"
        self.batch_type = batch_type    # Controls which kind of batch object is returned

        # Elements to strip from string
        self.token_filter = AUX_VOCAB
    
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


class FullSequenceBatchParser(ParserBase):
    """
    Initialise a batch parsing function to return the correct sequences and vocabulary for training a specific model.
    This model differs from the SimpleBatchParser as there is 1 mode and the caption length is variable.
    Modes are:
        standard:  "there is a red square ." -> "<S> there is a red square . </S>"
                   "a blue shape is a cross ." -> "<S> a blue shape is a cross . </S>"

    The get_batch_parser() fn returns function to perform this functionality
    """

    def __init__(self, src_vocab, sos_token="<S>", eos_token="</S>", padding_token="", max_seq_len=16):
        """
        Initialise the batch parser
        """
    
        max_vocab_idx = len(src_vocab)
        tgt_vocab = copy.deepcopy(src_vocab)
        tgt_vocab[sos_token] = max_vocab_idx
        tgt_vocab[eos_token] = max_vocab_idx + 1
    
        super().__init__(src_vocab=src_vocab,
                         tgt_vocab=tgt_vocab,
                         sos_token=sos_token,
                         eos_token=eos_token,
                         max_seq_len=max_seq_len)
    
        self.pad_token_id = self.tgt_vocab[padding_token]
    

    def crop_standard(self, batch):
        row = batch['caption']
        caption_ = tf.concat([[self.sos_token_id],
                              tf.map_fn(lambda elem: self.vocab_map.lookup(elem), row[:batch['caption_length']]),
                              [self.eos_token_id],
                              tf.map_fn(lambda elem: self.pad_token_id, row[batch['caption_length']:])],
                             axis=0)
        batch['caption'] = caption_
        return batch

    def get_batch_parser(self):
        """
        Transform a ShapeWorld "agreement-oneshape" batch to a model appropriate format
        i.e. "there is a red square" to "red square" or "red", or "square" based upon the vocab
        """

        crop_fn = self.crop_standard
        split_fn = self.split_seqs
    
        def batch_parser(batch):
            # Map the fn to correct vocab
            batch = tf.map_fn(crop_fn, batch)
        
            # Split sequences
            batch['complete_seqs'], batch['input_seqs'], batch['target_seqs'], batch['input_mask'], batch['seqs_len'] = \
                tf.map_fn(split_fn, batch['caption'], dtype=(tf.int32, tf.int32, tf.int32, tf.int32, tf.int32))
            return batch
    
        return batch_parser
    
    def score_cap_against_world_oneshape(self, world_model, inf_caption_idxs):
        """
        Score a caption against the world model for the same image.
        
        Examples:
            ref: blue square        | inf: "a shape is blue"                -> underspecify correct
            ref: red cross          | inf: "there is a cross"               -> underspecify correct
            ref: green cross        | inf: "a shape is green"               -> underspecify correct
            ref: green rectangle    | inf: "a green shape is a rectangle"   -> specific correct
            ref: blue circle        | inf: "there is a blue circle"         -> specific correct
            ref: yellow square      | inf: "a blue square is a shape"       -> shape correct only
            ref: yellow square      | inf: "there is a yellow shape"        -> color correct only
            ref: green circle       | inf: "there is a red square"          -> incorrect
            ref: grey semicircle    | inf: "a shape is blue"                -> incorrect

        Return a CaptionScore tuple
        """

        CaptionScore = namedtuple("CaptionScore", ["world_model", "inf_cap",            # world model dict, output cap
                                                   "ref_shape", "ref_color",
                                                   "shape_correct", "color_correct",    # specific shape and colors true
                                                   "specify_true",                      # shapes and colors correct
                                                   "no_color_specify_shape_true",       # "there is a square"
                                                   "specify_color_hypernym_shape_true", # "there is a red shape"
                                                   "no_color_hypernym_shape_true",      # "there is a shape"
                                                   "false"])                            # incorrect statements

        ref_color = world_model['entities'][0]['color']['name']
        ref_shape = world_model['entities'][0]['shape']['name']
        
        print("REF | SHAPE: %s | COLOUR %s" % (ref_shape, ref_color))
        
        inf_shapes = set([self.rev_vocab[w] for w in inf_caption_idxs if self.rev_vocab[w] in SHAPES])
        inf_colors = set([self.rev_vocab[w] for w in inf_caption_idxs if self.rev_vocab[w] in COLORS])
        inf_hyper = set([self.rev_vocab[w] for w in inf_caption_idxs if self.rev_vocab[w] in SHAPES_HYPERNYMS])

        inf_cap_str = " ".join([self.rev_vocab[w] for w in inf_caption_idxs if w!=self.pad_token_id])
        
        if ref_color in inf_colors and ref_shape in inf_shapes:
            print("SPECIFIC CORRECT")
            return CaptionScore(world_model, inf_cap_str, ref_shape, ref_color, 1, 1, 1, 0, 0, 0, 0)
        elif ref_shape in inf_shapes and not inf_colors:
            print("NO COLOR SPECIFY SHAPE")
            return CaptionScore(world_model, inf_cap_str, ref_shape, ref_color, 1, 0, 0, 1, 0, 0, 0)
        elif ref_color in inf_colors and inf_hyper and not inf_shapes:
            print("SPECIFY COLOR HYPERNYM SHAPE")
            return CaptionScore(world_model, inf_cap_str, ref_shape, ref_color, 0, 1, 0, 0, 1, 0, 0)
        elif inf_hyper and not inf_colors and not inf_shapes:
            print("NO COLOR HYPERNYM SHAPE")
            return CaptionScore(world_model, inf_cap_str, ref_shape, ref_color, 0, 0, 0, 0, 0, 1, 0)
        else:
            print("INCORRECT")
            return CaptionScore(world_model, inf_cap_str, ref_shape, ref_color, 0, 0, 0, 0, 0, 0, 1)

    def score_cap_against_word_spatial(self, world_model, ref_caption_idxs, inf_caption_idxs):
        print("score_cap_against_world needs implementing")
        return