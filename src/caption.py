"""
SEQ2SEQ IMAGE CAPTIONING
Borrows heavily from the im2txt model in tf.models
Tom Sherborne 17/5/18
"""

SHAPES = ['circle', 'cross', 'ellipse', 'pentagon', 'rectangle', 'semicircle', 'square', 'triangle']   # Specific shapes
COLORS = ['blue', 'cyan', 'gray', 'green', 'magenta', 'red', 'yellow']  # Color words

class Caption(object):
    """
    A wrapper class for Caption features
    """
    def __init__(self, caption_idxs, vocab, rev_vocab):
        """
        Initialise caption object
        :param caption_idxs: list like object for caption
        :param vocab: vocabulary mapping for idxs to words
        """
        self.caption_idxs = caption_idxs
        self.vocab = vocab              # {"shape": 2}
        self.rev_vocab = rev_vocab      # {3: "red"}
        self.shape_word_idx = self.vocab['shape']
        
        #   Get shape from caption, ['shape'] is allowed
        self.shape = [rev_vocab[s] for s in caption_idxs if rev_vocab[s] in SHAPES or rev_vocab[s] == "shape"]
        
        #   Get color from caption
        self.color = [rev_vocab[c] for c in caption_idxs if rev_vocab[c] in COLORS]
        
    def __eq__(self, other):
        """
        Oneshape: Captions are semantically equivalent if the shapes and colors are the same
        """
        if self.shape == (other.shape or self.shape_word_idx) and self.color == other.color:
            return True
        else:
            return False

    def __repr__(self):
        return "Caption:(idxs:%s):(str:%s):(shape:%s):(color:%s)" % \
               (self.caption_idxs,
                " ".join([self.rev_vocab[r] for r in self.caption_idxs]),
                self.shape,
                self.color)