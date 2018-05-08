import numpy as np
import pandas as pd
import csv
import pickle
from collections import OrderedDict

MASTER_VOCAB = ('.', 'a', 'above', 'all', 'an', 'and', 'are', 'as', 'at', 'behind', 'below', 'bigger', 'blue', 'but',
              'circle', 'circles', 'closer', 'cross', 'crosses', 'cyan', 'darker', 'eight', 'either', 'ellipse',
              'ellipses', 'exactly', 'farther', 'few', 'five', 'four', 'from', 'front', 'gray', 'green', 'half',
              'in', 'is', 'least', 'left', 'less', 'lighter', 'magenta', 'many', 'more', 'most', 'no', 'none', 'not',
              'of', 'one', 'or', 'pentagon', 'pentagons', 'quarter', 'quarters', 'rectangle', 'rectangles', 'red',
              'right', 'semicircle', 'semicircles', 'seven', 'shape', 'shapes', 'six', 'smaller', 'square', 'squares',
              'than', 'the', 'there', 'third', 'thirds', 'three', 'to', 'triangle', 'triangles', 'twice', 'two',
              'yellow', 'zero')


class GloveLoader(object):
    """Load Glove Embeddings of dims-size for vocab words"""
    def __init__(self, vocab, pkl_file, dims, load_new=False, glove_txt_file=None, oov_init=np.random.rand):
        self.vocab = vocab          # Vocab list
        self.pkl_file = pkl_file    # Location of PKL
        self.dims = dims            # Dimensionality of loaded embeddings
        self.oov_init = oov_init    # How to initialise OOV embeddings

        if load_new:
            # Generate new embeddings from file
            self.__parse_new_embeddings(glove_txt_file)
        
        # Load embeddings from Pickle
        self.embedding_dict = self.__load_glove()
        
        # make embedding matrix of size vocab x dim
        self.embedding_mat = np.array(list(self.embedding_dict.values()), dtype=np.float32)

    def get_embeddings_matrix(self):
        """Return embeddings matrix"""
        print("Loading Glove embedding matrix of size:", np.shape(self.embedding_mat))
        return self.embedding_mat
        
    def __load_glove(self):
        """Load the embeddings dictionary from file"""
        
        # Load the GloVe dictionary pickle
        try:
            glove_dict = pickle.load(open(self.pkl_file, 'rb'))
        except FileNotFoundError as e:
            print("File %s not found" % self.pkl_file)

        #   Get all desired embeddings from the Glove dictionary and randomly initalise the others
        #   The order of self.vocab is implicitly the embedding vocabulary index value
        embeds = OrderedDict()
        for vocab_word in self.vocab:
            if vocab_word in glove_dict:
                print("GloVe for word: %s found" % vocab_word)
                embeds[vocab_word] = glove_dict[vocab_word]
            else:
                print("GloVe for word: %s not found" % vocab_word)
                embeds[vocab_word] = self.oov_init(self.dims)

            embeds[vocab_word] = glove_dict[vocab_word]
        return embeds
        
    def __parse_new_embeddings(self, src_file):
        """
        Get new embeddings from src according to the MASTER vocab
        Thanks + credit to @jayelm for this snippet
        """
        assert src_file is not None, "Must provide a GloVe embeddings source"
        new_glove_src = src_file.replace('.txt', '.pkl')
        if self.pkl_file is not None:
            print("%s will be ignored in favour of %s" % (self.pkl_file, new_glove_src))
            self.pkl_file = new_glove_src
        
        embeddings = pd.read_csv(src_file,
                                 header=None,
                                 index_col=False,
                                 sep=' ',
                                 quoting=csv.QUOTE_NONE)

        words = np.array(embeddings[0])
        vecs = embeddings.iloc[:, 1:].as_matrix().astype(np.float32)
        del embeddings
        
        # Make dicts
        wv_map = dict(zip(words, vecs))
        glove_words = set(wv_map.keys())
        output_map = {}
        
        # Extract out relevant words in vocabulary
        for v in MASTER_VOCAB:
            if v in glove_words:
                output_map[v] = wv_map[v]
            else:
                print("%s not found in GloVe" % v)
        
        with open(new_glove_src, 'wb') as fout:
            pickle.dump(output_map, fout)
