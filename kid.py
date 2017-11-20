import gensim as gs
import codecs

model=gs.models.KeyedVectors.load_word2vec_format("wbtrain.bin",binary=True)
print('Read word2vec model successfully.')