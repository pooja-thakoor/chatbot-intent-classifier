from gensim.models import word2vec


class W2CFeatureGenerator:

    def __init__(self):
        # Set values for various parameters
        self.num_features = 5 # Word vector dimensionality
        self.min_word_count = 1  # Minimum word count
        self.num_workers = 4  # Number of threads to run in parallel
        self.context = 10  # Context window size
        self.down_sampling = 1e-3  # Downsample setting for frequent words

    def generate_features(self,sentences):
        # Initialize and train the model (this will take some time)
        print("Training model...")
        model = word2vec.Word2Vec(sentences, workers=self.num_workers, \
                                  size=self.num_features, min_count=self.min_word_count, \
                                  window=self.context, sample=self.down_sampling)

        # If you don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        model.init_sims(replace=True)

        # It can be helpful to create a meaningful model name and
        # save the model for later use. You can load it later using Word2Vec.load()
        model_name = "classifier"
        model.save('model/'+model_name)
        print("Keys after training - " + str(list(model.wv.vocab.keys())))
        print("Shape - " + str(model.wv.syn0.shape))
        print(model.wv.similarity("monei","payment"))
        return model