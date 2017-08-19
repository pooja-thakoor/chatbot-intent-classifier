from pre_processor import PreProcessor
import numpy as np

class FeatureExtracter:

    def perform_feature_extraction(self,data):
        print('Extracting features...')
        self.feature_list = []
        pre_processor = PreProcessor()
        for intent, docs in data.items():
            for query in docs:
                features_query = pre_processor.process(query)
                self.feature_list = self.feature_list + [features_query]
        return self.feature_list
    # end

    def make_feature_vector(self,words, model, num_features):
        feature_vector = np.zeros((num_features,), dtype="float32")
        num_words = 0
        index2word_set = set(model.wv.index2word)
        for word in words:
            if word in index2word_set:
                num_words = num_words + 1
                feature_vector = np.add(feature_vector, model[word])

        feature_vector = np.divide(feature_vector, num_words)
        return feature_vector
    #end

    def make_average_feature_vector(self,queries, model, num_features):
        counter = 0
        query_feature_vec = np.zeros((len(queries), num_features), dtype="float32")
        for query in queries:
            query_feature_vec[counter] = self.make_feature_vector(query, model, num_features)
            counter = counter + 1
        return query_feature_vec
    #end
