from gensim.parsing import preprocessing

class PreProcessor:

    def process(self, query, stop_word=True):
        if not stop_word:
            query = preprocessing.preprocess_string(query,filters=[lambda x: x.lower(),
                                                               preprocessing.strip_tags,
                                                               preprocessing.strip_punctuation,
                                                               preprocessing.strip_multiple_whitespaces,
                                                               preprocessing.strip_numeric,
                                                               lambda x:x,
                                                               preprocessing.strip_short,
                                                               preprocessing.stem_text])
        else:
            query = preprocessing.preprocess_string(query)
        return query
    #end