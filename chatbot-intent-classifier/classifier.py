import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from data_loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
from pre_processor import PreProcessor
from feature_extracter import FeatureExtracter

def get_clean_reviews(data):
    clean_reviews = []
    intent_result = []
    preprocessor = PreProcessor();
    for intent,docs in data.items():
        for example in docs:
            clean_reviews.append(preprocessor.process(example,stop_word=True))
            intent_result.append(intent)
    return clean_reviews,intent_result

if __name__ == "__main__":
    model = Word2Vec.load("model/classifier")
    num_features = 5

    data_loader = DataLoader()
    data = data_loader.read_in_data()

    feature_list,intent_result = get_clean_reviews(data)

    print("Creating average feature vectors for training data....")
    feature_extracter = FeatureExtracter()
    train_data_vector = feature_extracter.make_average_feature_vector(feature_list, model, num_features)

    data_test = data_loader.read_test_data()
    feature_list_test,_ = get_clean_reviews(data_test)
    print(feature_list_test)
    print("Creating average feature vectors for test data....")

    test_data_vector = feature_extracter.make_average_feature_vector(feature_list_test, model, num_features)

    forest = RandomForestClassifier(n_estimators=100)
    print("Fitting a random forest to labeled training data...")

    print(intent_result)
    forest = forest.fit(train_data_vector, intent_result)

    result = forest.predict(test_data_vector)
    print(result)
    # end