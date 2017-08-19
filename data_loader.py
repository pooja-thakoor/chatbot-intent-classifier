from w2c_generator import W2CFeatureGenerator
import os
from feature_extracter import FeatureExtracter

class DataLoader:
    def __init__(self):
        self.data = {}
        self.trainSet = []
        self.featureList = []
        self.TRAINING_DATA_PATH = 'training_data/'
        self.TEST_DATA_PATH = 'test_data/'
        self.MODEL_PATH = 'model/'
        if not os.path.exists(self.MODEL_PATH):
            os.makedirs(self.MODEL_PATH)

    def read_in_data(self):
        print('Extracting training data.....\n')
        self.data = {}
        for label in os.listdir(self.TRAINING_DATA_PATH):
            file = self.TRAINING_DATA_PATH + label
            f = open(file, 'r')
            self.data[label] = f.readlines()
            f.close
        return self.data
        # end

    def read_test_data(self):
        self.data = {}
        for label in os.listdir(self.TEST_DATA_PATH):
            file = self.TEST_DATA_PATH + label
            f = open(file, 'r')
            self.data[label] = f.readlines()
            f.close
        return self.data

if __name__ == "__main__":
    # dictionary tweets and label
    data_loader = DataLoader()
    data = data_loader.read_in_data()

    feature_extracter = FeatureExtracter();
    feature_list = feature_extracter.perform_feature_extraction(data);

    print("initial features - " + str(feature_list))
    w2c_generator = W2CFeatureGenerator();

    w2c_generator.generate_features(feature_list)

    # end