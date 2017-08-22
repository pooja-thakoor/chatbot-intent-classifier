# chatbot-intent-classifier : Intent Detection Engine for Chatbot

The project identifies the intent associated with the query based on the intents provided to classifier.

## Getting Started


### Prerequisites

* Python 3+ are supported.

### Dependencies

* gensim
* scikit-learn
* numpy
* scipy

### Installing

    git clone https://github.com/pooja-thakoor/chatbot-intent-classifier.git
    cd chatbot-intent-classifier
    pip install -r requirements.txt
    cd chatbot-intent-classifier

**Training**
```python
    python data_loader.py    
```

**Testing**
```python
    python classifier.py
```

## Authors

**Pooja Thakoor** 

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* https://radimrehurek.com/gensim/models/word2vec.html
* http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
