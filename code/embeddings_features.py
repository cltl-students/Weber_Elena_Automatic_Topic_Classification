import numpy as np
# import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import nltk
# from finished_scripts.backup_prepro import preprocessing_feedback_backup2
# from textblob import TextBlob


def create_vectorizer_traditional_features(feature_values):
    '''
    Function that creates vectorizer for set of feature values
    :param feature_values: list of dictionaries containing feature-value pairs
    :type feature_values: list of dictionairies (key and values are strings)
    :return vectorizer: vectorizer with feature values fitted
    '''
    vectorizer = DictVectorizer()
    vectorizer.fit(feature_values)
    
    return vectorizer

def combine_sparse_and_dense_features(dense_vectors, sparse_features):
    '''
    Function that takes sparse and dense feature representations and appends their vector representation
    :param dense_vectors: list of dense vector representations
    :param sparse_features: list of sparse vector representations
    :type dense_vector: list of arrays
    :type sparse_features: list of lists
    :return combined_vectors: list of arrays in which sparse and dense vectors are concatenated
    '''
    combined_vectors = []
    sparse_vectors = np.array(sparse_features.toarray())

    for index, vector in enumerate(sparse_vectors):
        combined_vector = np.concatenate((vector,dense_vectors[index]))
        combined_vectors.append(combined_vector)

    return combined_vectors

def tokenize_data(text):
    '''
    function to tokenize the data 
    :param text: a string that has to be tokenized
    :return text_tokens: the tokenized texts
    '''
    text_tokens = nltk.tokenize.word_tokenize(text)
    return text_tokens

# Transform features into embeddings:
def feat_to_input(data, emb_model, dimensions): #, features):
    '''
    Transform all features to NN input. Including  word embedding
    :param data: data that is to be embedded, list of strings
    :param emb_model: embedding model used 
    :param dimensions: dimensions of embedding model
    :return vectors: vectorized data 
    '''
    index2word_set = set(emb_model.index_to_key)

    vectors = []
    
    # for words in data:
    for comment in data:
        # # tokenize
        words = tokenize_data(comment)
        featureVec = np.zeros(dimensions, dtype="float32")
        
        unknown_word = 0
        for word in words:
            if word in index2word_set: 
                w_vec = emb_model[word]
                # combine each word embedding through normalized sum
                featureVec = np.add(featureVec,w_vec/np.linalg.norm(w_vec))
            else: 
                unknown_word +=1
            if unknown_word > 0:
                featureVec = np.divide(featureVec, unknown_word)        
        vectors.append(featureVec)
    return vectors 


