import sklearn
import pandas as pd
import sys 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from preprocessing import preprocessing_feedback
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report 
import seaborn as sns

def read_file(inputfile_train, inputfile_test):
    '''
    the function reads in the csv files of the train and test datasets and appends the sentences and the topics to lists
    :param inputfile_train: the filepath to the train dataset
    :param inputfile_test: the filepath to the test dataset
    :return training_instances: a list of feedback statements of the train dataset
    :return training_labels: a list of the corresponding topics of training_instances
    :return test_instances: a list of feedback statements of the test dataset
    :return test_labels: a list of the correcsponding topics of test_instances
    '''
    dftrain = pd.read_csv(inputfile_train, delimiter=';', header= 0, dtype= str, keep_default_na=False, encoding= 'latin1')
    dftest = pd.read_csv(inputfile_test, delimiter = ';', header= 0, dtype= str, keep_default_na=False, encoding= 'latin1')

    training_instances=[]
    for sentence in dftrain['Sentence_new_improved']:
        training_instances.append(sentence)

    test_instances=[]
    for sentence in dftest['Sentence_new_improved']:
        test_instances.append(sentence)

    training_labels=[]
    for label in dftrain['CsatTopicEn']:
        training_labels.append(label)

    test_labels=[]
    for label in dftest['CsatTopicEn']:
        test_labels.append(label)

    return training_instances, training_labels, test_instances, test_labels

def preprocessing_feedbacks(training_instances, test_instances):
    '''
    this function pre-processes the training and test instances derived in the function above
    :param training_instances: a list of feedback statements from the train dataset
    :param test_instances: a list of feedback statements from the test dataset
    :return texts_train: a list of pre-processed training instances
    :return texts_test: a list of pre-processed test instances
    '''
    texts_train = [" ".join(text) for text in preprocessing_feedback(training_instances)]
    texts_test = [" ".join(text) for text in preprocessing_feedback(test_instances)]

    return texts_train, texts_test

def countvectorize_tfidf(texts_train, texts_test):
    '''
    First CountVectorizer takes a lists of texts and converts the lists into a vector representation with numbers and then the raw values are converted into TF-IDF values to add more information
    :param texts_train: list of pre-processed feedback statements train data
    :param texts_test: list of pre-processed feedback statements test data
    :return training_tfidf_vectors: ndarray, array of shape (n_samples, n_features_new) 
    :return test_tfidf_vectors: ndarray, array of shape (n_samples, n_features_new)

    approach inspired by Piek Vossen, [https://github.com/cltl/ma-hlt-labs/blob/master/lab3.machine_learning/Lab3.5.ml.emotion-detection-bow.ipynb]
    '''
    utterance_vec =CountVectorizer()
    training_count_vectors = utterance_vec.fit_transform(texts_train)
    test_count_vectors = utterance_vec.transform(texts_test)
    # Convert raw frequency counts into TF-IDF values
    tfidf_transformer = TfidfTransformer()
    training_tfidf_vectors = tfidf_transformer.fit_transform(training_count_vectors)
    test_tfidf_vectors = tfidf_transformer.fit_transform(test_count_vectors)

    return training_tfidf_vectors, test_tfidf_vectors


def classifier(modelname, training_labels, test_instances, test_labels, training_tfidf_vectors, test_tfidf_vectors, dataversion):
    '''
    the function uses different classifiers to make predictions using the vectors created in the function above
    :param modelname: the name of the algorithm model, e.g. 'logreg', 'NB', 'SVM' 
    :param training_labels: list of labels corresponding to train set
    :param test_instances: list of feedback statements of test data
    :param test_labels: list of labels corresponding to feedback statements of test data
    :param training_tfidf_vectors: array of vectors created in function above of train data 
    :param test_tfidf_vectors: array of vectors created in function above of test data
    :param dataversion: since various data adaptation methods are implemented, the dataversion helps labeling the predictions
    :return targets_all: list of encoded classes using LabelEncoder
    :return test_classes: list of encoded and transformed test classes using LabelEncoder
    :return model: returns name of classification model 
    :return predictions: predictions on the test dataset after classification models were trained using training_tfidf_vectors 

    saves predictions to a csv file: feedback statement; gold labels; predicted labels
    '''
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(training_labels+test_labels)
    targets_all = list(label_encoder.classes_)
    training_classes = label_encoder.transform(training_labels)
    test_classes = label_encoder.transform(test_labels)

    if modelname == 'logreg':
        model = LogisticRegression(max_iter = 1000)
    if modelname == 'NB':
        model = MultinomialNB()
    if modelname == 'SVM':
        model = svm.LinearSVC(max_iter = 10000)

    model.fit(training_tfidf_vectors, training_classes)

    predictions = model.predict(test_tfidf_vectors)
    
    predicted_labels = label_encoder.classes_[predictions]

    result_frame = pd.DataFrame()
    result_frame['Sentence_new_improved']= test_instances
    result_frame['CsatTopicEn']=test_labels

    result_frame['CsatTopicEn Predictions']=predicted_labels

    outputpath = f'/results/predictions/predictions_{model}_{dataversion}.csv'
    result_frame.to_csv(outputpath, sep = ';', index = False)

    return targets_all, test_classes, model, predictions

def report_classification(test_classes, predictions, targets_all, dataversion, modelname):
    '''
    creates a classification report for the predictions
    :param test_classes: list of encoded and transformed test classes
    :param predictions: predictions on the test dataset after classification models were trained using training_tfidf_vectors 
    :param targets_all: list of encoded classes
    :param dataversion: since various data adaptation methods are implemented, the dataversion helps labeling the predictions
    :param modelname: the name of the algorithm model, e.g. 'logreg', 'NB', 'SVM' 
    
    saves report as csv file with dataversion and classification model
    '''
    report = classification_report(test_classes,predictions,digits = 3, target_names = sorted(targets_all),zero_division=1, output_dict= True)
    # print(f"Classication Report for", modelname)
    # print(report)
    report = pd.DataFrame(report).T
    report = report.round(3)
    report.to_csv(f'/results/bow/classification_report_{dataversion}_{modelname}.csv', sep = ';')

def confusion_matrix(test_classes, predictions, targets_all, model, dataversion):
    '''
    creates a confusion matrix out of the predictions and the gold labels
    :param test_classes: list of encoded and transformed test classes
    :param predictions: predictions on the test dataset after classification models were trained using training_tfidf_vectors 
    :param targets_all: list of encoded classes
    :param dataversion: since various data adaptation methods are implemented, the dataversion helps labeling the predictions
    :param modelname: the name of the algorithm model, e.g. 'logreg', 'NB', 'SVM' 
    
    saves confusion matrix as png file with dataversion and classification model
    '''
    fig, ax = plt.subplots(figsize = (12,8))
    cm = sklearn.metrics.confusion_matrix(test_classes,predictions)
    matrix = sns.heatmap(cm, annot=True, xticklabels = targets_all, yticklabels = targets_all,  fmt='', cmap='Reds', ax = ax)
    fig = matrix.get_figure()
    outputfilepath = f"/figures/bow/confusion_matrix_{model}_{dataversion}.png"
    plt.tight_layout()
    fig.savefig(outputfilepath) 

def main(argv=None):
    '''
    combines all functions to create the whole classification process 

    how to run: 
    python classifier.py filepath_trainingfile filepath_testfile dataversion

    dataversion can be: original, merged, other, undersampled, backtranslation10, backtranslation20 
    '''

    if argv is None:
        argv = sys.argv

    trainingfile = argv[1]
    testfile = argv[2]
    dataversion = argv[3]
    
    training_instances, training_labels, test_instances, test_labels = read_file(trainingfile, testfile)
    texts_train, texts_test = preprocessing_feedbacks(training_instances, test_instances)
    training_tfidf_vectors, test_tfidf_vectors = countvectorize_tfidf(texts_train, texts_test)
    for modelname in ['NB', 'logreg', 'SVM']:
        targets_all, test_classes, model, predictions = classifier(modelname, training_labels, test_instances, test_labels, training_tfidf_vectors, test_tfidf_vectors, dataversion)
        report_classification(test_classes, predictions, targets_all, dataversion, model)
        confusion_matrix(test_classes, predictions, targets_all, model, dataversion)
    
if __name__ == '__main__':
    main()