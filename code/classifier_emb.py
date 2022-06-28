import pandas as pd
from sklearn import svm 
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report 
import sklearn
import gensim 
import sys
from embeddings_features import feat_to_input

#######################################################################
## download the embedding models here and save them folder 'models' ###
## BankFin: https://github.com/sid321axn/bank_fin_embedding         ###
## GoogleNews: https://code.google.com/archive/p/word2vec/          ###
#######################################################################

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


def labelencoder(training_labels, test_labels):
    '''
    the function encodes the labels of the train and test data 
    :param training_labels: the list of training_labels created in the function above
    :param test_labels: the list of test_labels created in the function above
    :return targets_all: list of all classes
    :return training_classes: a numpy ndarray of the encoded training classes
    :return test_classes: a numpy ndarray of the encoded test classes
    '''
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(training_labels+test_labels)
    targets_all = list(label_encoder.classes_)
    training_classes = label_encoder.transform(training_labels)
    test_classes = label_encoder.transform(test_labels)

    return targets_all, training_classes, test_classes

def add_embeddings(embeddings_path, training_instances, test_instances, dimension):
    '''
    create vectors with embeddings
    :param embeddings_path: path to the embeddings file
    :param training_instances: list of feedback statements of train set
    :param test_instances: list of feedback statements of test set
    :param dimension: dimension of embeddings, the BankFin has a dimension of 100 and GoogleNews 300 
    :return train_vec, test_vec: list of the vectorized instances
    '''
    
    if embeddings_path == r'\models\embedding_model': #might need to adapt this if you want to reproduce the code 
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path)
    else: 
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary = True, encoding = 'latin1')
    
    train_vec = feat_to_input(training_instances, emb_model, dimension)
    test_vec = feat_to_input(test_instances, emb_model, dimension)

    return train_vec, test_vec

def classify_data(modelname, train_vec, test_vec, training_classes):
    '''
    the function trains classifiers and then predicts on the test data
    :param modelname: name of the classifier, either logreg or SVM
    :param train_vec: the vectorized instances of the train data
    :param test_vec: the vectorized instances of the test data
    :param training_classes: the encoded classes of the train data 
    :return predictions: returns the predicted topics as floats in a numpy ndarray
    '''

    if modelname == 'logreg':
        model = LogisticRegression(max_iter = 1000)

    if modelname == 'SVM':
        model = svm.LinearSVC(max_iter = 10000)
    
    model.fit(train_vec, training_classes)
    predictions = model.predict(test_vec)

    return predictions

def report_classification(test_classes, predictions, targets_all, embeddings_version, dataversion, modelname):
    '''
    creates the classification report and saves it as a csv file 
    :param test_classes: the encoded labels  
    :param predictions: the numpy ndarray of the predictions created in the function above
    :param targets_all: list of all classes
    :param embeddings_version: version of the embeddings, either 'finance' or 'google' 
    :param dataversion: version of dataset, either original, merged, other, backtranslation10, backtranslation20, or undersampled
    :param modelname: name of trained classifier 
    '''
    report = classification_report(test_classes,predictions,digits = 3, target_names = sorted(targets_all),zero_division=1, output_dict= True)
    report = pd.DataFrame(report).T
    report = report.round(3)
    report.to_csv(f'/results/embeddings/classification_report_{dataversion}_{modelname}_{embeddings_version}.csv', sep = ';')


def confusion_matrix(test_classes, predictions, targets_all, embeddings_version, dataversion, modelname):
    '''
    create a confusion matrix out of predictions and gold labels, saves it as a png
    :param test_classes: the encoded labels  
    :param predictions: the numpy ndarray of the predictions created in the function above
    :param targets_all: list of all classes
    :param embeddings_version: version of the embeddings, either 'finance' or 'google'
    :param dataversion: version of dataset, either original, merged, other, backtranslation10, backtranslation20, or undersampled
    :param modelname: name of trained classifier
    '''
    fig, ax = plt.subplots(figsize = (12,8))
    cm = sklearn.metrics.confusion_matrix(test_classes,predictions)
    matrix = sns.heatmap(cm, annot=True, xticklabels = targets_all, yticklabels = targets_all,  fmt='', cmap='Reds', ax = ax)
    fig = matrix.get_figure()
    outputfilepath = f"/figures/embeddings/confusion_matrix_{embeddings_version}_embeddings_{dataversion}_{modelname}.png"
    plt.tight_layout()
    fig.savefig(outputfilepath) 

def main(argv=None):
    '''
    combines the functions above for a complete classification process 
    
    how to run: 
    python classifier_emb.py trainingfilepath testfilepath embeddingspath dimensionofembeddings dataversion embeddingsversion

    dataversion can be: original, merged, other, backtranslation10, backtranslation20, or undersampled
    embeddingsversion can be: google, or finance 
    '''

    if argv is None:
        argv = sys.argv

    trainingfile = argv[1]
    testfile = argv[2]
    embeddings = argv[3]
    dimension = argv[4]
    dataversion = argv[5]
    embeddingsversion = argv[6]
    
    training_instances, training_labels, test_instances, test_labels = read_file(trainingfile, testfile)
    targets_all, training_classes, test_classes = labelencoder(training_labels, test_labels)
    train_vec, test_vec = add_embeddings(embeddings, training_instances, test_instances, int(dimension))
    for modelname in ['logreg', 'SVM']:
        predictions = classify_data(modelname, train_vec, test_vec, training_classes)
        report_classification(test_classes, predictions, targets_all, embeddingsversion, dataversion, modelname)
        confusion_matrix(test_classes, predictions, targets_all, embeddingsversion, dataversion, modelname)

    
if __name__ == '__main__':
    main()