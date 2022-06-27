# Automatic Topic Classification of Customer Feedback in the Banking Domain 

## A Thesis Project by *Elena Weber* 
## Master's Degree in 'Linguistics: Text Mining', Vrije Universiteit Amsterdam 2021/2022

This repository belongs to the Master's Thesis Project "*Automatic Topic Classification of Customer Feedback in the Banking Domain*" by Elena Weber, supervised by Dr. Ilia Markov and Gabriele Catanese. 
The project was carried out in collaboration with the company [Underlined](https://underlined.eu/), a Dutch software and consulting company providing solutions for CX Analytics and linking customer insights with company performance.

** Update**!!!! The project focuses on the Automatic Topic Classification of customer feedback derived from the banking domain. Topic Classification is about assigning a topic to a text or document with a predefined topic. For this, several traditional machine learning techniques, Bag of Words in combination with TF-IDF as well as embeddings, with the classification algorithms Support Vector Machine, Logistic Regression, and Naive Bayes are compared to fine-tuned transfer learning models, BERT, RoBERTa, and DistilBERT. Additionally, several topic and data adaptation methods are being tested to see if the classifier's performance improves. The topic adaptation methods focus on merging topics with overlapping content as well as merging topics that are inherently underrepresented within the dataset. In contrast, the data augmentation methods focus on undersampling and back-translation.

The process, setup, results, and evaluation of the results can be found in the thesis report, `XXXXXXXXXXXXXXXXXPDF.pdf`. 

**Note:** Since the data cannot be shared with third-parties, it cannot be found in this repository. Outputs that give an indication about its content have been hidden. 

## Content

### Folder Structure 
```
Thesis Project Structure 
└───code
│       |   classifier_bow_tfidf.py
        |   classifier_emb.py
        |   de_volksbank_terms.txt
        |   embeddings_features.py
        |   list_names.txt
        |   merging_and_splitting.py
        |   preprocessing.py
        |   statistics.ibynp
        |   transformers.ipynb
        |   undersampling.py
└───data
│       │   example_dataset.csv 
└───figures
│   └───bow
│   └───embeddings
│   └───transformers
└───models
└───results
│   └───bow
│   └───embeddings
│   └───predictions
│   └───transformers
│   .gitignore
│   LICENSE
│   README.md
│   requirements.txt
```

### \code
The folder [code](www.linkcode.com) contains the scripts, notebooks, and txt files needed to reproduce this project. 
To reproduce the project, follow the order of the files listed below:

* `statistics.ipynb` generates statistics regarding the fully annotated dataset. It also includes the code to split the data. 

For data adaptation: 
* `merging_and_splitting.py` depending on the specification, the script either merges topics with overlapping content (*Quality and Offering + Price and Quality* = **Price and Quality**, *Empoyee Contact + Employee Attitude and Behavior + Employee Knowledge & Skills* = **Employee**) or underrepresented topics into one (*Digital Options + Price and Quality + Quality and Offering + Employee Contact* = **Other**)

* `undersampling.py` this script undersamples the input data, it is recommended to use the train dataset for this

Topic Classification Task: 
* `classifier_bow_tfidf.py` this script trains various classifier, Naive Bayes, Logistic Regression, and Support Vector Machine and predicts the learned labels on the test data set using a Bag of Words combined with TF-IDF approach. For this the following files are needed:
    * `preprocessing.py` this script serves as the preprocessing function, containing tokenizing, removal of named entities (`list_names.txt`, `de_volksbank_terms.txt`), lowercasing, removing stopwords, removing frequent words, removing undefinded characters, removing punctuation, removing digits, and lemmatizing. 

   * `list_names.txt` 

    Optional: removing company-related terms using `de_volksbank_terms.txt` and spelling correction :

    * `de_volksbank_terms.txt` 

* `classifier_emb.py` this script trains the classifiers Logistic Regression and Support Vector Machine and predicts the learned labels on the test data. For this approach embeddings are being used. To access the BankFin embeddings download them [here](https://github.com/sid321axn/bank_fin_embedding) and the GoogleNews [here](https://code.google.com/archive/p/word2vec/) and save them in the [models](www.foldermodels.com) folder. To tokenize and vectorize the data the following script is needed: 

    * `embeddings_features.py`

* `transformers.ipynb` this notebook fine-tunes the pre-trained models on the different datasets, this has to be specified within the notebook by uncommenting certain parts. The fine-tuned model is then saved to [\models](www.foldermodels.com). The default for this notebook is *DistilBERT* and the original dataset.

### \data
The data folder only contains the `example_dataset.csv` since the data is not allowed to be shared. However, the csv file shows the structure of the data. 

### \figures
The folder is used to store the figures, for instance, the confusion matrices of each classification approach. It is separated into the different approaches and each contains one example:

* \bow
* \embeddings
* \transformers

### \models
In this folder, the embedding models and the transformer-based models are being stored. To download the embedding models, please find them here: [BankFin](https://github.com/sid321axn/bank_fin_embedding) and [GoogleNews](https://code.google.com/archive/p/word2vec/)

### \results
This folder contains the results, i.e., the classification reports and csv files with the feedback statements, their gold labels, and their predicted labels. The classification reports can be found in `\bow`, `\embeddings`, and `\transformers`, and the feedback statements plus predictions in the folder `\predictions`. Each folder contains one example. 
* \bow
* \embeddings
* \predictions
* \transformers

### `requirements.txt`
This txt file lists all the required packages and provides direct installation instructions using pip. 
***

## Thesis Report 
### `Weber-MasterThesis.pdf`
The pdf file contains the full thesis report.
***



# References
The code for the traditional machine learning approaches was partially inspired by Piek Vossen, [Source Code](https://github.com/cltl/ma-hlt-labs/tree/master/lab3.machine_learning).

The code to fine-tune the transformers was adapted from [Gabriele Catanese](https://github.com/cltl-students/catanese_gabriele_text_mining_thesis) who beforehand adapted it from [George Mihaila](https://gmihaila.medium.com/fine-tune-transformers-in-pytorch-using-transformers-57b40450635).
