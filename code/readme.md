# code
The folder contains the scripts, notebooks, and txt files needed to reproduce this project. 
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

* `classifier_emb.py` this script trains the classifiers Logistic Regression and Support Vector Machine and predicts the learned labels on the test data. For this approach embeddings are being used. To access the BankFin embeddings download them [here](https://github.com/sid321axn/bank_fin_embedding) and the GoogleNews [here](https://code.google.com/archive/p/word2vec/) and save them in the [models](https://github.com/cltl-students/Weber_Elena_Automatic_Topic_Classification/tree/main/models) folder. To tokenize and vectorize the data the following script is needed: 

    * `embeddings_features.py`
###
* `transformers.ipynb` this notebook fine-tunes the pre-trained models on the different datasets, this has to be specified within the notebook by uncommenting certain parts. The fine-tuned model is then saved to [\models](https://github.com/cltl-students/Weber_Elena_Automatic_Topic_Classification/tree/main/models). The default for this notebook is *DistilBERT* and the original dataset.
