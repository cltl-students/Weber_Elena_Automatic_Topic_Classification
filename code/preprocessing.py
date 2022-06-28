import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#  from textblob import TextBlob 
# from autocorrect import Speller

def preprocessing_feedback(sequences):
    '''
    This function provides all the pre-processing steps, i.e., tokenizing, removing names, lowercasing the names, removing stopwords, removing frequent words, removing undefined characters,
    remove punctuation, remove digits, and lemmatizing
    :param sequences: as an input it takes a list of the feedback statements
    :return: the function returns a list of the pre-processed feedback statements
    '''
    new_sequences = []

    for sequence in sequences:
        
        # tokenize the words
        tokens = nltk.word_tokenize(sequence)
        
        # if you wish to do a spelling check, uncomment the following
        # spell = Speller(lang='en')
        # tokens = spell(str(tokens))

        # remove names in the tokens
        remove_names = []
        with open(r'\code\list_names.txt', 'r', encoding ='utf-8') as f:
            lines = f.read().splitlines()
        for name in tokens:
            if name not in lines:
                remove_names.append(name)

        # lowercase the tokens
        lowercase_tokens = []
        for token in remove_names:
            lowercase_tokens.append(token.lower())

        #uncomment if you also wish to remove company-related named entities
        # remove_organizations = []
        # with open(r'\code\de_volksbank_terms.txt', 'r', encoding = 'utf-8') as f:
            # lines = f.read().splitlines()
        # for word in lowercase_tokens:
            # if word not in lines:
                # remove_organizations.append(word)

        # remove stopwords
        removed_stopwords = []
        stop_words = set(stopwords.words('english'))
        for word in lowercase_tokens:
        # in case you wanted to remove the company-related named entities, uncomment the following line and uncomment the one above:
        # for word in remove_organizations:
            if word not in stop_words:
                removed_stopwords.append(word)

        # remove frequent words, manually created list from most frequent words within data 
        removed_frequent_words = []
        # the list was created after manually checking the most frequent words within the sequences
        exclude_frequent_words = ['everything', 'always', 'also', 'take', 'u', 'call', 'give', 'get', 'sn', 'mr', 
        'receive', 'make', 'last', 'could', 'would', 'along', 'year', 'via', 'work', 'blg', 'like', 'better', 'pp', 'never', 'without', 'lot', 'new', 'thing', 
        'want', 'extra', 'still', 'much','possible', 'ask', 'long', 'come', 'little', 'first', 'pay', "'s", "n't", "'m", "'ll", "'ve", "'re", "'d", 'ik']
        for word in removed_stopwords:
            if word not in exclude_frequent_words:
                removed_frequent_words.append(word)

        #remove undefined characters
        remove_undefined_char = []
        undefined_char = ['ã¢â\\x80â\\xa0', '\x80', '0.0', 'ã¢ââ¬', 'ã¢ââ', 'assurantiãâ « n']
        for char in removed_frequent_words:
            if char not in undefined_char:
                remove_undefined_char.append(char)
        
        # remove punctuation
        removed_punctuation = []
        punctuation = ['.', ',', '!', '?', '/', '(', ')', '#', '@', '&', '*', '[', ']', '`', ':', '``', "'", '"', '...', '..', ';', '%']
        for char in remove_undefined_char:
            if char not in punctuation:
                removed_punctuation.append(char)
                # loop through it more often so the others are also found? 

        # remove digits
        remove_digits = []
        # Iterate through the string, adding non-numbers to the no_digits list
        for word in removed_punctuation:
            if not str(word).isdigit():
                remove_digits.append(word)

        # lemmatize, also additionally lemmatize verbs 
        pos_tags = nltk.pos_tag(remove_digits)
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for w, t in pos_tags:
            if 'V' in t:
                lemmas.append(lemmatizer.lemmatize(w, 'v'))
            else:
                lemmas.append(lemmatizer.lemmatize(w))
                
        # save all to new list
        new_sequences.append(lemmas)

    return new_sequences 