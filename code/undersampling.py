import pandas as pd
import sys

####################################################################################################################################################
####code adapted from https://medium.com/analytics-vidhya/undersampling-and-oversampling-an-old-and-a-new-approach-4f984a0e8392   [11-05-2022] #####
####################################################################################################################################################

def load_dataframe(inputpath): 
    '''
    load the file as a pandas dataframe
    :param inputpath: filepath to the file 
    :return sent: returns the dataframe of the feedback statement and its respective class 
    '''
    df = pd.read_csv(inputpath, delimiter=';', header= 0, dtype= str, keep_default_na=False, encoding= 'latin1')
    sent = df[['Sentence_new_improved', 'CsatTopicEn']]

    return sent

def undersample(df):
    '''
    undersampling the dataframe to its minority class
    :param df: dataframe that is going to be undersampled

    saves the dataframe as a csv file 
    '''
    classes = df.CsatTopicEn.value_counts().to_dict()
    least_class_amount = min(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df['CsatTopicEn'] == key]) 
    classes_sample = []
    for i in range(0,len(classes_list)-1):
        classes_sample.append(classes_list[i].sample(least_class_amount))
    df_maybe = pd.concat(classes_sample)
    final_df_undersampled = pd.concat([df_maybe,classes_list[-1]], axis=0)
    final_df_undersampled = final_df_undersampled.reset_index(drop=True)
    final_df_undersampled.to_csv(r'C:\Users\elena\Desktop\thesis\data\train_undersampled.csv', index = False, sep = ';')

def main(argv=None):

    if argv is None:
        argv = sys.argv

    inputfile = argv[1]

    dataframe = load_dataframe(inputfile)
    undersample(dataframe)
    
if __name__ == '__main__':
    main()