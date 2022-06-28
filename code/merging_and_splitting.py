import pandas as pd
from sklearn.model_selection import train_test_split
import sys

#################################################################################
## Script to merge topics with overlapping content or underrepresented topics ##
##                      into one topic called 'Other'                         ##
###############################################################################


def merge_topics(filepath, dataversion = None):
    '''
    this function merges the following topics with overlapping content:
    - Quality and Offering + Price and Quality = Price and Quality
    - Empoyee Contact + Employee Attitude and Behavior + Employee Knowledge & Skills = Employee 
    or merges underrepresented topics to one topic called 'Other':
    - Digital Options + Price and Quality + Quality and Offering + Employee Contact = Other 
    :param filepath: the filepath to the original dataset 
    :param dataversion: default set to None, if set to 'merged' the topics with overlapping content are being merged, if set to 'other' underrepresented topics are being merged
    :return: the new dataframe with the updated classes 

    saves the new file as as csv with respective dataversion as a name 
    '''

    new_df = pd.read_csv(filepath, delimiter=';', header= 0, dtype= str, keep_default_na=False, encoding= 'latin1', quotechar= '"')
    if dataversion == 'merged':
        replacement_mapping_dict = {
        "Quality and Offering": "Price and Quality",
        "Employee Contact": "Employee",
        "Employee Attitude and Behavior": "Employee",
        "Employee Knowledge & Skills": "Employee"}
        new_df["CsatTopicEn"] = new_df["CsatTopicEn"].replace(replacement_mapping_dict)


    elif dataversion == 'other':
        replacement_mapping_dict_other = {
        "Digital Options": "Other",
        "Price and Quality": "Other",
        "Quality and Offering": "Other",
        "Employee Contact": "Other"}
        new_df["CsatTopicEn"] = new_df["CsatTopicEn"].replace(replacement_mapping_dict_other)


    outputpath = f'/data/{dataversion}.csv'
    new_df.to_csv(outputpath, sep = ';')

    return new_df

def split_data(new_df, dataversion):
    '''
    stratified splitting into train (80%), test (10%), and validation dataset (10%)
    :param new_df: the newly created dataframe with the updated classes
    :param dataversion: name of the class adaptation procedure, 'merged' or 'other' 

    saves the three datasets as csv files 
    '''

    train,test_temp = train_test_split(new_df, test_size=0.20, random_state=5)
    test, validation = train_test_split(test_temp, test_size=0.50, random_state=0)

    #save the data
    
    train.to_csv(f"/data/train_{dataversion}.csv",index=False, sep= ';')
    validation.to_csv(f"/data/valid_{dataversion}.csv",index=False, sep= ';')
    test.to_csv(f"/data/test_{dataversion}.csv",index=False, sep= ';')

def main(argv=None):
    '''
    combining the merging approaches and the stratified splitting 

    how to run:
    python merging_and_splitting.py inputfile dataversion

    dataversion can be: merged, other
    '''

    if argv is None:
        argv = sys.argv

    inputfile = argv[1]
    dataversion = argv[2]


    new_dataframe = merge_topics(inputfile, dataversion)
    split_data(new_dataframe, dataversion)


    
if __name__ == '__main__':
    main()