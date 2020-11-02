import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    function to load flat file data into SQL Lite database
    parameters include filepath for messages and file path for categories
    file.
    function returns pandas dataframe that merges the two datasets by id   
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(categories, messages, on = 'id')
    return df


def clean_data(df):
    """
    function performs data cleansing by taking the pandas dataframe that 
    was created by merging two flat files.
    data cleansing include applying lambda functions to transform category column 
    names.
    transformations include splitting categories into seperate values and using first
    row of categories dataframe to create column names for category data
    also includes renames category columns with new names, remove duplicates
    """
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[1]
    category_colnames = row.apply(lambda word: word.split('-')[0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda number: number.split('-')[1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    df.drop('categories',axis=1,inplace=True)
    df = df.join(categories)
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    save the data frame as a table within SQL Lite database
    function parameters including passing the dataframe and file
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()