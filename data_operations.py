import pandas as pd
from tools.experimental.RAG.embeddings import get_embedding, reduce_df
import os
# TODO: probably delete or move to some RAG related module
embedded_articles_path = os.path.join("data", "embedded_data.csv")
original_articles_path = os.path.join("data", "medium.csv")

def _modify_articles_df(articles_df):
    '''
    The goal of this method is to create vector embeddings from the provided dataframe.
    The embeddings are created only for the rows that do not exceed OpenAI API limits.
    '''
    try:
        articles_df = reduce_df(articles_df)
        if articles_df is None:
            print("An error occurred while processing the DataFrame. Please upload the file again.")
            return None
        
        print("Creating embeddings ...")
        articles_df['embedded_values'] = articles_df['Text'].apply(get_embedding)
        if articles_df['embedded_values'].isnull().values.any():
            print("An error occurred while creating embeddings. Please upload the file again.")
            return None

        print("Saving csv ...")
        articles_df.to_csv(embedded_articles_path, index=False)
    except Exception as e:
        print(f"An error occurred while modifying the DataFrame: {str(e)}")
        return None
    
    return articles_df


def _load_and_prepare_csv():
    '''
    Goal of this method is to load csv and apply modify_articles_df method on that file. 
    It is implemented in order to apply to DRY principle. (Don't Repeat Yourself)
    '''
    try:
        articles_df = pd.read_csv(original_articles_path)
        articles_df = _modify_articles_df(articles_df)
        return(articles_df)
    except Exception as e:
        print(f"An error occurred while loading 'medium.csv': {str(e)}")
        articles_df = None
        return(articles_df)


def load_articles_df():
    if os.path.isfile(embedded_articles_path):
        print("The file 'embedded_data.csv' already exists in this directory.")
        response = input("Do you want to load data once again and apply embedding anyway? (Y/N) - default N ").strip().upper()
        if response != 'Y':
            articles_df = pd.read_csv(embedded_articles_path)
        else:
            articles_df = _load_and_prepare_csv()
    else:
            articles_df = _load_and_prepare_csv()

    return articles_df
