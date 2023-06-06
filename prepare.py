from sklearn.model_selection import train_test_split
import pandas as pd
from pydataset import data
from env import get_db_url
import acquire


def split_titanic_data(df):
   
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.survived)
    
    
    print("Train Titanic Data:")
    print(train.head())

    print("Test Titanic Data:")
    print(test.head())

    print("Validation Titanic Data:")
    print(validate.head())

def split_telco_data(df):
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.active)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.active)
    return train, validate, test

def split_iris_data(df):
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species_setosa)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.species_setosa)
    return train, validate, test
