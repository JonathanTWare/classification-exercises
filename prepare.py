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
    
    print(f'Train:{train.shape}')
    print(f'Test:{test.shape}')
    print(f'Val: {validate.shape}')
    
    print("Train Titanic Data:")
    print(train)
    print("Test Titanic Data:")
    print(test)
    print("Validation Titanic Data:")
    print(validate)

def split_telco_data(df):
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.active)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.active)
    
    print(f'Train:{train.shape}')
    print(f'Test:{test.shape}')
    print(f'Val: {validate.shape}')

    print("Train Telco Data:")
    print(train.shape)
    print(train)
    print("Test Telco Data:")
    print(test)
    print("Validation Telco Data:")
    print(validate)

def split_iris_data(df):
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.species)
    
    print(f'Train:{train.shape}')
    print(f'Test:{test.shape}')
    print(f'Val: {validate.shape}')

    print("Train Iris Data:")
    print(train.shape)
    print(train)
    print("Test Iris Data:")
    print(test)
    print("Validation Iris Data:")
    print(validate)
