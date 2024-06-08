import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''
        
        try:
            #Loading the file which we downloaded from database
            logging.info("Loading the dataset")
            df = pd.read_csv(r'notebook\\UCI_Credit_Card.csv')
            df.head()
            
            #Replacing the column names in the dataset
            df.rename(columns={"default.payment.next.month": "Default"}, inplace=True)
            
            logging.info("Doing Feature engineering to the dataset")
            
            # Initialise SMOTE object
            smote = SMOTE()
            
            X = df.iloc[:,:-1]
            y = df['Default']
            
            
            # fit predictor and target variable
            X_smote, y_smote = smote.fit_resample(X, y)
            
            # Combine balanced X and y
            df_final = pd.DataFrame(X_smote, columns=df.columns[:-1])
            df_final['Default'] = y_smote
            
            # Replace values in SEX, MARRIAGE and EDUCATION variables with original name
            df_final['SEX'] =  df_final['SEX'].replace({1:'male', 2:'female'})
            df_final['EDUCATION'] = df_final['EDUCATION'].replace({1:'Graduation', 2:'University', 3:'High_School', 0:'Others', 4:'Others', 5:'Others', 6:'Others'})
            df_final['MARRIAGE'] = df_final['MARRIAGE'].replace({1:'Married', 2:'Single', 0:'Others', 3:'Others'})
            
            # Change column names PAY_0 to PAY_6
            df_final.rename(columns={'PAY_0':'PAY_SEPT', 'PAY_2':'PAY_AUG', 'PAY_3':'PAY_JULY', 
                   'PAY_4':'PAY_JUNE', 'PAY_5':'PAY_MAY', 'PAY_6':'PAY_APRIL'}, inplace=True)

            # Change column names PAY_AMT_1 to PAY_AMT_6
            df_final.rename(columns={'PAY_AMT1':'PAY_AMT_SEPT', 'PAY_AMT2':'PAY_AMT_AUG', 'PAY_AMT3':'PAY_AMT_JULY', 
                   'PAY_AMT4':'PAY_AMT_JUNE', 'PAY_AMT5':'PAY_AMT_MAY', 'PAY_AMT6':'PAY_AMT_APRIL'}, inplace=True)
                
            # Change column names BILL_AMT_1 to BILL_AMT_6
            df_final.rename(columns={'BILL_AMT1':'BILL_AMT_SEPT', 'BILL_AMT2':'BILL_AMT_AUG', 'BILL_AMT3':'BILL_AMT_JULY', 
                   'BILL_AMT4':'BILL_AMT_JUNE', 'BILL_AMT5':'BILL_AMT_MAY', 'BILL_AMT6':'BILL_AMT_APRIL'}, inplace=True)
            
            df_final['EDUCATION'].replace({0:4,5:4,6:4}, inplace=True)
            df_final['MARRIAGE'].replace({0:3}, inplace=True)
            
            numerical_columns = ["PAY_SEPT", "PAY_AUG", "PAY_JULY", 
                 "PAY_JUNE", "PAY_MAY", "PAY_APRIL", "BILL_AMT_SEPT", "BILL_AMT_AUG", "BILL_AMT_JULY", 
                  "BILL_AMT_JUNE", "BILL_AMT_MAY", "BILL_AMT_APRIL"]
            
            categorical_columns = ["SEX","EDUCATION","MARRIAGE"]
            
            num_pipeline = Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ('scaler',StandardScaler())
            ])
             
            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
            ])
            
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            
        
            #numeric_transformer = StandardScaler()
            #oh_transformer = OneHotEncoder()
            
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns ),
                ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Reading the train and test file")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name="Default"
           
            numerical_columns = ["'PAY_SEPT", "PAY_AUG", "PAY_JULY", 
                 "PAY_JUNE", "PAY_MAY", "PAY_APRIL", "BILL_AMT_SEPT", "BILL_AMT_AUG", "BILL_AMT_JULY", 
                  "BILL_AMT_JUNE", "BILL_AMT_MAY", "BILL_AMT_APRIL"]
            
            
            ## Divide the train dataset to independent and dependent features
            
            
            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
             ## Divide the test dataset to independent and dependent features
             
            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info("Applying Preprocessing on training and test dataframe")
            
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)
            
            
            train_arr = np.c_[
                input_features_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object")
            
            
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
            
            
        
        except Exception as e:
            raise CustomException(e,sys)