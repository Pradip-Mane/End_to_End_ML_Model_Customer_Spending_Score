import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            num_col=["Age","Annual Income (k$)"]
            cat_col=["Gender"]
        
            num_pipeline=Pipeline(steps=[("Imputer",SimpleImputer(strategy="mean")),
                                        ("scalar", StandardScaler())])
            
            cat_pipeline=Pipeline(steps=[("Imputer",SimpleImputer(strategy="most_frequent")),
                                        ("one_hot_encoder",OneHotEncoder())])
            
            logging.info("Numrical column standard scalling completed")    
            logging.info("Categorical column encoding completed")

            preprocessor=ColumnTransformer([("num_pipeline",num_pipeline,num_col),
                                            ("cat_pipeline",cat_pipeline,cat_col)])
            
            return preprocessor




        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()
            target_col="Spending Score (1-100)"
            num_col=["Age","Annual Income (k$)"]
            cat_col=["Gender"]

            input_features_train_df=train_df.drop(target_col, axis=1)
            target_train_df=train_df[target_col]

            input_features_test_df=test_df.drop(target_col,axis=1)
            target_test_df=test_df[target_col]

            logging.info("Applying preprocessing object to training abd testing features")

            input_features_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessing_obj.transform(input_features_test_df)

            train_arr=np.c_[input_features_train_arr,np.array(target_train_df)]
            test_arr=np.c_[input_features_test_arr,np.array(target_test_df)]

            logging.info(f"Saved preprocessing object.")

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
        


        
