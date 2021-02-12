from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

#ds = pd.read_csv('https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv')
url= 'https://raw.githubusercontent.com/nandex7/Capstone-Project---Azure-Machine-Learning-Engineer/main/data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
ds = TabularDatasetFactory.from_delimited_files(path=url,validate='False',separator=',',infer_column_types=True,include_path=False,
set_column_types=None,support_multi_line=False,partition_format=None)

#ds.head(5)

def clean_data(data):
    # Dict for cleaning data
    
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    #x_df = data.dropna()
    #Boolean Values
    x_df.drop('customerID',inplace=True, axis=1)
    x_df['Partner']= x_df.Partner.apply(lambda s:1 if s==True else 0)
    x_df['gender'] =x_df.gender.apply(lambda s:1 if s=="Male" else 0)
    x_df['Dependents']= x_df.Dependents.apply(lambda s:1 if s==True else 0)
    x_df['PhoneService']=x_df.PhoneService.apply(lambda s:1 if s==True else 0)
    x_df['PaperlessBilling']=x_df.PaperlessBilling.apply(lambda s:1 if s==True else 0)
    x_df['Churn']=x_df.Churn.apply(lambda s:1 if s==True else 0)
    
    #Categorical Values Hot Encoding.
    categorical= ['MultipleLines','InternetService'
    , 'OnlineBackup','OnlineSecurity','DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies', 'Contract','PaymentMethod']
    x_df = pd.get_dummies(x_df, columns = categorical)
    
    y_df = x_df.Churn
    x_df = x_df.drop('Churn', axis=1)
    return x_df, y_df
    
x, y = clean_data(ds)

#x.head(5)
#y.head(5)
# TODO: Split data into train and test sets.
### YOUR CODE HERE ###a
x_train, x_test, y_train,y_test= train_test_split(x,y,test_size=0.33, random_state=42)

run = Run.get_context()


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()