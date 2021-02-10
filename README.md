# Capstone-Project---Azure-Machine-Learning-Engineer

## Overview
This project is part of the Udacity Azure ML Nanodegree.
We create Two models: one using Automated ML (denoted as AutoML ) and one customized model whose hyperparameters are tuned using HyperDrive and compare the performance of both the models and deploy the best performing model.


![capstone-diagram](images/capstone-diagram.png)  

## Project Set Up and Installation

Visual Studio code.

![Index](images/Index.png)

Azure Visual Studio code Plugins

![AzureMLplugin](images/AzureMLplugin.png)  

Azure Visual Studio code - Jupyter and python.

![PluginsJupyter](images/PluginsJupyter.png)  


## Azure Libraries.
pip install azureml-sdk
pip install azureml-core


## Dataset

### Overview

Data set used for this project https://www.kaggle.com/ronitf/heart-disease-uci  (https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv) contains the information of Heart Disease and his dataset contains the medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features.

 The goal field refers to the presence of heart disease in the patient.


Thirteen (13) clinical features:

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- high blood pressure: if the patient has hypertension (boolean)
- platelets: platelets in the blood (kiloplatelets/mL)

- serum creatinine: level of serum creatinine in the blood (mg/dL)
- serum sodium: level of serum sodium in the blood (mEq/L)
- sex: woman or man (binary)
- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)
- [DEATH_EVENT] death event: if the patient deceased during the follow-up period (boolean)

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

We use the data for the prediction whether a person has a heart failure or no. For that we will be using all the variables and we will make some cleaning information for the Hyperparameters option.

AuotML .- For the AutomL we need the Authentication. Login first to ml.azure.com and We need to download the configuration file.
![Configfile](images/Configfile.png)

We need to create a Compute Resource and create the  AutoML Configuration , deploy the best model and register and publish the best model and test the model. 

Hyperparameter Tunning .- We need to configurate the hyperdrive , for this we are going to test with bandpolicys and parameter sampling, using different options: 

    ps = RandomParameterSampling({
		'--C':choice(0.01,0.05, 0.1, 0.5,1),
		'--max_iter':choice(5, 10, 20, 50, 100)
	}
    )

    policy = BanditPolicy(evaluation_interval=2
    ,slack_factor=0.1
    )


We also we create a SKLearn estimator(Binary logistic Regression using train.py file  )We split for train an test the data and use this estimator in the hyperdrive configuation and we get the best model based in the parameters to check which is the best options hyperdrive vs automl.
 

### Access
*TODO*: Explain how you are accessing the data in your workspace.

We load the information to the workspace in azure for the automl using the url provided in the requeriments in udacity platform. 

    if key in ws.datasets.keys(): 
            found = True
            dataset = ws.datasets[key] 

    if not found:
            # Create AML Dataset and register it into Workspace
            example_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv'
            dataset = Dataset.Tabular.from_delimited_files(example_data)        
            #Register Dataset in Workspace
            dataset = dataset.register(workspace=ws,
                                   name=key,
                                   description=description_text)


    df = dataset.to_pandas_dataframe()
    df.describe()

We load the information using the url and Tabular Dataset Factory.

    from azureml.data.dataset_factory import TabularDatasetFactory

    factory = TabularDatasetFactory()
    test_data_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
    test_ds = factory.from_delimited_files(test_data_path)

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
