# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"customerID": pd.Series(["example_value"], dtype="object"), "gender": pd.Series(["example_value"], dtype="object"), "SeniorCitizen": pd.Series([0], dtype="int64"), "Partner": pd.Series([False], dtype="bool"), "Dependents": pd.Series([False], dtype="bool"), "tenure": pd.Series([0], dtype="int64"), "PhoneService": pd.Series([False], dtype="bool"), "MultipleLines": pd.Series(["example_value"], dtype="object"), "InternetService": pd.Series(["example_value"], dtype="object"), "OnlineSecurity": pd.Series(["example_value"], dtype="object"), "OnlineBackup": pd.Series(["example_value"], dtype="object"), "DeviceProtection": pd.Series(["example_value"], dtype="object"), "TechSupport": pd.Series(["example_value"], dtype="object"), "StreamingTV": pd.Series(["example_value"], dtype="object"), "StreamingMovies": pd.Series(["example_value"], dtype="object"), "Contract": pd.Series(["example_value"], dtype="object"), "PaperlessBilling": pd.Series([False], dtype="bool"), "PaymentMethod": pd.Series(["example_value"], dtype="object"), "MonthlyCharges": pd.Series([0.0], dtype="float64"), "TotalCharges": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
