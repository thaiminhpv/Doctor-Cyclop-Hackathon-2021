import numpy as np
import pandas as pd

from server.model_inference.config import labels
from server.model_inference.core_model import get_model_prediction
from server.util.prediction_to_json import pandas_to_json


def get_predictions(images):
    ids = list(images.keys())
    out = np.hstack((np.asarray(ids)[np.newaxis,].T, (np.zeros((len(ids), len(labels))))))
    df_sub = pd.DataFrame(out, columns=['StudyInstanceUID', *labels])

    predicted_df = get_model_prediction(df_sub, images)
    predicted_json = pandas_to_json(predicted_df)

    return predicted_json
