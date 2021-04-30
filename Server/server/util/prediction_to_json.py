import pandas as pd

thresholds = {
    "ETT - Abnormal": 0.5,
    "ETT - Borderline": 0.5,
    "ETT - Normal": 0.5,
    "NGT - Abnormal": 0.5,
    "NGT - Borderline": 0.5,
    "NGT - Incompletely Imaged": 0.5,
    "NGT - Normal": 0.5,
    "CVC - Abnormal": 0.5,
    "CVC - Borderline": 0.5,
    "CVC - Normal": 0.5
}


def pandas_to_json(prediction: pd.DataFrame):
    out_json = {}
    result = []

    for i, row in prediction.iterrows():
        el = {
            'caseID': row['StudyInstanceUID']
        }

        for label in thresholds:
            threshold = thresholds[label]
            el[label] = {
                'score': row[label],
                'label': row[label] > threshold
            }
        # print(el)
        result.append(el)

    out_json['result'] = result
    return out_json
