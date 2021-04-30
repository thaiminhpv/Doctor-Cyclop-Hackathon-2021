from flask import Flask, request, Response
from skimage import io
from server.model_inference.predictor import get_predictions

ON_COLAB = True

if ON_COLAB:
    from flask_ngrok import run_with_ngrok
    app = Flask(__name__)
    run_with_ngrok(app)
else:
    app = Flask(__name__)


@app.route('/', methods=['POST'])
def upload_files():
    case_list = request.json['caseList']
    images = {
        case['caseId']: io.imread(case['url'])
        for case in case_list
    }  # {caseId: raw_data}
    print('Got cases: ', list(images.keys()))

    out = get_predictions(images)

    return out, 200
