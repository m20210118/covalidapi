from flask import *
from flask_cors import *
import json
import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
from tqdm import tqdm
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")


app = Flask(__name__)
model = tf.keras.models.load_model('legit.model')


@app.route('/', methods=['GET'])
def home_page():
    data = {'message': 'Home page'}
    response = json.dumps(data)
    return response


@app.route('/validate/', methods=['GET','POST'])# /validate/?text=
@cross_origin()
def validate_tweet():
    user_text = ""
    if request.method == 'POST':
        content = request.json
        user_text = content['text']
    else:
        user_text = str(request.args.get('text'))

    custom_text = []

    for r in tqdm([user_text]):
        emb = use(r)
        custom_emb = tf.reshape(emb, [-1]).numpy()
        custom_text.append(custom_emb)

    custom_text = np.array(custom_text)
    prediction = model.predict(custom_text)
    print("Real: ", prediction[0][1] * 100)
    print("Fake: ", prediction[0][0] * 100)
    data = {'user_text':user_text,'real': prediction[0][1] * 100, 'fake':prediction[0][0] * 100 }

    response = json.dumps(data)
    return response


if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)