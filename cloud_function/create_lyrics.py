import base64
from flask import Flask
from flask import make_response
from flask import jsonify
import requests

from typing import Dict

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from io import BytesIO
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.keras import preprocessing

PROJECT_ID = "dogbot-298321"
ENDPOINT_ID = "3456600674937077760"
LOCATION = "us-central1"
API_ENDPOINT = "us-central1-aiplatform.googleapis.com"

class_names = ['beatles', 'bob-dylan', 'michael-jackson', 'zappa']
nl_models = ['3632804010357948416', '', '', '']

def load_image(url):
    response = requests.get(url)
    img_bytes = BytesIO(response.content)
    img = Image.open(img_bytes)
    img = img.convert('RGB')
    img = img.resize((180,180), Image.NEAREST)
    arr = preprocessing.image.img_to_array(img)
    # arr = tf.expand_dims(arr, 0)
    return tf.convert_to_tensor(arr)

def write_lyrics(request):
    request_json = request.get_json()
    print(request_json)

    if request_json['url']:
        url = request_json['url']
        image_array = load_image(url)
        
        client_options = {"api_endpoint": API_ENDPOINT}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        vals = image_array.numpy().tolist()
        instance = json_format.ParseDict(vals, Value())
        instances=[instance]
        parameters_dict = {}
        parameters = json_format.ParseDict(parameters_dict, Value())
        endpoint = client.endpoint_path(
            project=PROJECT_ID, location=LOCATION, endpoint=ENDPOINT_ID
        )
        response = client.predict(
            endpoint=endpoint, instances=instances, parameters=parameters
        )
        print("response")
        print(" deployed_model_id:", response.deployed_model_id)

        predictions = response.predictions

        predicted_class = np.argmax(predictions[0])
        p_index = predicted_class.astype(int)

        # UPDATE
        model_id = nl_models[0]

        nl_instance = json_format.ParseDict(["it"], Value())
        nl_instances = [nl_instance]
        nl_endpoint = client.endpoint_path(
            project=PROJECT_ID, location=LOCATION, endpoint=nl_endpoint
        )
        nl_response = client.predict(
            endpoint=nl_endpoint, instances=nl_instances, parameters=parameters4
        )

        nl_predictions = nl_response.predictions[0]

        response = make_response(
            jsonify({"predicted_class": str(nl_predictions)}), 200
            )

        response.headers["Content-Type"] = "application/json"
        return response
    else:
        response = make_response(
            jsonify({"error": "No URL provided."}), 500)
        response.headers["Content-Type"] = "application/json"
        return response
