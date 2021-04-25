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
ENDPOINT_ID = "3456600674937077760" # Entry point to the pipeline (classification)
LOCATION = "us-central1"
API_ENDPOINT = "us-central1-aiplatform.googleapis.com"

class_names = ['beatles', 'bob-dylan', 'michael-jackson', 'zappa']
nl_models = ['3632804010357948416', '', '', '']

# TODO: I know this is brittle but time is of the essence
# Already tokenized on the word "I"
# These are the seed values we will pass directly to the NL model
# Secret artists are here
artist_embeddings = {
    "beatles": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    # "billie-eilish": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]],
    "bob-dylan": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    # "eminem": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]],
    "michael-jackson": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    # "nickelback": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]],
    "zappa": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
}

def load_image(url):
    response = requests.get(url)
    img_bytes = BytesIO(response.content)
    img = Image.open(img_bytes)
    img = img.convert('RGB')
    img = img.resize((180,180), Image.NEAREST)
    arr = preprocessing.image.img_to_array(img)
    return tf.convert_to_tensor(arr)

def get_image_prediction(url, prediction_client):
    image_array = load_image(url)
    vals = image_array.numpy().tolist()
    
    instance = json_format.ParseDict(vals, Value())
    instances = [instance]
    
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    
    endpoint = prediction_client.endpoint_path(
        project=PROJECT_ID, location=LOCATION, endpoint=ENDPOINT_ID
    )
    response = prediction_client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )

    predictions = response.predictions
    predicted_class = np.argmax(predictions[0])
    p_index = predicted_class.astype(int)

    return p_index

def get_lyrics_for_artist(artist_index, prediction_client):
    endpoint_id = nl_models[artist_index]
    vals = artist_embeddings[class_names[artist_index]]
    
    instance = json_format.ParseDict(vals, Value())
    instances = [instance]

    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())

    endpoint = prediction_client.endpoint_path(
        project=PROJECT_ID, location=LOCATION, endpoint=endpoint_id
    )
    response = prediction_client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )

    predictions = response.predictions
    return str(predictions)

def write_lyrics(request):
    request_json = request.get_json()
    print(request_json)

    if request_json['url']:
        url = request_json['url']
        client_options = {"api_endpoint": API_ENDPOINT}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        prediction_class = get_image_prediction(url, client)

        # Update this 
        predicted_lyrics = get_lyrics_for_artist(0, client)

        response = make_response(
            jsonify({"predicted_lyrics": predicted_lyrics}), 200
            )

        response.headers["Content-Type"] = "application/json"
        return response
    else:
        response = make_response(
            jsonify({"error": "No URL provided."}), 500)
        response.headers["Content-Type"] = "application/json"
        return response
