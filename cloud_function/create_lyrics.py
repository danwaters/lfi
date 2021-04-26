import base64
from flask import Flask
from flask import make_response
from flask import jsonify
import requests

from typing import Dict

from google.cloud import aiplatform
from google.cloud import storage
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from io import BytesIO
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.keras import preprocessing

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

PROJECT_ID = "dogbot-298321"
ENDPOINT_ID = "3456600674937077760" # Entry point to the pipeline (classification)
LOCATION = "us-central1"
API_ENDPOINT = "us-central1-aiplatform.googleapis.com"

class_names = ['beatles', 'bob-dylan', 'michael-jackson', 'zappa']
nl_models = ['3632804010357948416', '5286750973509763072', '3122771352558239744', '4656247025677893632']

NUM_WORDS = 3
tokenizer = Tokenizer()
# TODO: I know this is brittle but time is of the essence
# Already tokenized on the word "I"
# These are the seed values we will pass directly to the NL model
# Secret artists are here
# TODO: This hack is actually no longer needed
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

def setup_sequences_for(artist_index):
    artist_name = class_names[artist_index]
    source_blob_name = f'train/{artist_name}.txt'

    storage_client = storage.Client()
    bucket = storage_client.bucket('dw-music-models')
    blob = bucket.blob(source_blob_name)

    text = blob.download_as_string().decode('utf-8')
    sentences = text.lower().replace("\r\n", "\n").split("\n")

    # fit the tokenizer
    tokenizer.fit_on_texts(sentences)

    # Create input sequences, using list of tokens
    input_sequences = []
    for sentence in sentences:
        word_list = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(word_list)):
            n_gram_sequence = word_list[:i+1] # include next
            input_sequences.append(n_gram_sequence)

    max_len = max([len(x) for x in input_sequences]) 
    return max_len

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

def get_word_prediction(tokens, prediction_client, artist_index):
    endpoint_id = nl_models[artist_index]
    
    print(f'Shape: {tokens.shape}')
    tokens = tokens[0].tolist()
    print(f'Tokens: {tokens}')

    instance = json_format.ParseDict(tokens, Value())
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
    return predictions

def generate_text(client, artist_index, max_seq_length, seed_text="she", next_words=100, sequence_word_length=6):
    """ This method generates next_words words based on the seed text by
    repeatedly feeding the last sequence_length words into the LSTM to make the
    prediction. It keeps track of every word generated and prints the result."""
    # Seed and run predictions
    total_text = seed_text
    prediction_client = client
    for i in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], 
                                   maxlen=max_seq_length - 1,
                                   padding='pre')

        print(f'token_list: {token_list}')

        # Get the prediction
        predictions = get_word_prediction(token_list, prediction_client, artist_index)
        predicted = np.argmax(predictions, axis=-1)

        # Add the actual word
        output_word = ""
        for word, index in tokenizer.word_index.items():
          if index == predicted:
            output_word = word
            break
        total_text += " " + output_word
        seed_text = seed_text + " " + output_word

        # if seed_text is n words or more, drop the first word in the sequence.
        seed_words = seed_text.split(' ')
        if len(seed_words) >= sequence_word_length:
          seed_text = ' '.join(seed_words[1:])

        out = ""
        for i, w in enumerate(total_text.split(' ')):
            out = out + " " + w
            if i % sequence_word_length == 0 and i > 0: # insert line breaks every 5 words
                out += "\r\n"

    print(out)            
    return out

def get_lyrics_for_artist(artist_index, prediction_client, seed_text):
    max_len = setup_sequences_for(artist_index)
    return generate_text(prediction_client, artist_index, max_len, seed_text=seed_text)
    

def write_lyrics(request):
    request_json = request.get_json()
    print(request_json)

    if request_json['url']:
        url = request_json['url']

        # Make it possible to seed the first words through the API
        seed = 'I'
        if request_json['seed']:
            seed = request_json['seed']

        client_options = {"api_endpoint": API_ENDPOINT}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        prediction_class = get_image_prediction(url, client)

        # Update this 
        predicted_lyrics = get_lyrics_for_artist(prediction_class, client, seed)

        response = make_response(
            jsonify({"predicted_lyrics": predicted_lyrics, "image_url": url, "predicted_class": class_names[prediction_class]}), 200
            )

        response.headers["Content-Type"] = "application/json"
        return response
    else:
        response = make_response(
            jsonify({"error": "No URL provided."}), 500)
        response.headers["Content-Type"] = "application/json"
        return response
