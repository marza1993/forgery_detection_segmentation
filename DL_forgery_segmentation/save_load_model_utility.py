from keras import Model
from keras.models import model_from_json

# TODO: ragionare se spostare nella classe segmentation model

def save_model(model, model_path, weights_path):

    # salvo il json
    model_json = model.to_json()
    with open(model_path, 'w') as json_file:
        json_file.write(model_json)
    # salvo i pesi
    model.save_weights(weights_path)



def load_model(model_path, weights_path):

    # leggo ill json
    loaded_model_json = None
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    # carico la struttura del modello dal json
    loaded_model = model_from_json(loaded_model_json)
    # carico i pesi
    loaded_model.load_weights(weights_path)

    return loaded_model
