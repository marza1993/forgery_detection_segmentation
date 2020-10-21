from keras import Model
from keras.models import model_from_json

# TODO: ragionare se spostare nella classe segmentation model

def save_model(weights_path):
    model_json = model.to_json()
    with open(model_path, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(weights_path)



def load_model(model_path, weights_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)

    return loaded_model
