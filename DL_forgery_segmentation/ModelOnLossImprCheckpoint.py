import keras
import save_load_model_utility


WEIGHTS_FILE_NAME = "weights.hdf5"
MODEL_FILE_NAME = "model.hdf5"
LOSS_FILE_NAME = "best_loss.txt"


class ModelOnLossImprCheckpoint(keras.callbacks.Callback):


    def __init__(self, model_base_path):
        super().__init__()

        # imposto i percorsi in cui salvare i pesi
        self.model_base_path = model_base_path
        self.weights_path = model_base_path + WEIGHTS_FILE_NAME
        self.loss_file_path = model_base_path + LOSS_FILE_NAME
        self.model_path = model_base_path + MODEL_FILE_NAME


    def on_epoch_end(self, epoch, logs=None):
        
        # leggo l'ultima best validation loss salvata
        old_val_loss = None
        with open(self.loss_file_path,'r') as file_loss:
            riga = file_loss.readline()
            old_val_loss = float(riga)

        # se il valore corrente della validation loss è migliore di quello precedente allora salvo il modello
        # e sovrascrivo la loss nel file
        new_val_loss = logs['val_loss']
        if new_val_loss < old_val_loss:

            # salvo il modello
            save_load_model_utility.save_model(self.model, self.model_path, self.weights_path)

            print('\nEpoca %05d: val_loss migliorata da %0.5f a %0.5f, salvo il modello in %s' 
                  % (epoch + 1, old_val_loss, new_val_loss, self.model_path))

            # sovrascrivo il file con il nuovo valore
            with open(self.loss_file_path, 'w') as file_loss:
                file_loss.write(str(new_val_loss))


        else:
            print('\nEpoca %05d: val_loss non è migliorata da %0.5f' 
                  % (epoch + 1, old_val_loss))
