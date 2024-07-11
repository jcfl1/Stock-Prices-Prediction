import numpy as np
from sklearn.metrics import classification_report
from Models.LSTM.LSTM import LSTMModel
from Models.GRU.GRU import GRUModel
from busca_hiperparametros_NN import run_hiperparameter_search
import torch

import sys, os
sys.path.append('..')

RANDOM_SEED = 33
def treinar(X_train= None, y_train= None, ModelClass= LSTMModel, n_trials= 100, file_to_save_model= 'all_features.pt'):
    """busca de hiperparametro e depois treino e save"""
    study = run_hiperparameter_search(X_train, y_train, ModelClass= ModelClass, n_trials= n_trials)

    model = ModelClass(input_size= X_train.shape[-1],
                    hidden_size= study.best_params["hidden_size"], 
                    num_layers= study.best_params["num_layers"],
                    output_size= 1)

    model.fit(X_train, y_train,
            learning_rate= study.best_params['learning_rate'],
            )
    
    if file_to_save_model:
        path_to_file= None     
        if type(model) == type(LSTMModel(1,1,1,1)):
            path_to_file = os.path.join('Models','LSTM',file_to_save_model)
        elif type(model) == type(GRUModel(1,1,1,1)):
            path_to_file = os.path.join('Models','GRU',file_to_save_model)

        if path_to_file is None:
            print("do not know the model")
        else:
            print("saving model on:", path_to_file)
            torch.save(model, path_to_file)

    return model

def avaliar(model, X_train= None, y_train= None, X_test= None, y_test= None):
    if X_train is not None and y_train is not None:
        y_pred = model.predict(X_train)
        print(classification_report(y_train,y_pred))

    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        print(classification_report(y_test,y_pred))


def treinar_avaliar(ModelClass,stock, n_trials= 100,file_to_save_model= 'all_features.pt'):
    get_dataset_path = lambda  stock, get_labels,get_train: os.path.join("..","FinalDatasets", stock,f"{stock}_{'y' if get_labels else 'X'}_timeseries_{'train' if get_train else 'test'}.npy")

    X_train = np.load(get_dataset_path(stock= stock, get_labels= False, get_train= True))
    y_train = np.load(get_dataset_path(stock= stock, get_labels= True, get_train= True))
    X_test  = np.load(get_dataset_path(stock= stock, get_labels= False, get_train= False))
    y_test  = np.load(get_dataset_path(stock= stock, get_labels= True, get_train= False))

    model = treinar( X_train, y_train,ModelClass, n_trials, file_to_save_model)
    avaliar(model, X_train, y_train, X_test, y_test)



if __name__ == "__main__":
    treinar_avaliar(GRUModel,  "VALE3")