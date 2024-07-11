import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, classification_report
from Models.LSTM.LSTM import LSTMModel
import optuna
import torch

import sys, os
sys.path.append('..')

RANDOM_SEED = 33

def stop_when_metric_reached(study, trial, target_value):
    # Check if the best value in the study has reached the target value
    if study.best_value >= target_value:
        # Raise an Optuna specific exception to stop the study
        study.stop()

def run_hiperparameter_search(X_train,y_train, ModelClass= LSTMModel, n_trials= 100):

    # Objective function for Optuna
    def objective(trial):
        input_size = X_train.shape[-1]
        output_size = 1

        # Hyperparameter search space
        hidden_size = trial.suggest_int('hidden_size', 16, 128)
        num_layers = trial.suggest_int('num_layers', 1, 5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        # Model, loss function, and optimizer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ModelClass(input_size, hidden_size, num_layers, output_size, device) # ja aplica self.to(device)
        
        # Define the number of folds
        n_splits = 5

        # Create KFold object
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

        f1_scores_sum =  0

        for train_indices, val_indices in kf.split(X_train,y_train):
            X_train_folds = X_train[train_indices]
            y_train_folds = y_train[train_indices]

            X_val_folds = X_train[val_indices]
            y_val_folds = y_train[val_indices]

            model.fit(X_train_folds, y_train_folds, learning_rate)
            y_pred = model.predict(X_val_folds)
            f1_scores_sum += f1_score(y_val_folds, y_pred)
        
        # the metric is the mean of all f1 scores
        return f1_scores_sum / n_splits 
    

    # Running the optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials= n_trials,callbacks=[lambda study, trial: stop_when_metric_reached(study, trial, target_value=1.0)])

    # Print the results
    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study