import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import pickle

def prepare_dataset(df):
    df = df.drop('stock', axis='columns')
    df = df.set_index('Date', drop=True)
    return df

def extract_target_columns(df):
    df_aux = df.copy()
    # subtrai as colunas de forma que o valor em 'close one day variation ==
    # o valor da proxima linha menos o valor da linha atual
    df_aux['Close one day variation'] = df_aux['Close'].shift(-1) - df_aux['Close']
    #a ultima linha nao tem proxima, por isso seu valor is NaN
    df_aux.dropna(inplace= True)

    # pega apenas o sinal da variacao, para ser nossa variavel alvo
    df_aux['hasRise'] = df_aux['Close one day variation'] > 0
  
    df_aux['hasRise'] = df_aux['hasRise'].map({True:1, False:0})
    

    # apagando colunas auxiliares
    df_aux.drop(columns=["Close one day variation"], inplace= True)

    return df_aux

def split_df(df, split_day='2023-12-01'):
    
    df_train = df[:split_day]
    df_test = df[split_day:]

    return df_train, df_test


def normalize_dfs(df_train, df_test):
    std_scaler = StandardScaler()
    df_train_aux = df_train.drop('hasRise', axis='columns').reset_index(drop=True)
    df_test_aux = df_test.drop('hasRise', axis='columns').reset_index(drop=True)

    std_scaler = std_scaler.fit(df_train_aux)
    train_norm = std_scaler.transform(df_train_aux)
    test_norm = std_scaler.transform(df_test_aux)

    df_train_norm = pd.DataFrame(train_norm, columns=df_train_aux.columns)
    df_test_norm  = pd.DataFrame(test_norm, columns=df_test_aux.columns)
    
    # recuperando a coluna de target e de date
    df_train_norm['hasRise'] = df_train['hasRise'].reset_index(drop=True)
    df_test_norm['hasRise']  = df_test['hasRise'].reset_index(drop=True)

    df_train_norm['Date'] = df_train.index
    df_train_norm = df_train_norm.set_index('Date', drop=True)
    df_test_norm['Date']  = df_test.index
    df_test_norm = df_test_norm.set_index('Date', drop=True)
    
    return df_train_norm, df_test_norm, std_scaler
# deveriamos salvar como csv os dados ao final desta funcao e nunca mais usa-la
# alem disso pode ser util salvar o std_scaler em um pickle
def get_dataset(PATH):
    df = pd.read_csv(PATH)
    df = prepare_dataset(df)
    df = extract_target_columns(df)
    df_train, df_test = split_df(df)
    df_train, df_test,std_scaler = normalize_dfs(df_train, df_test)
    return df_train, df_test, std_scaler

def get_sequences_X_y(df,window_size = 7):
    """window_size N significa que o modelo vai pegar os N dias consecutivos
      para prever se o preco vai subir do dia N para o dia N+1
      O hasRise do N° contem o target desses N dias """
    
    targets = df.iloc[window_size:]["hasRise"]
    seqs = df.drop('hasRise', axis='columns') # features

    sequences_list =[]
    for start in range(0, len(df) - window_size):
        end = start + window_size
        seq = seqs.iloc[start:end] # pego do primeiro dia da sequencia ate o window_size°
        sequences_list.append(seq)

    # X deve ter shape: (N° dias - window_size, window_size, N° colunas por dia)
    # y deve ter shape: (N° dias - window_size, N° colunas por dia)
    X, y = np.stack(sequences_list, axis=0), targets.to_numpy()
    return X, y

def prepare_and_save_tabular_and_sequence_datasets(path, stock_name):
    path_save = os.path.join('FinalDatasets', stock_name)
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    df_train, df_test, std_scaler = get_dataset(path)
    
    df_train.to_csv(f'{path_save}/{stock_name}_tabular_train.csv')
    df_test.to_csv(f'{path_save}/{stock_name}_tabular_test.csv')
    with open(f'{path_save}/{stock_name}_scaler.pkl', 'wb') as f:
        pickle.dump(std_scaler, f)

    X_test, y_test  = get_sequences_X_y(df_test)
    X_train, y_train  = get_sequences_X_y(df_train)

    np.save(f'{path_save}/{stock_name}_X_timeseries_train.npy', X_train)
    np.save(f'{path_save}/{stock_name}_y_timeseries_train.npy', y_train)
    
    np.save(f'{path_save}/{stock_name}_X_timeseries_test.npy', X_test)
    np.save(f'{path_save}/{stock_name}_y_timeseries_test.npy', y_test)
    return

if __name__ == "__main__":

    path_petr4 = os.path.join("TimeSeries","PETR4_prices_and_tweets_NLI_scores.csv")
    path_vale3 = os.path.join("TimeSeries","VALE3_prices_and_tweets_NLI_scores.csv")

    prepare_and_save_tabular_and_sequence_datasets(path_petr4, 'PETR4')
    prepare_and_save_tabular_and_sequence_datasets(path_vale3, 'VALE3')