from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def prepare_dataset(df):
    df = df.drop('stock', axis='columns')
    df = df.set_index('Date', drop=True)
    return df

def extract_target_columns(df):
    df_aux = df.copy()
    close_price_per_day_1_day_shifted = df_aux['Close'][1:]
    close_price_per_day_1_day_shifted = pd.concat([close_price_per_day_1_day_shifted, pd.Series(np.nan)], ignore_index=True)
    # Incluindo o valor do fechamento do dia seguinte no dataframe auxiliar
    df_aux['Close for next day'] = close_price_per_day_1_day_shifted
    # Removendo o útlimo dia de coleta de dados por não sabermos o preço de fechamento do dia seguinte
    df_aux = df_aux[:-1]

    df_aux['Close one day variation'] = df_aux['Close for next day'] - df_aux['Close']
    df_aux['hasRise'] = df_aux['Close one day variation'] > 0
    df_aux['hasRise'] = df_aux['hasRise'].map({True:1, False:0})

    return df_aux

def split_df(df, split_day='2023-12-01'):
    print(df.head())
    df_train = df[:split_day]
    df_test = df[split_day:]
    print(f'train.shape: {df_train.shape}, test.shape:{df_test.shape}')

    return df_train, df_test


def normalize_and_separate_X_and_y(df_train, df_test):
    # Separando coluna target antes da normalização
    hasRise_train = df_train['hasRise']
    hasRise_test = df_test['hasRise']

    std_scaler = StandardScaler()
    std_scaler = std_scaler.fit(df_train.drop('hasRise', axis='columns'))
    train_norm = std_scaler.transform(df_train.drop('hasRise', axis='columns'))
    test_norm = std_scaler.transform(df_test.drop('hasRise', axis='columns'))

    df_train_norm = pd.DataFrame(train_norm, columns=df_train.drop('hasRise', axis='columns').columns)
    df_test_norm = pd.DataFrame(test_norm, columns=df_test.drop('hasRise', axis='columns').columns)
    
    # Separando colunas de target após normalização
    close_train_norm = df_train_norm['Close for next day'] # train
    closeVariation_train_norm = df_train_norm['Close one day variation']

    close_test_norm = df_test_norm['Close for next day'] # test
    closeVariation_test_norm = df_test_norm['Close one day variation']
    
    X_train = df_train_norm.drop(columns=["Close for next day", "Close one day variation"])
    X_test  = df_test_norm.drop(columns= ["Close for next day", "Close one day variation"])


    return df_train_norm, close_train_norm, closeVariation_train_norm, hasRise_train, \
            df_test_norm, close_test_norm, closeVariation_test_norm, hasRise_test


def get_dataset(PATH):
    df = pd.read_csv(PATH)
    df = prepare_dataset(df)
    df = extract_target_columns(df)
    df_train, df_test = split_df(df)
    return normalize_and_separate_X_and_y(df_train, df_test)

def get_sequences(df,window_size = 7,test_size = 1):
    
    batchs_list = []
    for start in range(0, len(df) - window_size - test_size + 1, test_size):
        train = df.iloc[start:start + window_size]
        test = df.iloc[start + window_size:start + window_size + test_size]
        batchs_list.append((train,test))


if __name__ == "__main__":
    get_dataset("TimeSeries\PETR4_time_series.csv")