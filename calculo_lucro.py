import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()
import pandas as pd
from datetime import datetime


def calculo_lucro(model, df, X, acao= "PETR4", capital_inicial = 100):

    start_date = df["Date"]
    preco_inicial = pdr.get_data_yahoo(acao, start=start_date, end=start_date)["Open"]
    df["Close"] = df["Close"]*preco_inicial
    
    # Suponha que seu dataset tem as seguintes colunas:
    # 'Date': Data da observação
    # 'Prediction': Previsão do modelo (1 para alta, 0 para baixa)

    y_pred = model.predict(X)

    df["Prediction"] = y_pred

    # Inicializando variáveis
   # Capital inicial para investimento
    capital = capital_inicial
    posicao = 0  # Quantidade de ações que possuímos

    # Iterando sobre cada linha do dataset
    for i in range(1, len(df)):
        if df.loc[i, 'Prediction'] == 1:  # Previsão de alta
            # Comprar ações se tivermos capital
            if capital >= df.loc[i, 'Close']:
                acoes_compradas = capital // df.loc[i, 'Close']
                capital -= acoes_compradas * df.loc[i, 'Close']
                posicao += acoes_compradas
        elif df.loc[i, 'Prediction'] == 0:  # Previsão de baixa
            # Vender todas as ações se tivermos alguma
            if posicao > 0:
                capital += posicao * df.loc[i, 'Close']
                posicao = 0

    # Valor final considerando o valor das ações restantes
    valor_final = capital + posicao * df.loc[len(df) - 1, 'Close']

    # Calculando o retorno
    retorno = (valor_final - capital_inicial) / capital_inicial * 100
    print(f"Retorno: {retorno:.2f}%")

if __name__ == "__main__":
    from Treino.Models.LSTM.LSTM import LSTMModel
    import torch
    import os
    import numpy as np

    #model = LSTMModel(hidden_size= 104, num_layers= 1)
    for acao in ["PETR4", "VALE3"]:

        model = torch.load(os.path.join("Treino","Models","LSTM",f"{acao}_all_features.pt"))
        model.eval()

        df = pd.read_csv(os.path.join("FinalDatasets",acao,f"{acao}_tabular_test.csv"))
        X = np.load(os.path.join("FinalDatasets",acao,f"{acao}_X_timeseries_test.npy"))
        calculo_lucro(model, df, X, acao)