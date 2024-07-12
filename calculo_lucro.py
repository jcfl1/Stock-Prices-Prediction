import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()
import pandas as pd
from datetime import datetime
from datetime import timedelta


def calculo_lucro(model, df, X, acao= "PETR4", capital_inicial = 10_000):

    datas = pd.to_datetime(df["Date"])
    start_date = datas.iloc[0]
    end_date = datas.iloc[-1]
    df = pdr.get_data_yahoo(acao+'.SA', start=start_date, end=end_date + timedelta(days=1))

    
    # Suponha que seu dataset tem as seguintes colunas:
    # 'Date': Data da observação
    # 'Prediction': Previsão do modelo (1 para alta, 0 para baixa)

    y_pred = model.predict(X)

    df.drop(index=df.index[:7], inplace= True)

    df["Prediction"] = y_pred

    # Inicializando variáveis
   # Capital inicial para investimento
    capital = capital_inicial
    posicao = 0  # Quantidade de ações que possuímos

    # Iterando sobre cada linha do dataset
    for i in range(0, len(df)):
        if df['Prediction'].iloc[i] == 1:  # Previsão de alta
            # Comprar ações se tivermos capital
            if capital >= df['Close'].iloc[i]:
                acoes_compradas = np.floor(capital) // np.ceil(df["Close"].iloc[i])
                capital -= acoes_compradas * df["Close"].iloc[i]
                posicao += acoes_compradas
        elif df['Prediction'].iloc[i] == 0:  # Previsão de baixa
            # Vender todas as ações se tivermos alguma
            if posicao > 0:
                capital += posicao * df["Close"].iloc[i]
                posicao = 0
        valor_hoje = capital + posicao * df['Close'].iloc[i]
        print(valor_hoje,capital, posicao)

    # Valor final considerando o valor das ações restantes
    valor_final = capital + posicao * df['Close'].iloc[len(df) - 1]

    # Calculando o retorno
    retorno = (valor_final - capital_inicial) / capital_inicial * 100
    print(f"Retorno: {retorno:.2f}%")

if __name__ == "__main__":
    from Treino.Models.LSTM.LSTM import LSTMModel
    import torch
    import os
    import numpy as np

    get_dataset_path = lambda  stock, get_labels, get_train: os.path.join("FinalDatasets", stock,f"{stock}_{'y' if get_labels else 'X'}_timeseries_{'train' if get_train else 'test'}.npy")

    X_train = np.load(get_dataset_path(stock= "VALE3", get_labels= False, get_train= True))
    y_train = np.load(get_dataset_path(stock= "VALE3", get_labels= True, get_train= True))
    X_test = np.load(get_dataset_path(stock= "VALE3", get_labels= False, get_train= False))

    df_test = pd.read_csv(os.path.join("FinalDatasets","VALE3","VALE3_tabular_test.csv"))
  
    model = LSTMModel(input_size=X_train.shape[-1], hidden_size= 104, num_layers= 1, output_size= 1)
    model.fit(X_train, y_train, learning_rate= 0.0029308294553194235)
    model.eval()
    calculo_lucro(model, df_test,X_test,"VALE3")
    # for acao in ["PETR4", "VALE3"]:

    #     model = torch.load(os.path.join("Treino","Models","LSTM",f"{acao}_all_features.pt"))
    #     model.eval()

    #     X = np.load(os.path.join("FinalDatasets",acao,f"{acao}_X_timeseries_test.npy"))
    #     calculo_lucro(model, df, X, acao)