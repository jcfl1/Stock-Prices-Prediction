link do vídeo de apresentação: [apresentação](https://drive.google.com/file/d/1w2KOnkOt4ZLxmvJPMLpXVa1jEnKywPOF/view?usp=sharing)


# Stock-Prices-Prediction

Professor Luciano Barbosa Projeto da disciplina: IF697 - Introdução Ciência de Dados
Alunos: Rodrigo Abreu e Julio Cesar

Previsão de subida ou descida do preço de ações baseados na serie temporal de seu histórico e em informações adicionais sobre as empresas como textos no twitter e demonstrativos das empresas. Utilizamos como ações do projeto PETR4 e VALE3.

O trabalho de coleta, pré-processamento e análises está separado nos diretórios: `Price` usando como fonte de dados os preços de ações e `Text` usando como fonte de dados tweets sobre ações. Visto que são fontes de dados signifcativamente distintas que exigem tratamentos diferentes, separamos os trabalhos realizados dessas duas fontes de dados.

## Coleta de dados, pré-processamento e análises focadas em preços de ações

Os trabalhos realizados a favor da coleta de dados, pré-processamento e análises para dados focados no preço das ações estão concentrados no diretório `Price` e o notebook `price_data_preparation.ipynb`. Além disso, o notebook `create_time_series.ipynb` é utilizado para unir dados relacionados à divulgações de balanços das ações com os dados históricos de preços.

## Coleta de dados, Demonstrativos e análise.

Na pasta `Demonstrativos` temos o script que faz o scrapping dos demonstrativos obtidos a partir de uma página HTML, o notebook que faz a limpeza e exploração dos dados.

## Coleta de dados, pré-processamento e análises focadas em tweets relacionados às ações

No contexto de predição de preço de ações, as análises técnicas e fundamentalista são abordagens distintas na predição do preço de ações. A primeira foca em padrões históricos de preço e volume para prever movimentos futuros do mercado. Já a análise fundamentalista avalia o valor intrínseco de uma ação, considerando fatores econômicos, financeiros e qualitativos, como balanços patrimoniais, lucros, gestão da empresa e condições macroeconômicas. Em prol da análise fundamentalista, pesquisas indicam que dados textuais respectivos à impressão do público sobre ações são relevantes para o desempenho de modelos de predição de preço de ações \[1,2\]. O uso de tweets permite captar o sentimento do mercado e as percepções dos investidores, enriquecendo a análise com informações atualizadas que podem afetar os preços das ações. Nesse contexto, incluímos **tweets** como insumo de informação para predição de preço de ações!

Os trabalhos realizados a favor da coleta de dados, pré-processamento e análises focadas em tweets que mencionam ações estão concentrados no diretório `Text` e o notebook `text_data_preparation.ipynb`. 

Os trabalhos relacionados a lei de Zipf estão no script `Zipf.py` e no notebook `text_data_preparation.ipynb`.

## União das informações em uma única série temporal 

Foi feita no notebook:  `aggregating_tweets_keywords_relations.ipynb`, 

## Cálculo Lucro

O script para fazer cálculo do lucro a partir do modelo está no arquivo cálculo lucro. Para executá-lo, deve-se ter um modelo com o método predict que diga a tendência de movimento da ação (1 para subir e 0 para cair de valor). A função recebe como parâmetro também as features que o modelo usará no formato que o modelo precisa (argumento: X) e também um dataframe correspondente que tenha uma coluna "Date" com as datas correspondentes aos dados em X. 

O cálculo é feito usando uma abordagem agressiva de venda e compra, vendendo ou comprando tudo que for possível de acordo com a previsão do modelo.

## Exibir gráficos de correlações de features com a variável alvo hasRise do dataset final pré-processado

investigate_correlations.ipynb


## Aprimorar nosso entendimento dos dados e realizar análises através de clustering, 
clustering.ipynb usa o algoritmo KMeans.


## Treinamento e Avaliação

O script `create_train_sets.py` fornece uma série de funções para fazer as últimas modificações nos datasets, como separar em treino e teste (a partir de uma determinada data) transforma os dados tabulares em sequências 

Para o xgboost, o regressor logístico e o autosklearn, os códigos de busca de hiperparâmetros,  treinamento avaliação, e melhoria dos modelos após diagnóstico da avaliação. Estão nos notebooks com seus nomes.
Os códigos para as redes recorrentes estão na pasta Treino:
    Os modelos estão em Models
    O código para busca de hiperparâmetros está no script com esse nome.
    Os scripts para busca, treino e avaliação do modelo estão no `treino_avaliacao.py`
    Os notebooks `treino.ipynb` e `treino_corrigido.ipynb` contém a execução (busca, treino e avaliação) dos modelos, o diagnóstico da avaliação e a correção no algoritmo de treino (mudança nas features da base, mudança nos modelos) para amenizar overfitting.


<br><br>

\[1] *Xu et al, "Stock Movement Prediction from Tweets and Historical Prices", Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, 2018* 

\[2] *G. Vargas et al, "B3 Stock Price Prediction Using LSTM Neural Networks and Sentiment Analysis," in IEEE Latin America Transactions, vol. 20, 2022* 
