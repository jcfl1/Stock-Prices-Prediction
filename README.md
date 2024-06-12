# Stock-Prices-Prediction

Professor Luciano Barbosa Projeto da disciplina: IF697 - Introdução Ciência de Dados

Previsão de subida ou descida do preço de ações baseados na serie temporal de seu histórico e em informações adicionais sobre as empresas como textos no twitter. Utilizamos como ações do projeto PETR4 e VALE3.

O trabalho de coleta, pré-processamento e análises está separado nos diretórios: `Price` usando como fonte de dados os preços de ações e `Text` usando como fonte de dados tweets sobre ações. Visto que são fontes de dados signifcativamente distintas que exigem tratamentos diferentes, separamos os trabalhos realizados dessas duas fontes de dados.

### Coleta de dados, pré-processamento e análises focadas em preços de ações

Os trabalhos realizados a favor da coleta de dados, pré-processamento e análises para dados focados no preço das ações estão concentrados no diretório `Price` e o notebook `price_data_preparation.ipynb`. Além disso, o notebook `create_time_series.ipynb` é utilizado para unir dados relacionados à divulgações de balanços das ações com os dados históricos de preços.

### Coleta de dados, pré-processamento e análises focadas em tweets relacionados às ações

No contexto de predição de preço de ações, as análises técnicas e fundamentalista são abordagens distintas na predição do preço de ações. A primeira foca em padrões históricos de preço e volume para prever movimentos futuros do mercado. Já a análise fundamentalista avalia o valor intrínseco de uma ação, considerando fatores econômicos, financeiros e qualitativos, como balanços patrimoniais, lucros, gestão da empresa e condições macroeconômicas. Em prol da análise fundamentalista, pesquisas indicam que dados textuais respectivos à impressão do público sobre ações são relevantes para o desempenho de modelos de predição de preço de ações \[1,2\]. O uso de tweets permite captar o sentimento do mercado e as percepções dos investidores, enriquecendo a análise com informações atualizadas que podem afetar os preços das ações. Nesse contexto, incluímos **tweets** como insumo de informação para predição de preço de ações!

Os trabalhos realizados a favor da coleta de dados, pré-processamento e análises focadas em tweets que mencionam ações estão concentrados no diretório `Text` e o notebook `text_data_preparation.ipynb`. 


<br><br>

\[1] *Xu et al, "Stock Movement Prediction from Tweets and Historical Prices", Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, 2018* 

\[2] *G. Vargas et al, "B3 Stock Price Prediction Using LSTM Neural Networks and Sentiment Analysis," in IEEE Latin America Transactions, vol. 20, 2022* 