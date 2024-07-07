import pandas as pd
import numpy as np
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