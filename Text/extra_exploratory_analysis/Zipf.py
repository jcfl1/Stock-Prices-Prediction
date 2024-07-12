import numpy as np
from scipy.stats import linregress
from collections import defaultdict
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

def ngram_frequency(text, n):
    words = text.split()
    freq_dict = defaultdict(int)
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        freq_dict[ngram] += 1
    return freq_dict

def zipf_regression(freq_dict, min_freq= 3, titulo= "Zipf's Law", show= False):
    # Ordenar as frequências em ordem decrescente
    sorted_freqs = np.array(sorted(freq_dict.values(), reverse=True))
    # apenas os que aparecem mais de min_freq
    # e exclui os 10 primeiros
    sorted_freqs = sorted_freqs[sorted_freqs > min_freq][10:]
    
    # Plotar em escala log-log
    ranks = np.arange(1, len(sorted_freqs) + 1)
    
    plt.loglog(ranks, sorted_freqs, marker="o", linestyle="none", label= "frequêncial real")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title(titulo)

    # Ajustar uma linha reta aos dados log-log
    log_ranks = np.log(ranks)
    log_freqs = np.log(sorted_freqs)
    slope, intercept, r_value, p_value, std_err = linregress(log_ranks, log_freqs)
    plt.plot(ranks, np.exp(intercept + slope * log_ranks), color="red", label=f"Slope = {slope:.2f}")
    plt.legend()
    if show:
        plt.show()

    # Retornar o coeficiente de correlação 
    # e p_value para teste de hipotese

    """The p-value for a hypothesis test whose null hypothesis is that the slope is zero,
    using Wald Test with t-distribution of the test statistic. 
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html)"""
    return r_value ** 2, p_value


if __name__ == '__main__':
    
    text = "this is a test this is only a test"
    n = 2
    freq_dict = ngram_frequency(text, n)
    r_squared,p_value = zipf_regression(freq_dict)
    print(f"R^2: {r_squared:.4f}")

