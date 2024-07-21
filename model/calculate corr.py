import pandas as pd
import numpy as np

def calculate_correlation():
    # Read the CSV file
    data = pd.read_csv('dataset_5.csv')
    data = data.replace({True: 1, False: 0})

    data_x = data.iloc[:, 0:-1]
    data_y = data.iloc[:, -1]

    correlations = []
    column = data_x.shape[1]
    for i in range (column):
        a = data_x.iloc[:, i]
        std = a.std()
        if std == 0:
            corr = 0
        else:
            b = a.dropna()
            n = b.shape[0]
            if n <=2:
                corr = 0
            else:
                c = data_y.loc[b.index]
                corr = np.corrcoef(b, c)
                corr = corr[0,1]
                #absolute_corr = np.abs(corr)[0, 1]
        correlations.append(corr)
    print(correlations)

calculate_correlation()

