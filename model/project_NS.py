
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl



# Experiment 1, Plot a figure for leaf size and rmse for assessing overfitting
def experiment_1(train_x, train_y, test_x, test_y):
    rmse_in_sample_fig1 = np.array([])
    rmse_out_sample_fig1 = np.array([])
    for leaf_size in range(1, 50):
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)  # constructor
        learner.add_evidence(train_x, train_y)  # training step
        Ypred_in_fig1 = learner.query(train_x)
        Ypred_out_fig1 = learner.query(test_x)  # query
        rmse_in_fig1 = math.sqrt(((train_y - Ypred_in_fig1) ** 2).sum() / train_y.shape[0])
        rmse_out_fig1 = math.sqrt(((test_y - Ypred_out_fig1) ** 2).sum() / test_y.shape[0])
        rmse_in_sample_fig1 = np.append(rmse_in_sample_fig1, rmse_in_fig1)
        rmse_out_sample_fig1 = np.append(rmse_out_sample_fig1, rmse_out_fig1)

    # plot figure 1
    plt.plot(range(1, 50), rmse_in_sample_fig1, label="in sample")
    plt.plot(range(1, 50), rmse_out_sample_fig1, label="out sample")
    plt.xlabel('leaf size')
    plt.ylabel("rmse")
    plt.title("Figure1. leaf size and rmse")
    plt.grid(linestyle='-')
    plt.legend()
    plt.savefig("figure 1")
    plt.clf()


# Experiment 2 research the use of bagging and its effect on overfitting
def experiment_2(train_x, train_y, test_x, test_y):
    rmse_in_sample_fig2 = np.array([])
    rmse_out_sample_fig2 = np.array([])
    for leaf_size in range(1, 100):
        learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leaf_size}, bags=10, boost = False, verbose = False)  # constructor
        learner.add_evidence(train_x, train_y)  # training step
        Ypred_in_fig2 = learner.query(train_x)
        Ypred_out_fig2 = learner.query(test_x)  # query
        rmse_in_fig2 = math.sqrt(((train_y - Ypred_in_fig2) ** 2).sum() / train_y.shape[0])
        rmse_out_fig2 = math.sqrt(((test_y - Ypred_out_fig2) ** 2).sum() / test_y.shape[0])
        rmse_in_sample_fig2 = np.append(rmse_in_sample_fig2, rmse_in_fig2)
        rmse_out_sample_fig2 = np.append(rmse_out_sample_fig2, rmse_out_fig2)


    # plot figure 2
    plt.plot(range(1, 100), rmse_in_sample_fig2, label="in sample")
    plt.plot(range(1, 100), rmse_out_sample_fig2, label="out sample")
    plt.xlabel('leaf size')
    plt.ylabel("rmse")
    plt.title("Figure 2. leaf size and rmse with 10 bags")
    plt.grid(linestyle='-')
    plt.legend()
    plt.savefig("figure 2")
    plt.clf()


  		  	   		  		 			  		 			     			  	 
if __name__ == "__main__":

    data = pd.read_csv('dataset_5.csv')
    data = data.replace({True: 1, False: 0})
    data = data.to_numpy()
    data = data.astype('float')
    np.random.shuffle(data)

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]


    experiment_1(train_x, train_y, test_x, test_y)
    experiment_2(train_x, train_y, test_x, test_y)






