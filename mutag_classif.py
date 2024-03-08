import json 
import numpy as np 
import torch
from tqdm import tqdm 
import networkx as nx
import pandas as pd 

from fugw_solver.solver import solver
import ot

from collections import defaultdict

import multiprocessing

def adjacency_to_distance_matrix(adj):
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    length = dict(nx.all_pairs_shortest_path_length(G))
    num_nodes = adj.shape[0]
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            try:
                distance_matrix[i, j] = length[i][j]
            except KeyError:  # no path between i and j
                distance_matrix[i, j] = np.inf  # or some large number to denote no path
    return distance_matrix

def predict(args):
    """Predicts the class of the i-th molecule using a k-NN"""
    i, k, rho, alpha, features, distance_matrices, y, pot = args
    source_features = torch.tensor(features[i]).T.float()
    source_geometry = torch.tensor(distance_matrices[i]).float()
    distances = []

    for j, (target_features, target_geometry, classif) in enumerate(zip(features, distance_matrices, y)):
        if j == i:
            continue
        target_features = torch.tensor(target_features).T.float()
        target_geometry = torch.tensor(target_geometry).float()
    
        if pot:
            C1 = ot.dist(source_geometry)
            C2 = ot.dist(target_geometry)
            M = ot.dist(source_features.T, target_features.T)
            Gwg, logw = ot.fused_gromov_wasserstein(M, C1, C2, log=True, alpha=alpha)
            loss = logw["fgw_dist"]
        else:
            P, loss_ls = solver(source_features, target_features, source_geometry, target_geometry, rho=rho, alpha=alpha, nits_bcd=10, nits_uot=100)
            loss = loss_ls[-1]["total"]
        distances.append((loss, classif))

    distances.sort(key=lambda x: x[0])
    distances = distances[:k]
    classes = list(map(lambda x: x[1], distances))
    return 1 if sum(classes) >= (len(classes)/2) else 0


# def knn(rho):
#     accuracy = 0
#     rho = 1
#     print(f"Starting the knn for rho={rho} and k={K}")
#     with tqdm(range(n)) as pbar:
#         for i in pbar:
#             prediction = predict(i, rho)
#             accuracy += (prediction == y[i])
#             pbar.set_postfix(accuracy=100. * accuracy / (i+1))
#     accuracy /=W n
#     print(f"The accuracy for the MUTAG dataset with rho={rho} and k={K} is {accuracy}")
#     return accuracy


# Multiprocessing
# Compute the accuracy with Leave One Out cross validation
def knn(rho, alpha, k, features, distance_matrices, y, pot=False):
    accuracy = 0
    n = len(features)
    # print(f"Starting the knn for rho={rho}, alpha={alpha}, k={k}")
    args = [(i, k, rho, alpha, features, distance_matrices, y, pot) for i in range(n)]  # Prepare arguments for each process

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        predictions = list(tqdm(pool.imap(predict, args), total=n))

    # Calculate accuracy based on the returned predictions
    for i, prediction in enumerate(predictions):
        accuracy += (prediction == y[i])
    accuracy /= n
    print(f"The accuracy with rho={rho}, alpha={alpha}, k={k}{', with pot' if pot else ''}, is {accuracy}")
    return accuracy

if __name__ == '__main__':
    with open("dataset/mutag.json", 'r') as f:
        data = json.load(f)

    CSV_NAME = "run_loo_3.csv"
    POT = False

    features = data["node_feat"]
    y = data["y"]
    y = list(map(lambda x: x[0], y))
    distance_matrices = []
    for (sources, targets), num_nodes in zip(data["edge_index"], data["num_nodes"]):
        adj = np.zeros((num_nodes, num_nodes))
        adj[sources, targets] = 1
        distance_matrix = adjacency_to_distance_matrix(adj)
        distance_matrices.append(distance_matrix)

    results = defaultdict(list)
    for rho in [0.1, 1, 10, 100, 1000]:
        for alpha in [0.1, 0.9]:
            for k in [5]:
                train_accuracy = knn(rho, alpha, k, features, distance_matrices, y, POT)
                results["alpha"].append(alpha)
                results["rho"].append(rho)
                results["k"].append(k)
                results["accuracy"].append(train_accuracy)
                
    pd.DataFrame(results).to_csv(f"results/{CSV_NAME}", index=False)


    # TRAIN_SIZE = 100
    # features_train, features_test = features[:TRAIN_SIZE], features[TRAIN_SIZE:]
    # y_train, y_test = y[:TRAIN_SIZE], y[TRAIN_SIZE:]
    # distance_matrices_train, distance_matrices_test = distance_matrices[:TRAIN_SIZE], distance_matrices[TRAIN_SIZE:]
    #
    # print("Finding the best hyperparameters ...")
    # best_accuracy = 0
    # best_params = None
    # results = defaultdict(list)
    # for rho in [0.1, 1, 10, 100, 1000]:
    #     for alpha in [0.4, 0.5, 0.6]:
    #         for k in [5, 10, 15]:
    #             train_accuracy = knn(rho, alpha, k, features_train, distance_matrices_train, y_train, pot)
    #             results["alpha"].append(alpha)
    #             results["rho"].append(rho)
    #             results["k"].append(k)
    #             results["accuracy"].append(train_accuracy)
    #             pd.DataFrame(results).to_csv(f"results/{csv_name}", index=False)
    #             if train_accuracy > best_accuracy:
    #                 best_accuracy = train_accuracy
    #                 best_params = (rho, alpha, k)

    # print(f"The best parameters are rho={best_params[0]}, alpha={best_params[1]}, k={best_params[2]}")
    
    # print("Testing...")
    # test_accuracy = knn(*best_params, features_test, distance_matrices_test, y_test, pot)
    # print(f"The test accuracy is {test_accuracy}")
