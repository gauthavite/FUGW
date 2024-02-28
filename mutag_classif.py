import json 
import numpy as np 
import torch
from tqdm import tqdm 

from fugw_solver.solver import solver

import multiprocessing


def predict(args):
    """Predicts the class of the i-th molecule using a k-NN"""
    i, k, rho, alpha, features, adjacencies, y = args
    source_features = torch.tensor(features[i]).T.float()
    source_geometry = torch.tensor(adjacencies[i]).float()
    distances = []
    for j, (target_features, target_geometry, classif) in enumerate(zip(features, adjacencies, y)):
        if j == i:
            continue
        target_features = torch.tensor(target_features).T.float()
        target_geometry = torch.tensor(target_geometry).float()
        P, loss_ls = solver(source_features, target_features, source_geometry, target_geometry, rho=rho, alpha=alpha, nits_bcd=10, nits_uot=100)
        distances.append((loss_ls[-1]["total"], classif))

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
def knn(rho, alpha, k, features, adjacencies, y):
    accuracy = 0
    print(f"Starting the knn for rho={rho}, alpha={alpha}, k={k}")
    args = [(i, k, rho, alpha, features, adjacencies, y) for i in range(n)]  # Prepare arguments for each process

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        predictions = list(tqdm(pool.imap(predict, args), total=n))

    # Calculate accuracy based on the returned predictions
    for i, prediction in enumerate(predictions):
        accuracy += (prediction == y[i])
    accuracy /= n
    print(f"The accuracy with rho={rho}, alpha={alpha}, k={k} is {accuracy}")
    return accuracy

if __name__ == '__main__':
    with open("dataset/mutag.json", 'r') as f:
        data = json.load(f)

    features = data["node_feat"]
    n = len(features)
    y = data["y"]
    y = list(map(lambda x: x[0], y))
    adjacencies = []
    for (sources, targets), num_nodes in zip(data["edge_index"], data["num_nodes"]):
        adj = np.zeros((num_nodes, num_nodes))
        adj[sources, targets] = 1
        adjacencies.append(adj)

    features_train, features_test = features[:150], features[150:]
    y_train, y_test = y[:150], y[150:]
    adjacencies_train, adjacencies_test = adjacencies[:150], adjacencies[150:]
    
    print("Finding the best hyperparameters ...")
    best_accuracy = float("inf")
    best_params = None
    for rho in [0.1, 1, 10]:
        for alpha in [0.4, 0.5, 0.6]:
            for k in [5, 10, 15]:
                train_accuracy = knn(rho, alpha, k, features_train, adjacencies_train, y_train)
                best_params = (rho, alpha, k)
    print(f"The best parameters are rho={rho}, alpha={alpha}, k={k}")
    
    print("Testing...")
    test_accuracy = knn(100, features, adjacencies, y)
    print(f"The test accuracy is {test_accuracy}")