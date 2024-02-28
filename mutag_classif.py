import json 
import numpy as np 
import torch
from tqdm import tqdm 

from fugw_solver.solver import solver

with open("dataset/mutag.json", 'r') as f:
    data = json.load(f)

### Hyperparameter
k = 10

features = data["node_feat"]
n = len(features)
y = data["y"]
y = list(map(lambda x: x[0], y))
adjacencies = []
for (sources, targets), num_nodes in zip(data["edge_index"], data["num_nodes"]):
    adj = np.zeros((num_nodes, num_nodes))
    adj[sources, targets] = 1
    adjacencies.append(adj)


def predict(i : int, k:int, rho):
    """Predicts the class of the i-th molecule using a k-NN"""
    source_features = torch.tensor(features[i]).T.float()
    source_geometry = torch.tensor(adjacencies[i]).float()
    distances = []
    for j, (target_features, target_geometry, classif) in enumerate(zip(features, adjacencies, y)):
        if j == i:
            continue
        target_features = torch.tensor(target_features).T.float()
        target_geometry = torch.tensor(target_geometry).float()
        P, loss_ls = solver(source_features, target_features, source_geometry, target_geometry, rho=rho)
        distances.append((loss_ls[-1]["total"], classif))

    distances.sort(key=lambda x: x[0])
    distances = distances[:k]
    classes = list(map(lambda x: x[1], distances))
    return 1 if sum(classes) >= (len(classes)/2) else 0

def knn(rho):
    accuracy = 0
    rho = 1
    print(f"Starting the knn for rho={rho} and k={k}")
    with tqdm(range(n)) as pbar:
        for i in pbar:
            prediction = predict(i, k, rho)
            accuracy += (prediction == y[i])
            pbar.set_postfix(accuracy=100. * accuracy / (i+1))
    accuracy /= n
    print(f"The accuracy for the MUTAG dataset with rho={rho} and k={k} is {accuracy}")
    return accuracy

knn(1)
knn(100)