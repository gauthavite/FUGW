import matplotlib.pyplot as plt
import numpy as np
import torch
from fugw_solver.solver import solver
from collections import defaultdict

def create_3d_sphere(r):
    phi = np.linspace(0, 2*np.pi, 20, endpoint=False)
    theta = np.linspace(0, np.pi, 10)
    phi_red, phi_blue = [], []
    for angle in phi:
        if (angle >= 0 and angle < np.pi):
            phi_red.append(angle)
        else:
            phi_blue.append(angle)

    r_xy = r * np.sin(theta)
    theta_red, phi_red = np.meshgrid(theta, phi_red)
    theta_blue, phi_blue = np.meshgrid(theta, phi_blue)

    x_red, y_red, z_red = np.cos(phi_red) * r_xy, np.sin(phi_red) * r_xy, r * np.cos(theta_red)
    x_blue, y_blue, z_blue = np.cos(phi_blue) * r_xy, np.sin(phi_blue) * r_xy, r * np.cos(theta_blue)
    
    return x_red, y_red, z_red, x_blue, y_blue, z_blue

def create_2d_sphere(r):
    theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    theta_red, theta_blue = [], []
    for angle in theta:
        if (angle >= 0 and angle < np.pi):
            theta_red.append(angle)
        else:
            theta_blue.append(angle)
    x_red, x_blue = r * np.cos(theta_red), r * np.cos(theta_blue)
    y_red, y_blue = r * np.sin(theta_red), r * np.sin(theta_blue)
    
    return x_red, y_red, x_blue, y_blue

def create_3d_circle(r):
    theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    theta_red, theta_blue = [], []
    for angle in theta:
        if (angle >= 0 and angle < np.pi):
            theta_red.append(angle)
        else:
            theta_blue.append(angle)
    x_red, x_blue = r * np.cos(theta_red), r * np.cos(theta_blue)
    y_red, y_blue = r * np.sin(theta_red), r * np.sin(theta_blue)

    z_red = y_red**2
    z_blue = y_blue**2

    print(len(x_red))

    for i in range(11, 14):
        point = x_blue[i], y_blue[i], z_blue[i]
        new_point = point + 0.1 * np.random.randn(3)
        x_red = np.append(x_red, new_point[0])
        y_red = np.append(y_red, new_point[1])
        z_red = np.append(z_red, new_point[2])

    print(len(x_red))
    
    return x_red, y_red, z_red, x_blue, y_blue, z_blue

# Function to plot 3D sphere with red crosses and blue dots
def plot_3d_sphere():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_red, y_red, z_red, x_blue, y_blue, z_blue = create_3d_sphere(1)

    ax.scatter(x_red, y_red, z_red,
               c='red', marker='x')

    ax.scatter(x_blue, y_blue, z_blue,
               c='blue', marker='o')
    ax.set_box_aspect([1, 1, 1])

    plt.title('3D Sphere with Red Crosses and Blue Dots')
    plt.show()

def plot_3d_circle():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_red, y_red, z_red, x_blue, y_blue, z_blue = create_3d_circle(1)

    ax.scatter(x_red, y_red, z_red,
               c='red', marker='x')

    ax.scatter(x_blue, y_blue, z_blue,
               c='blue', marker='o')
    ax.set_box_aspect([1, 1, 1])

    plt.title('3D Sphere with Red Crosses and Blue Dots')
    plt.show()

# Function to plot 2D circle with red crosses and blue dots
def plot_2d_circle():
    fig, ax = plt.subplots()

    x_red, y_red, x_blue, y_blue = create_2d_sphere(2)

    ax.scatter(x_red, y_red, c='red', marker='x')
    ax.scatter(x_blue, y_blue, c='blue', marker='o')
    ax.set_aspect('equal')
    plt.title('2D Circle with Red Crosses and Blue Dots')
    plt.axis('equal')
    plt.show()

#plot_3d_sphere()
#plot_2d_circle()
#plot_3d_circle()


x_red_source, y_red_source, z_red_source, x_blue_source, y_blue_source, z_blue_source = create_3d_circle(1)
x_red_target, y_red_target, x_blue_target, y_blue_target = create_2d_sphere(2)

fig, ax = plt.subplots()

ax.scatter(x_red_target, y_red_target, c='red', marker='o')
ax.scatter(x_blue_target, y_blue_target, c='blue', marker='o')
ax.set_aspect('equal')
plt.title('2D target samples')
plt.axis('equal')
plt.savefig('target 2D')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_red_source, y_red_source, z_red_source,
           c='red', marker='o', depthshade=False)

ax.scatter(x_blue_source, y_blue_source, z_blue_source,
           c='blue', marker='o', depthshade=False)
ax.set_box_aspect([1, 1, 1])

plt.title('3D source samples')
plt.savefig('target 3D')
plt.show()


x_red_source, y_red_source, z_red_source, x_blue_source, y_blue_source, z_blue_source = x_red_source.flatten(), y_red_source.flatten(), z_red_source.flatten(), x_blue_source.flatten(), y_blue_source.flatten(), z_blue_source.flatten()
x_red_target, y_red_target, x_blue_target, y_blue_target = x_red_target.flatten(), y_red_target.flatten(), x_blue_target.flatten(), y_blue_target.flatten()

x_source, y_source, z_source = np.concatenate((x_red_source, x_blue_source)), np.concatenate((y_red_source, y_blue_source)), np.concatenate((z_red_source, z_blue_source))
x_target, y_target = np.concatenate((x_red_target, x_blue_target)), np.concatenate((y_red_target, y_blue_target))

pos_source = x_source, y_source, z_source
pos_target = x_target, y_target

A_source = np.column_stack((x_source, y_source, z_source))
A_target = np.column_stack((x_target, y_target))

source_geometry = torch.tensor(A_source, dtype=torch.float32)
source_geometry = torch.cdist(source_geometry, source_geometry)

target_geometry = torch.tensor(A_target, dtype=torch.float32)
target_geometry = torch.cdist(target_geometry, target_geometry)

source_features = [1 for i in range(len(x_red_source))] + [0 for i in range(len(x_blue_source))]
target_features = [1 for i in range(len(x_red_target))] + [0 for i in range(len(x_blue_target))]

source_features_normalized = torch.tensor(source_features, dtype=torch.float32).unsqueeze(0)
target_features_normalized = torch.tensor(target_features, dtype=torch.float32).unsqueeze(0)

w_s = torch.ones(source_features_normalized.shape[1], device='cpu') / source_features_normalized.shape[1]
w_t = torch.ones(target_features_normalized.shape[1], device='cpu') / target_features_normalized.shape[1]


alpha = 0.3
rho = 0.4
eps = 1e-4
P, loss_ls = solver(alpha=alpha, rho=rho, eps=eps, 
    source_features=source_features_normalized,
    target_features=target_features_normalized,
    source_geometry=source_geometry,
    target_geometry=target_geometry,
    device='cpu',
    w_s=w_s,
    w_t=w_t,
    nits_bcd=5
)

loss = defaultdict(list)
for l in loss_ls:
    for k,v in l.items():
        loss[k].append(v)

fig, ax = plt.subplots(figsize=(10, 4))
ax.set_title("Mapping training loss")
ax.set_ylabel("Loss")
ax.set_xlabel("BCD step")
ax.stackplot(
    np.arange(len(loss_ls)),
    [
        (1 - alpha) * np.array(loss["wasserstein"]),
        alpha * np.array(loss["gromov_wasserstein"]),
        rho * np.array(loss["marginal_constraint_dim1"]),
        rho * np.array(loss["marginal_constraint_dim2"]),
        eps * np.array(loss["regularization"]),
    ],
    labels=[
        "wasserstein",
        "gromov_wasserstein",
        "marginal_constraint_dim1",
        "marginal_constraint_dim2",
        "regularization",
    ],
    alpha=0.8,
)
ax.legend()
plt.savefig('loss')
plt.show()

print(P.sum())

def plot_graph_matching_3D(params, pos_source, pos_target, P):

    x_red_source, x_blue_source, x_red_target, x_blue_target, y_red_source, y_blue_source, y_red_target, y_blue_target, z_red_source, z_blue_source = params
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')  

    pos_source = np.array(pos_source).T
    pos_target = np.array(pos_target).T

    ax.scatter(x_blue_source, y_blue_source, z_blue_source,
               c='blue', marker='o', s=6, depthshade=False)
    
    ax.scatter(x_red_source, y_red_source, z_red_source,
               c='red', marker='o', s=10, depthshade=False)

    ax.scatter(-x_red_target, y_red_target, 0,
               c='red', marker='o', s=10, depthshade=False)
    
    ax.scatter(-x_blue_target, y_blue_target, 0,
               c='blue', marker='o', s=6, depthshade=False)
    
    ax.set_aspect('equal')

    
    transport_matrix = P / P.max()

    for i in range(transport_matrix.shape[0]):
        for j in range(transport_matrix.shape[1]):
            x1, y1, z1 = pos_source[i]
            x2, y2 = pos_target[j]
            ax.plot([x1, -x2], [y1, y2], [z1, 0], color='green', alpha=float(transport_matrix[i, j]))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig('transport')
    plt.show()
    

params = x_red_source, x_blue_source, x_red_target, x_blue_target, y_red_source, y_blue_source, y_red_target, y_blue_target, z_red_source, z_blue_source
plot_graph_matching_3D(params, pos_source, pos_target, P)