# %%
from collections.abc import Callable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from scipy import stats

PYTENSOR_FLAGS = "linker=numba"


# %%
def hmc_step(
    starting_position: np.ndarray,
    ll_func: Callable,
    grad_func: Callable,
    step_size=0.01,
    n_steps=1,
    mass_matrix=None,
    debug_mode=False,
) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
    """

    Parameters
    -------
    Current_position : ndarray, vector
    ll_func : calleable
    grad_fun : calleable
    step_size : float
    n_steps : int
    mass_matrix: ndarray, matrix

    """
    position = starting_position
    positions = [starting_position]

    # generate initial momentum, either from a mass matrix
    # or a unit normal
    if mass_matrix is not None:
        momentum = stats.multivariate_normal(cov=mass_matrix).rvs()
    else:
        momentum = stats.norm(0, 1).rvs(len(position))

    starting_momentum = momentum
    momentums = [starting_momentum]

    # find kinetic and potential every of starting position
    current_u = ll_func(starting_position)
    current_k = np.sum(starting_momentum**2) / 2

    # make half a step
    momentum = momentum - step_size * grad_func(position) / 2

    # the central loop
    # make full steps for both position and momentum
    for i in range(n_steps):
        position = position + step_size * momentum
        # make a full step for momentum
        # except at the very end of trajectory
        if i != n_steps:
            momentum = momentum - step_size * grad_func(position)
            momentums.append(momentum)

        positions.append(position)

        if debug_mode:
            proposed_u = ll_func(position)
            proposed_k = np.sum(momentum**2) / 2
            energy_error = current_u - proposed_u + current_k - proposed_k
            print("intermediate energy error: ", np.exp(energy_error))

    momentum = momentum - step_size * grad_func(position) / 2

    # induce symmetry
    momentum = -momentum

    # find the energy error
    proposed_u = ll_func(position)
    proposed_k = np.sum(momentum**2) / 2
    energy_error = current_u - proposed_u + current_k - proposed_k

    if debug_mode:
        print("final energy error: ", np.exp(energy_error))

    # accept or reject
    accept = False
    if np.log(np.random.random()) < energy_error:
        # accept
        new_pos = position
        accept = True
    else:
        # reject
        new_pos = starting_position

    return new_pos, accept, np.array(positions), np.array(momentums)


# %%
# gather the likelihood and gradient of a 2d standard normal
rv = pm.Normal.dist()
x = pt.vector("x")
logp = -pm.logp(rv, value=x).sum()
dlogp = pt.jacobian(logp, x)

ll_func = pytensor.function(inputs=[x], outputs=logp)
grad_func = pytensor.function(inputs=[x], outputs=dlogp)

# %%
# simulate a single trajectory
result = [np.array([1, 1])]

result = hmc_step(
    starting_position=result[0],
    ll_func=ll_func,
    grad_func=grad_func,
    n_steps=23,
    step_size=0.1,
)

# plot a single trajectory
positions = result[2]
momentums = result[3]

plt.quiver(
    positions[:, 0],
    positions[:, 1],
    momentums[:, 0],
    momentums[:, 1],
    headlength=7,
    scale=60,
    headwidth=6,
    alpha=0.8,
    color="black",
)
plt.plot(positions[:, 0], positions[:, 1], "-", lw=3, color="black")

# %%
result = [np.random.normal(size=9)]

result = hmc_step(
    starting_position=result[0],
    ll_func=ll_func,
    grad_func=grad_func,
    n_steps=23,
    step_size=0.1,
)
# plot a single trajectory
positions = result[2]
plt.plot(positions)
# %%
# generate a full dataset
# 25 samples, 23 subsamples
n_edges = 20
n_samples = 25
n_steps = 46
initial_position = np.random.normal(size=n_edges)
position = initial_position

all_positions = np.zeros((n_samples, n_edges))
all_subpositions = np.zeros((n_samples, n_steps, n_edges))
for i in range(n_samples):
    position, _, subpositions, _ = hmc_step(
        starting_position=position,
        ll_func=ll_func,
        grad_func=grad_func,
        n_steps=n_steps,
        step_size=0.05,
    )
    all_positions[i, :] = position
    all_subpositions[i, :, :] = subpositions[:-1, :]

all_positions.shape, all_subpositions.shape

# %%
plt.plot(all_positions[:, 0])

# %%
plt.plot(all_subpositions.reshape(n_samples * n_steps, n_edges)[:, 0])

# %%
edge_list = np.array(
    [
        [1, 0],
        [2, 0],
        [2, 1],
        [3, 0],
        [3, 1],
        [4, 2],
        [4, 3],
        [5, 0],
        [5, 1],
        [5, 2],
        [5, 4],
        [6, 1],
        [6, 5],
        [7, 2],
        [7, 4],
        [7, 5],
        [7, 6],
        [8, 4],
        [8, 6],
        [8, 7],
    ]
)

g = np.zeros((9, 9))

for edge in edge_list:
    g[edge[0], edge[1]] = 1
    g[edge[1], edge[0]] = 1

G = nx.Graph(g)
nx.draw(G, with_labels=True)
edge_list.shape

# %%
# transform back and forth between logits and numbers
edge_probabilities = pm.math.logit(
    [
        2 / 3,
        1 / 2,
        1 / 2,
        3 / 5,
        2 / 5,
        3 / 5,
        2 / 3,
        2 / 5,
        1 / 2,
        1 / 2,
        2 / 5,
        4 / 5,
        1 / 2,
        1 / 2,
        3 / 5,
        1 / 2,
        2 / 3,
        3 / 5,
        3 / 5,
        3 / 5,
    ],
).eval()
edge_probabilities = np.expand_dims(edge_probabilities, axis=0)
edge_probabilities.shape

# %% transform the logits to probabilities
all_positions.shape, all_subpositions.shape
all_positions_prob = pm.math.invlogit(0.5 * (all_positions + edge_probabilities)).eval()
all_subpositions_prob = pm.math.invlogit(
    1 * (all_subpositions + edge_probabilities)
).eval()

plt.plot(all_positions_prob[:, 0])

# %%
plt.plot(all_subpositions_prob.reshape(n_samples * n_steps, n_edges)[:, :5])
# %%
# Sample the edges from a bernoulli distribution
# edge_samples = np.random.binomial(1, all_positions_prob)
edge_samples = np.round(all_positions_prob)
edge_samples[:4, :]


# %%
# prototype the visualization
# phase one: edge weight animation
from matplotlib import colormaps

cmap = colormaps["gist_gray"]

g = np.zeros((9, 9))
for i in range(len(edge_list)):
    row, col = edge_list[i]
    g[row, col] = all_positions_prob[0, i]


G = nx.Graph(g)
pos = nx.layout.spring_layout(G)
nx.draw_networkx(
    G,
    pos=pos,
    with_labels=False,
    node_color="lightgray",
    edge_cmap=cmap,
    edge_color=all_positions_prob[0, :],
)
plt.title("Edge weight animation")


# %%
g = np.zeros((9, 9))
for i in range(len(edge_list)):
    row, col = edge_list[i]
    g[row, col] = edge_samples[0, i]

G = nx.Graph(g)
nx.draw_networkx(G, pos=pos, with_labels=False, node_color="lightgray")
plt.title("Sampled graph")

# %%
pos = nx.layout.spring_layout(G, iterations=1, pos=pos)
nx.draw_networkx(G, pos=pos, with_labels=False, node_color="lightgray")
plt.title("One iteration of spring layout")


# %%
# Export to json


# %%
import json

data = {
    # Fixed graph topology: list of [source, target] pairs
    "edges": edge_list.tolist(),  # (20, 2)
    # Phase 1: HMC trajectory weights, probabilities in [0, 1]
    "subpositions": all_subpositions_prob.tolist(),  # (25, 23, 20)
    # Phase 2: Binomial samples, binary edge existence
    "edge_samples": edge_samples.tolist(),  # (25, 20)
    # Starting layout — so D3 doesn't randomize node positions on load
    "initial_pos": {str(k): list(v) for k, v in pos.items()},  # {node_id: [x, y]}
}

with open("../assets/graph_data.json", "w") as f:
    json.dump(data, f)


# %%
# export to .csv
import xarray as xr

subpositions_xr = xr.DataArray(
    all_subpositions_prob,
    dims=("positions", "subpositions", "edges"),
    coords={
        "positions": np.arange(n_samples),
        "subpositions": np.arange(n_steps),
        "edges": np.arange(n_edges),
    },
)
subpositions_xr.to_dataframe(name="subpositions_data").reset_index().to_csv(
    "subpositions.csv"
)

positions_xr = xr.DataArray(
    all_positions_prob,
    dims=("positions", "edges"),
    coords={
        "positions": np.arange(n_samples),
        "edges": np.arange(n_edges),
    },
)
subpositions_xr.to_dataframe(name="positions_data").reset_index().to_csv(
    "positions.csv"
)
