import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION
    for i in range(max_iter):
        tmp = values.copy()
        for state in range(mdp.observation_space.n):
            tmp[state] = float("-inf")
            mdp.reset_state(state)
            for action in range(mdp.action_space.n):
                dest, rec, _, _ = mdp.step(action, transition=False)
                temp_value = rec + gamma * values[dest]
                if temp_value > tmp[state]:
                    tmp[state] = temp_value
        if (values==tmp).all():
            break
        values = tmp
    # END SOLUTION
    return values


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    print(env.grid)
    for i in range(max_iter):
        tmp = values.copy()
        for row in range(4):
            for col in range(4):
                if env.grid[row][col] in ['W', 'N', 'P']:
                    continue
                env.set_state(row, col)
                tmp[row, col] = float("-inf")
                for action in range(env.action_space.n):
                    (x, y), rec, end, dic = env.step(action, make_move=False)
                    if end is True:
                        temp_value = rec
                    else:
                        temp_value = rec + gamma * values[x, y]
                    if temp_value > tmp[row, col]:
                        tmp[row, col] = temp_value
        if np.isclose(values, tmp, rtol=theta).all():
            break
        values = tmp
    return values
    # END SOLUTION


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            # print((row, col), next_state, reward, probability)
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    for _ in range(max_iter):
        tmp_values = values.copy()
        delta = 0
        for row in range(4):
            for col in range(4):
                env.set_state(row, col)
                delta = value_iteration_per_state(env, tmp_values, gamma, values, theta)
        if delta < theta:
            break
        values = tmp_values
    return values
    # END SOLUTION
