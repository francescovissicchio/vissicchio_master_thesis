import random as rd
import networkx as nx
from models.sis import Simulation

def initial_state(G):
    state = {node: 'S' for node in G.nodes}
    patient_zero = rd.choice(list(G.nodes))
    state[patient_zero] = 'E'
    return state

def state_transition(G, current_state, beta, alpha, gamma, vacc_prob, quar_prob):
    next_state = {}

    for node in G.nodes:
        state = current_state[node]

        if state == 'S':
            if rd.random() < vacc_prob:
                next_state[node] = 'V'
                continue
            for neighbor in G.neighbors(node):
                if current_state[neighbor] == 'I':
                    if rd.random() < beta:
                        next_state[node] = 'E'
                        break

        elif state == 'E':
            if rd.random() < alpha:
                next_state[node] = 'I'

        elif state == 'I':
            if rd.random() < quar_prob:
                next_state[node] = 'Q'
            elif rd.random() < gamma:
                next_state[node] = 'R'

        elif state == 'Q':
            if rd.random() < gamma:
                next_state[node] = 'R'

    return next_state

def get_simulation(G, params):
    beta = params['beta']
    alpha = params['alpha']
    gamma = params['gamma']
    vacc_prob = params['vacc_prob']
    quar_prob = params['quar_prob']

    def transition(G, state):
        return state_transition(G, state, beta, alpha, gamma, vacc_prob, quar_prob)

    return Simulation(G, initial_state=initial_state, state_transition=transition, name="SEIRVQ")

