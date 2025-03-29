import random as rd
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl

# Parametri SEIRV
BETA = 0.2        # contagio
ALPHA = 0.3       # incubazione
GAMMA = 0.1       # guarigione
VACC_PROB = 0.01  # probabilità di vaccinazione per i suscettibili

def initial_state_seirv(G):
    state = {node: 'S' for node in G.nodes}
    patient_zero = rd.choice(list(G.nodes))
    state[patient_zero] = 'E'  # Iniziamo con un esposto
    return state

def state_transition_seirv(G, current_state):
    next_state = {}

    for node in G.nodes:
        state = current_state[node]

        if state == 'S':
            # Prova a vaccinarsi
            if rd.random() < VACC_PROB:
                next_state[node] = 'V'
                continue

            # Prova a essere infettato dai vicini
            for neighbor in G.neighbors(node):
                if current_state[neighbor] == 'I':
                    if rd.random() < BETA:
                        next_state[node] = 'E'
                        break

        elif state == 'E':
            if rd.random() < ALPHA:
                next_state[node] = 'I'

        elif state == 'I':
            if rd.random() < GAMMA:
                next_state[node] = 'R'

        # R e V non cambiano più stato

    return next_state

class SimulationSEIRV:
    def __init__(self, G, initial_state=initial_state_seirv, state_transition=state_transition_seirv,
                 stop_condition=None, name='SEIRV Simulation'):
        self.G = G.copy()
        self._initial_state = initial_state
        self._state_transition = state_transition
        self._stop_condition = stop_condition
        self.name = name

        self._states = []
        self._value_index = {}
        self._cmap = plt.cm.get_cmap('tab10')
        self._initialize()
        self._pos = nx.spring_layout(G)

    def _initialize(self):
        if callable(self._initial_state):
            state = self._initial_state(self.G)
        else:
            state = self._initial_state
        nx.set_node_attributes(self.G, state, 'state')
        self._append_state(state)

    def _append_state(self, state):
        self._states.append(state)
        for val in set(state.values()):
            if val not in self._value_index:
                self._value_index[val] = len(self._value_index)

    def _step(self):
        state = nx.get_node_attributes(self.G, 'state')
        if self._stop_condition and self._stop_condition(self.G, state):
            raise StopIteration
        new_state = self._state_transition(self.G, state.copy())
        state.update(new_state)
        nx.set_node_attributes(self.G, state, 'state')
        self._append_state(state)

    def run(self, steps=1):
        for _ in range(steps):
            try:
                self._step()
            except StopIteration:
                break

    def state(self, step=-1):
        return self._states[step]

    def draw(self, step=-1, labels=None, **kwargs):
        state = self.state(step)
        node_colors = [self._cmap(self._value_index[state[n]]) for n in self.G.nodes]
        nx.draw(self.G, pos=self._pos, node_color=node_colors, **kwargs)

        if labels is None:
            labels = sorted(set(state.values()), key=self._value_index.get)
        patches = [mpl.patches.Patch(color=self._cmap(self._value_index[l]), label=l) for l in labels]
        plt.legend(handles=patches)

        step_title = 'initial state' if step == 0 else f'step {step}'
        plt.title(f'{self.name}: {step_title}')

    def plot(self):
        x = range(len(self._states))
        counts = [Counter(s.values()) for s in self._states]
        labels = sorted(set(val for c in counts for val in c), key=self._value_index.get)

        for label in labels:
            series = [c.get(label, 0) / len(self.G) for c in counts]
            plt.plot(x, series, label=label)

        plt.title(f'{self.name}: state proportions over time')
        plt.xlabel('Step')
        plt.ylabel('Proportion')
        plt.legend()
        plt.show()
