import random as rd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from collections import Counter
import csv

def initial_state(G):
    return {node: 'I' if node == rd.choice(list(G.nodes)) else 'S' for node in G.nodes}

def state_transition(G, current_state, mu, beta):
    next_state = {}
    for node in G.nodes:
        if current_state[node] == 'I':
            if rd.random() < mu:
                next_state[node] = 'S'
        elif current_state[node] == 'S':
            for neighbor in G.neighbors(node):
                if current_state[neighbor] == 'I':
                    if rd.random() < beta:
                        next_state[node] = 'I'
                        break
    return next_state

class Simulation:
    def __init__(self, G, initial_state, state_transition, stop_condition=None, name=''):
        self.G = G.copy()
        self._initial_state = initial_state
        self._state_transition = state_transition
        self._stop_condition = stop_condition
        self.name = name or 'Simulation'
        self._states = []
        self._value_index = {}
        self._cmap = plt.cm.get_cmap('tab10')
        self._initialize()
        self._pos = nx.spring_layout(G)

    def save_to_csv(self, filename, aggregate=False):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            if aggregate:
                # Scrive: Step, Stato1, Stato2, ...
                counts = [Counter(s.values()) for s in self._states]
                labels = sorted({k for count in counts for k in count}, key=self._value_index.get)
                writer.writerow(['Step'] + labels)
                for step, count in enumerate(counts):
                    row = [step] + [count.get(label, 0) for label in labels]
                    writer.writerow(row)
            else:
                # Scrive: Step, Node, State
                writer.writerow(['Step', 'Node', 'State'])
                for step, state in enumerate(self._states):
                    for node, value in state.items():
                        writer.writerow([step, node, value])

    def _initialize(self):
        state = self._initial_state(self.G)
        nx.set_node_attributes(self.G, state, 'state')
        self._append_state(state)

    def _append_state(self, state):
        self._states.append(state)
        for value in set(state.values()):
            if value not in self._value_index:
                self._value_index[value] = len(self._value_index)

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
        title = f'{self.name}: {"initial state" if step == 0 else f"step {step}"}'
        plt.title(title)


    def plot(self):
        x = range(len(self._states))
        counts = [Counter(s.values()) for s in self._states]
        labels = sorted({k for count in counts for k in count}, key=self._value_index.get)
        for label in labels:
            series = [c.get(label, 0) / len(self.G) for c in counts]
            plt.plot(x, series, label=label)
        plt.title(f'{self.name}: state proportions over time')
        plt.xlabel('Step')
        plt.ylabel('Proportion')
        plt.legend()

    @property
    def snapshots(self):
        return self._states


def get_simulation(G, params):
    mu = params['mu']
    beta = params['beta']

    def transition(G, state):
        return state_transition(G, state, mu, beta)

    return Simulation(G, initial_state=initial_state, state_transition=transition, name="SIS")
