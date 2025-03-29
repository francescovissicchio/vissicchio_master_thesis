import networkx as nx
from utility import create_graph
from SimulationSIS import Simulation as SimulationSIS, initial_state as SIS_initial, state_transition as SIS_trans
from SimulationSEIR import SimulationSEIR
from SimulationSEIRV import SimulationSEIRV
from SimulationSEIRVQ import SimulationSEIRVQ
import matplotlib.pyplot as plt

# Selezione modello
print("Choose a model:")
print("0: SIS")
print("1: SEIR")
print("2: SEIRV")
print("3: SEIRVQ")

model_choice = int(input("Enter your choice: "))

# Selezione tipo di grafo
graph_type = int(input("Choose the Graph Model: 0 for ER, 1 for GNP, 2 for Barabasi-Albert, 3 for SBM: "))

if graph_type == 0 or graph_type == 1:
    n = int(input("Number of nodes: "))
    p = float(input("Edge creation probability (between 0 to 1): "))
    graph = create_graph('erdos-reny' if graph_type == 0 else 'GNP', {'n': n, 'p': p})

elif graph_type == 2:
    n = int(input("Number of nodes: "))
    m = int(input("Edges per new node (m): "))
    graph = create_graph('barabasi-albert', {'n': n, 'm': m})

elif graph_type == 3:
    n = int(input("Total number of nodes: "))
    graph = create_graph('sbm', {
        'n': n,
        'sizes': [n // 2, n // 2],
        'probs': [[0.8, 0.2], [0.2, 0.8]]
    })
else:
    print("Invalid graph type.")
    exit()

# Simulazione dinamica
if model_choice == 0:
    sim = SimulationSIS(graph, initial_state=SIS_initial, state_transition=SIS_trans, name='SIS Model')
elif model_choice == 1:
    sim = SimulationSEIR(graph)
elif model_choice == 2:
    sim = SimulationSEIRV(graph)
elif model_choice == 3:
    sim = SimulationSEIRVQ(graph)
else:
    print("Invalid model choice.")
    exit()

# Esegui simulazione
steps = int(input("How many steps to simulate? "))
sim.run(steps)

plt.figure()      # Crea una nuova figura per il disegno del grafo
sim.draw()

plt.figure()      # Crea una nuova figura per il plot temporale
sim.plot()

plt.show()        # Mostra entrambe le figure in due finestre separate

