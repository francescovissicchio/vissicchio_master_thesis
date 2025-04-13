import networkx as nx
from utility import create_graph
import matplotlib.pyplot as plt
from models import sis, seir, seirv, seirvq

model_map = {
    0: ("SIS", sis),
    1: ("SEIR", seir),
    2: ("SEIRV", seirv),
    3: ("SEIRVQ", seirvq)
}

print("Choose a model:")
for i, (name, _) in model_map.items():
    print(f"{i}: {name}")
model_choice = int(input("Enter your choice: "))

model_entry = model_map.get(model_choice)
if model_entry is None:
    print("Invalid model choice.")
    exit()

model_name, model_module = model_entry

# Parametri personalizzati
params = {}
if model_name == "SIS":
    params['beta'] = float(input("Infection probability (beta): "))
    params['mu'] = float(input("Recovery probability (mu): "))
elif model_name == "SEIR":
    params['beta'] = float(input("Infection probability (beta): "))
    params['alpha'] = float(input("Incubation probability (alpha): "))
    params['gamma'] = float(input("Recovery probability (gamma): "))
elif model_name == "SEIRV":
    params['beta'] = float(input("Infection probability (beta): "))
    params['alpha'] = float(input("Incubation probability (alpha): "))
    params['gamma'] = float(input("Recovery probability (gamma): "))
    params['vacc_prob'] = float(input("Vaccination probability: "))
elif model_name == "SEIRVQ":
    params['beta'] = float(input("Infection probability (beta): "))
    params['alpha'] = float(input("Incubation probability (alpha): "))
    params['gamma'] = float(input("Recovery probability (gamma): "))
    params['vacc_prob'] = float(input("Vaccination probability: "))
    params['quar_prob'] = float(input("Quarantine probability: "))

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

# Costruzione della simulazione
sim = model_module.get_simulation(graph, params)

# Esegui simulazione
steps = int(input("How many steps to simulate? "))
sim.run(steps)

save = input("Save simulation to CSV? (y/n): ").lower()
if save == 'y':
    agg = input("Aggregate format (counts per step)? (y/n): ").lower() == 'y'
    filename = input("Enter filename (e.g. results/sim.csv): ")
    sim.save_to_csv(filename=filename, aggregate=agg)
    print(f"Saved to {filename}")

plt.figure()      # Crea una nuova figura per il disegno del grafo
sim.draw()

plt.figure()      # Crea una nuova figura per il plot temporale
sim.plot()

plt.show()        # Mostra entrambe le figure in due finestre separate

