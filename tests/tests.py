import os
import traceback
from datetime import datetime
from utility import create_graph
from models import sis, seir, seirv, seirvq

# Modelli e parametri di esempio
model_map = {
    "SIS": (sis, {'beta': 0.2, 'mu': 0.1}),
    "SEIR": (seir, {'beta': 0.2, 'alpha': 0.1, 'gamma': 0.05}),
    "SEIRV": (seirv, {'beta': 0.2, 'alpha': 0.1, 'gamma': 0.05, 'vacc_prob': 0.02}),
    "SEIRVQ": (seirvq, {'beta': 0.2, 'alpha': 0.1, 'gamma': 0.05, 'vacc_prob': 0.02, 'quar_prob': 0.05})
}

# Tipi di grafo
graph_map = {
    "ER": lambda: create_graph("erdos-reny", {'n': 100, 'p': 0.05}),
    "GNP": lambda: create_graph("gnp", {'n': 100, 'p': 0.05}),
    "Barabasi-Albert": lambda: create_graph("barabasi-albert", {'n': 100, 'm': 2}),
    "SBM": lambda: create_graph("sbm", {
        'n': 100,
        'sizes': [50, 50],
        'probs': [[0.8, 0.2], [0.2, 0.8]]
    })
}

# Percorsi output
output_dir = "../test_outputs"
log_file = os.path.join(output_dir, "error_log.txt")
os.makedirs(output_dir, exist_ok=True)

# Logging iniziale
with open(log_file, "w") as f:
    f.write(f"== LOG TEST ESECUZIONE {datetime.now()} ==\n\n")

print("== Inizio Test Esteso ==")

for model_name, (model_module, params) in model_map.items():
    for graph_name, graph_func in graph_map.items():
        label = f"[{model_name} + {graph_name}]"
        try:
            graph = graph_func()
            sim = model_module.get_simulation(graph, params)
            sim.run(50)

            # Salvataggio CSV
            filename = os.path.join(output_dir, f"{model_name}_{graph_name}.csv".replace(" ", "_"))
            sim.save_to_csv(filename=filename, aggregate=True)

            print(f"{label:40} ✅ OK — salvato: {filename}")

        except Exception as e:
            print(f"{label:40} ❌ ERRORE")
            with open(log_file, "a") as f:
                f.write(f"{label}\n{traceback.format_exc(limit=3)}\n")

print("\n== Fine Test ==")
print(f"Errori (se presenti) salvati in: {log_file}")
