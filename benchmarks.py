import time
import psutil
import os
import matplotlib.pyplot as plt
import pandas as pd
from utility import create_graph
from models import sis, seir, seirv, seirvq
import scipy

process = psutil.Process(os.getpid())

model_map = {
    "SIS": (sis, {'beta': 0.2, 'mu': 0.1}),
    "SEIR": (seir, {'beta': 0.2, 'alpha': 0.1, 'gamma': 0.05}),
    "SEIRV": (seirv, {'beta': 0.2, 'alpha': 0.1, 'gamma': 0.05, 'vacc_prob': 0.02}),
    "SEIRVQ": (seirvq, {'beta': 0.2, 'alpha': 0.1, 'gamma': 0.05, 'vacc_prob': 0.02, 'quar_prob': 0.05})
}

graph_types = {
    "ER": lambda n: create_graph("erdos-reny", {'n': n, 'p': 0.01}),
    "Barabasi-Albert": lambda n: create_graph("barabasi-albert", {'n': n, 'm': 3}),
    "SBM": lambda n: create_graph("sbm", {
        'n': n,
        'sizes': [n // 2, n // 2],
        'probs': [[0.05, 0.01], [0.01, 0.05]]
    })
}

graph_sizes = [1000, 5000, 10000]
steps = 10
results = []

print("== BENCHMARK ESTESO ==")

for model_name, (model_module, params) in model_map.items():
    for graph_type, graph_func in graph_types.items():
        for n in graph_sizes:
            label = f"{model_name} | {graph_type} | n={n}"
            try:
                graph = graph_func(n)
                mem_before = process.memory_info().rss / (1024 * 1024)  # in MB
                start = time.time()
                sim = model_module.get_simulation(graph, params)
                sim.run(steps)
                duration = time.time() - start
                mem_after = process.memory_info().rss / (1024 * 1024)  # in MB
                mem_used = mem_after - mem_before

                results.append({
                    "model": model_name,
                    "graph_type": graph_type,
                    "n": n,
                    "time_sec": round(duration, 2),
                    "mem_MB": round(mem_used, 2)
                })

                print(f"{label:50} ✅ {duration:.2f} sec | RAM: {mem_used:.2f} MB")

            except Exception as e:
                print(f"{label:50} ❌ ERRORE — {e}")
                results.append({
                    "model": model_name,
                    "graph_type": graph_type,
                    "n": n,
                    "time_sec": None,
                    "mem_MB": None,
                    "error": str(e)
                })

# Salva CSV
df = pd.DataFrame(results)
df.to_csv("benchmark_results.csv", index=False)
print("\nRisultati salvati in benchmark_results.csv")

# Crea grafico
for model in df["model"].unique():
    plt.figure()
    for graph_type in df["graph_type"].unique():
        sub = df[(df["model"] == model) & (df["graph_type"] == graph_type)]
        if not sub["time_sec"].isnull().all():
            plt.plot(sub["n"], sub["time_sec"], label=graph_type, marker='o')
    plt.title(f"Tempo vs Numero Nodi — {model}")
    plt.xlabel("Numero Nodi")
    plt.ylabel("Tempo (secondi)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"benchmark_{model}.png")
    plt.close()

print("Grafici salvati come benchmark_[MODELLO].png")
print(nx.info(G))

