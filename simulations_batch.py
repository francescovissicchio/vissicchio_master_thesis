import os
import random
import csv
from datetime import datetime
import networkx as nx
from utility import create_graph
from models import sis, seir, seirv, seirvq
import matplotlib.pyplot as plt
from tqdm import tqdm

# === MODELLI DISPONIBILI ===
model_map = {
    0: ("SIS", sis),
    1: ("SEIR", seir),
    2: ("SEIRV", seirv),
    3: ("SEIRVQ", seirvq)
}

param_prompts = {
    "SIS": ["beta", "mu"],
    "SEIR": ["beta", "alpha", "gamma"],
    "SEIRV": ["beta", "alpha", "gamma", "vacc_prob"],
    "SEIRVQ": ["beta", "alpha", "gamma", "vacc_prob", "quar_prob"]
}

STATE_NAMES = ['E', 'S', 'V', 'I', 'Q', 'R']

# === DIRECTORY DI OUTPUT PER DATA ===
today_str = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = os.path.join("simulation_output", today_str)
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_path = os.path.join(OUTPUT_DIR, "log.txt")
summary_path = os.path.join(OUTPUT_DIR, "summary.csv")

# === HEADER SUMMARY CSV ===
with open(summary_path, mode='w', newline='', encoding='utf-8') as f_summary:
    writer = csv.writer(f_summary)
    writer.writerow([
        "ID", "Model", "GraphType", "GraphDesc",
        "Steps", "Parameters", "Folder", "DateTime"
    ])

# === FUNZIONI DI SUPPORTO ===
def get_random_param():
    return round(random.uniform(0.01, 0.99), 3)

def get_graph_config(graph_type):
    n = random.randint(10, 500)
    if graph_type in [0, 1]:
        p = round(random.uniform(0.01, 0.3), 3)
        return {'type': 'erdos-reny' if graph_type == 0 else 'gnp', 'args': {'n': n, 'p': p}}, f"n={n}, p={p}"
    elif graph_type == 2:
        m = random.randint(1, min(n-1, 10))
        return {'type': 'barabasi-albert', 'args': {'n': n, 'm': m}}, f"n={n}, m={m}"
    elif graph_type == 3:
        sizes = [n // 2, n - n // 2]
        probs = [[0.8, 0.2], [0.2, 0.8]]
        return {'type': 'sbm', 'args': {'n': n, 'sizes': sizes, 'probs': probs}}, f"n={n}, sizes={sizes}, probs={probs}"

# === CICLO SIMULAZIONI ===
for i in tqdm(range(1, 201), desc="Simulazioni"):
    try:
        model_choice = random.choice(list(model_map.keys()))
        model_name, model_module = model_map[model_choice]
        param_keys = param_prompts[model_name]
        params = {key: get_random_param() for key in param_keys}

        graph_type = random.randint(0, 3)
        graph_info, graph_desc = get_graph_config(graph_type)
        graph = create_graph(graph_info['type'], graph_info['args'])

        sim = model_module.get_simulation(graph, params)
        steps = random.randint(5, 50)
        sim.run(steps)

        base_name = f"{i:03d}_{model_name}_graph{graph_type}"
        sim_folder = os.path.join(OUTPUT_DIR, f"sim_{i:03d}")
        os.makedirs(sim_folder, exist_ok=True)

        timestamp = datetime.now().isoformat()

        info_text = f"Modello: {model_name}\n"
        info_text += f"Parametri: {params}\n"
        info_text += f"Grafo: {graph_info['type']} ({graph_desc})\n"
        info_text += f"Step simulazione: {steps}\n"
        info_text += f"Data/Ora: {timestamp}\n\n"

        # === SALVATAGGIO FIGURE ===
        if hasattr(sim, 'draw'):
            try:
                plt.figure()
                sim.draw()
                plt.savefig(os.path.join(sim_folder, f"{base_name}_graph.png"))
                plt.close()
            except Exception as e:
                print(f"⚠️ Errore nel disegno del grafo: {e}")

        if hasattr(sim, 'plot'):
            try:
                plt.figure()
                sim.plot()
                plt.savefig(os.path.join(sim_folder, f"{base_name}_plot.png"))
                plt.close()
            except Exception as e:
                print(f"⚠️ Errore nel plot dei dati: {e}")

        # === SALVATAGGIO EDGELIST ===
        edgelist_path = os.path.join(sim_folder, f"{base_name}_edgelist.csv")
        nx.write_edgelist(graph, edgelist_path, delimiter=",", data=False)

        # === SALVATAGGIO CSV AGGREGATO ===
        agg_file = os.path.join(sim_folder, f"{base_name}_agg.csv")
        sim.save_to_csv(filename=agg_file, aggregate=True)
        with open(agg_file, "r+", encoding="utf-8") as f:
            content = f.read()
            f.seek(0, 0)
            f.write("# INFORMAZIONI SIMULAZIONE\n")
            f.write("# -------------------------\n")
            f.writelines(f"# {line}\n" for line in info_text.strip().split("\n"))
            f.write("# -------------------------\n\n")
            f.write(content)

        # === SALVATAGGIO CSV PER NODO ===
        np_file = os.path.join(sim_folder, f"{base_name}_np.csv")
        with open(np_file, "w", newline='', encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=["time", "node"] + STATE_NAMES)
            writer.writeheader()
            for t, snapshot in enumerate(sim.snapshots):
                for node_id, state in snapshot.items():
                    row = {"time": t, "node": node_id}
                    for s in STATE_NAMES:
                        row[s] = 1 if state == s else 0
                    writer.writerow(row)

        # === METADATA HEADER SU CSV _np ===
        with open(np_file, "r+", encoding="utf-8") as f:
            content = f.read()
            f.seek(0, 0)
            f.write("# INFORMAZIONI SIMULAZIONE\n")
            f.write("# -------------------------\n")
            f.writelines(f"# {line}\n" for line in info_text.strip().split("\n"))
            f.write("# -------------------------\n\n")
            f.write(content)

        # === LOG ===
        with open(log_path, mode='a', encoding='utf-8') as f_log:
            f_log.write(f"[{timestamp}] Completata simulazione {i:03d}: {model_name}, grafo {graph_info['type']}, step {steps}\n")

        # === SUMMARY ===
        with open(summary_path, mode='a', newline='', encoding='utf-8') as f_summary:
            writer = csv.writer(f_summary)
            writer.writerow([
                f"{i:03d}", model_name, graph_info['type'], graph_desc,
                steps, str(params), f"sim_{i:03d}", timestamp
            ])

    except Exception as e:
        error_msg = f"❌ Errore nella simulazione {i:03d}: {e}"
        print(error_msg)
        with open(log_path, mode='a', encoding='utf-8') as f_log:
            f_log.write(f"[{datetime.now().isoformat()}] {error_msg}\n")



