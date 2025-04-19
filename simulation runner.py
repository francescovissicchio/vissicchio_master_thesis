import os
import networkx as nx
import matplotlib.pyplot as plt
from utility import create_graph
from models import sis, seir, seirv, seirvq
from datetime import datetime

model_map = {
    0: ("SIS", sis),
    1: ("SEIR", seir),
    2: ("SEIRV", seirv),
    3: ("SEIRVQ", seirvq)
}

param_prompts = {
    "SIS": [("beta", "Probabilità di infezione (beta): "),
            ("mu", "Probabilità di guarigione (mu): ")],
    "SEIR": [("beta", "Probabilità di infezione (beta): "),
             ("alpha", "Probabilità di incubazione (alpha): "),
             ("gamma", "Probabilità di guarigione (gamma): ")],
    "SEIRV": [("beta", "Probabilità di infezione (beta): "),
              ("alpha", "Probabilità di incubazione (alpha): "),
              ("gamma", "Probabilità di guarigione (gamma): "),
              ("vacc_prob", "Probabilità di vaccinazione: ")],
    "SEIRVQ": [("beta", "Probabilità di infezione (beta): "),
               ("alpha", "Probabilità di incubazione (alpha): "),
               ("gamma", "Probabilità di guarigione (gamma): "),
               ("vacc_prob", "Probabilità di vaccinazione: "),
               ("quar_prob", "Probabilità di quarantena: ")]
}

def get_float(prompt, min_val=0.0, max_val=1.0):
    while True:
        try:
            val = float(input(prompt))
            if not (min_val <= val <= max_val):
                raise ValueError
            return val
        except ValueError:
            print(f"Inserisci un numero valido tra {min_val} e {max_val}.")

def get_int(prompt, min_val=1):
    while True:
        try:
            val = int(input(prompt))
            if val < min_val:
                raise ValueError
            return val
        except ValueError:
            print(f"Inserisci un intero >= {min_val}.")

while True:
    user_input = input("\nType 'q' or 'quit' to exit, or press Enter to run a new simulation: ").lower()
    if user_input in {'q', 'quit'}:
        print("Exiting simulation loop.")
        break

    print("\nScegli un modello:")
    for i, (name, _) in model_map.items():
        print(f"{i}: {name}")

    try:
        model_choice = int(input("Inserisci la tua scelta: "))
        model_name, model_module = model_map[model_choice]
    except (ValueError, KeyError):
        print("Scelta non valida.")
        continue

    # Parametri del modello
    params = {}
    for key, prompt in param_prompts[model_name]:
        params[key] = get_float(prompt)

    # Tipo di grafo
    graph_type = get_int("Tipo di grafo: 0 per ER, 1 per GNP, 2 per Barabasi-Albert, 3 per SBM: ", 0)
    if graph_type not in {0, 1, 2, 3}:
        print("Tipo di grafo non valido.")
        continue

    # Costruzione del grafo
    if graph_type in [0, 1]:
        n = get_int("Numero di nodi (max 1000): ", min_val=1)
        if n > 1000:
            print("⚠️ Numero massimo di nodi consentito: 1000.")
            continue
        p = get_float("Probabilità di creazione di arco (tra 0 e 1): ")
        graph = create_graph('erdos-reny' if graph_type == 0 else 'gnp', {'n': n, 'p': p})
    elif graph_type == 2:
        n = get_int("Numero di nodi (max 1000): ", min_val=1)
        if n > 1000:
            print("⚠️ Numero massimo di nodi consentito: 1000.")
            continue
        m = get_int("Numero di archi per nuovo nodo (m): ")
        graph = create_graph('barabasi-albert', {'n': n, 'm': m})
    elif graph_type == 3:
        n = get_int("Numero totale di nodi (max 1000): ", min_val=1)
        if n > 1000:
            print("⚠️ Numero massimo di nodi consentito: 1000.")
            continue
        graph = create_graph('sbm', {
            'n': n,
            'sizes': [n // 2, n // 2],
            'probs': [[0.8, 0.2], [0.2, 0.8]]
        })

    # Creazione simulazione
    try:
        sim = model_module.get_simulation(graph, params)
    except Exception as e:
        print(f"Errore nella creazione della simulazione: {e}")
        continue

    steps = get_int("Quanti step simulare? (max 50): ", min_val=1)
    if steps > 50:
        print("⚠️ Numero massimo di step consentito: 50.")
        continue

    print(f"Esecuzione simulazione per {steps} step...")

    try:
        sim.run(steps)
    except Exception as e:
        print(f"Errore durante l'esecuzione della simulazione: {e}")
        continue

    # Visualizzazione
    if hasattr(sim, 'draw'):
        try:
            plt.figure()
            sim.draw()
        except Exception as e:
            print(f"Errore nel disegno del grafo: {e}")

    if hasattr(sim, 'plot'):
        try:
            plt.figure()
            sim.plot()
        except Exception as e:
            print(f"Errore nel grafico dei dati: {e}")

    plt.show()

    # Salvataggio
    save = input("Vuoi salvare in CSV? (y/n): ").lower()
    if save == 'y' and hasattr(sim, 'save_to_csv'):
        agg = input("Formato aggregato (conteggio per step)? (y/n): ").lower() == 'y'
        filename = input("Nome file (lascia vuoto per generarlo automaticamente): ").strip()
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/simulation_{timestamp}.csv"

        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            sim.save_to_csv(filename=filename, aggregate=agg)
            print(f"Salvato in {filename}")
        except Exception as e:
            print(f"Errore nel salvataggio: {e}")


