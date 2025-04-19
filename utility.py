import networkx as nx


def create_graph(model, parameters):
    model = model.lower()  # normalizza il nome modello

    if model in {'erdos-reny', 'gnp'}:
        n = parameters['n']
        p = parameters['p']
        G = nx.erdos_renyi_graph(n, p)
    elif model == 'barabasi-albert':
        n = parameters['n']
        m = parameters['m']
        G = nx.barabasi_albert_graph(n, m)
    elif model == 'sbm':
        sizes = parameters['sizes']
        probs = parameters['probs']
        G = nx.stochastic_block_model(sizes, probs, seed=0)
    else:
        raise ValueError(f"Tipo di grafo non riconosciuto: {model}")

    return G


'''
if __name__ == "__main__":
    # Test cases for different network models

    # Erdős-Rényi random graph
    er_graph = create_graph('erdos-reny', {'n': 10, 'p': 0.5})
    print("Erdős-Rényi Graph:")
    print(f"Number of nodes: {xx.number_of_nodes(er_graph)}")
    print(f"Number of edges: {xx.number_of_edges(er_graph)}")

    # G(n,p) random graph
    gnp_graph = create_graph('GNP', {'n': 10, 'p': 0.5})
    print("\nG(n,p) Graph:")
    print(f"Number of nodes: {xx.number_of_nodes(gnp_graph)}")
    print(f"Number of edges: {xx.number_of_edges(gnp_graph)}")

    # Barabási-Albert preferential attachment model
    ba_graph = create_graph('barabasi-albert', {'n': 100, 'm': 2})
    print("\nBarabási-Albert Graph:")
    print(f"Number of nodes: {xx.number_of_nodes(ba_graph)}")
    print(f"Number of edges: {xx.number_of_edges(ba_graph)}")

    # Stochastic Block Model
    sbm_graph = create_graph('sbm', {'n': 100, 'sizes': [50, 50], 'probs': [[0.8, 0.2], [0.2, 0.8]]})
    print("\nStochastic Block Model Graph:")
    print(f"Number of nodes: {nx.number_of_nodes(sbm_graph)}")
    print(f"Number of edges: {nx.number_of_edges(sbm_graph)}")
'''