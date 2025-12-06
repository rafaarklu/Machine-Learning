import numpy as np

# Grafo representado como dicionário: cada nó tem sua lista de links de saída
graph = {
    "A": ["B", "C"],
    "B": ["C"],
    "C": ["A", "D"],
    "D": ["C"]
}

nodes = list(graph.keys())
n = len(nodes)

# Parâmetros
d = 0.85   # damping factor
epsilon = 1e-6  # critério de convergência

# Inicialização: todos começam com 1/n
pagerank = {node: 1/n for node in nodes}

converged = False

while not converged:
    new_pr = {}

    for node in nodes:
        # Teletransporte
        pr_value = (1 - d) / n

        # Somatório das contribuições
        for other in nodes:
            if node in graph[other]:  # se other aponta para node
                pr_value += d * (pagerank[other] / len(graph[other]))

        new_pr[node] = pr_value

    # Verificar convergência
    diff = sum(abs(new_pr[node] - pagerank[node]) for node in nodes)

    pagerank = new_pr

    if diff < epsilon:
        converged = True

print("PageRank Convergido:")
for node, pr in pagerank.items():
    print(node, round(pr, 4))




print("\nComparação com NetworkX:")




import networkx as nx

G = nx.DiGraph()

for node, links in graph.items():
    for t in links:
        G.add_edge(node, t)

nx_pr = nx.pagerank(G, alpha=0.85)

print("PageRank NetworkX:")
for node, pr in nx_pr.items():
    print(node, round(pr, 4))
