# Page Rank

O PageRank é um algoritmo que mede a importância relativa de cada nó dentro de um grafo. A importância de um nó é determinada pela quantidade e qualidade dos links que apontam para ele.

Os resultados apresentados na imagem e gerados código a seguir mostram os valores finais de PageRank para os nós A, B, C e D.


``` python exec="off"

graph = {
    "A": ["B", "C"],
    "B": ["C"],
    "C": ["A", "D"],
    "D": ["C"]
}

```

---

## Codigo do Page Rank

=== "output"

    ``` python exec="on" html="1"
    --8<-- "./docs/pagerank/pgrank.py"
    ```


=== "code"

    ``` python exec="off" 
    --8<-- "./docs/pagerank/pgrank.py"
    ```

# Interpretação
Os resultados de PageRank Convergido e PageRank NetworkX são idênticos, o que confirma a correta implementação e convergência do algoritmo manual em relação à biblioteca padrão (NetworkX).




O valor de PageRank de um nó pode ser interpretado como a probabilidade de um "navegador aleatório" (o modelo subjacente ao algoritmo) estar naquele nó em um determinado momento. Quanto maior o valor, mais importante é o nó dentro da estrutura do grafo.

 
## Quais são os nós mais importantes?

 - Nó C: O Mais Importante 
    - PageRank: 0.4292 (Mais Alto)
    - O nó C possui o maior PageRank, sendo considerado o mais importante da rede.Justificativa no Grafo: O nó C recebe links de todos os outros três nós (A -> C, B -> C, D -> C). Mesmo que os nós A e D recebam a mesma quantidade de PageRank de C, a contribuição de A, B e D para C o torna o centro de influência da rede.


 - Nós A e D: Importância Intermediária/Igual
    - PageRank: 0.2199 (Idêntico)
    - Os nós A e D têm a mesma importância no grafo, ocupando a segunda posição.Justificativa no Grafo: Ambos recebem PageRank apenas do nó C (C -> A e C -> D). Como o nó C divide seu PageRank uniformemente entre A e D, e não há outras fontes de entrada para A ou D, eles terminam com o mesmo valor de importância.

 - Nó B: O Menos Importante
    - PageRank: 0.131 (Mais Baixo)
    - O nó B tem o valor de PageRank mais baixo.Justificativa no Grafo: O nó B recebe links apenas do nó A (A -> B). Como A também aponta para C, o PageRank de A é dividido. Além disso, B não recebe nenhuma outra fonte de entrada. A baixa entrada de PageRank faz com que ele seja o nó com menor relevância.