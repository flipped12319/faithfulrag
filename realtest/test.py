import json
import networkx as nx

G = nx.MultiDiGraph()  # 允许同一对节点多条边
triples = [
    # 核心实体
    {
        "subject": "Arthur's Magazine",
        "relation": "type",
        "object": "American literary periodical",
        "chunk_id": "c1"
    },
    {
        "subject": "Arthur's Magazine",
        "relation": "publication_start",
        "object": "1844",
        "chunk_id": "c1"
    },
    {
        "subject": "Arthur's Magazine",
        "relation": "publication_end",
        "object": "1846",
        "chunk_id": "c1"
    },
    {
        "subject": "Arthur's Magazine",
        "relation": "location",
        "object": "Philadelphia",
        "chunk_id": "c1"
    },

    # 时间推理链
    {
        "subject": "1844",
        "relation": "in_century",
        "object": "19th century",
        "chunk_id": "c2"
    },
    {
        "subject": "1846",
        "relation": "in_century",
        "object": "19th century",
        "chunk_id": "c2"
    },

    # 城市补充知识
    {
        "subject": "Philadelphia",
        "relation": "located_in",
        "object": "United States",
        "chunk_id": "c3"
    }
]


for value in triples:
    print(value)
    print(value["subject"])



for t in triples:
    G.add_node(
        t["subject"],
        entity_id=None
    )
    G.add_node(
        t["object"],
    )
    G.add_edge(
        t["subject"],
        t["object"],
        relation=t["relation"],

    )

# 示例：查看所有从 Arthur's Magazine 出发的边
for u, v, data in G.edges(data=True):
    print(u, "--", data["relation"], "-->", v, )

def get_k_hop_subgraph(G, seed_nodes, k=1):
    """
    G: networkx graph (MultiDiGraph)
    seed_nodes: iterable of node ids
    k: hop number
    """
    visited = set(seed_nodes)
    frontier = set(seed_nodes)

    for _ in range(k):
        next_frontier = set()
        for node in frontier:
            # 出边邻居
            next_frontier.update(G.successors(node))
            # 入边邻居
            next_frontier.update(G.predecessors(node))
        # 去掉已经访问过的
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier

    return G.subgraph(visited).copy()
seed_nodes = ["Arthur's Magazine"]
subG_1 = get_k_hop_subgraph(G, seed_nodes, k=2)
for u, v, data in subG_1.edges(data=True):
    print(u, "--", data["relation"], "-->", v, )

