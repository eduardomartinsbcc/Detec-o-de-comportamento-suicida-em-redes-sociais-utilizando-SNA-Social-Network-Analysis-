

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ast
import itertools

# ---------------------------------------------------------------------
# 1. Carregamento e normalização
# ---------------------------------------------------------------------
from pathlib import Path

# Caminho ABSOLUTO baseado na localização deste script -- funciona
# independente de qual seja o diretório atual (cwd) do terminal.
caminho_arquivo = Path(__file__).resolve().parent / 'train_suicide_2026.xlsx'
df = pd.read_excel(caminho_arquivo)

df['suicide risk'] = (
    df['suicide risk'].astype(str).str.strip().str.lower().str.capitalize()
)

# Descarta posts sem evidência textual preenchida (não temos "identidade"
# de nó pra eles nesta rede específica)
df = df.dropna(subset=['evidence for suicide risk level']).reset_index(drop=True)


def parse_fatores(texto):
    if not isinstance(texto, str):
        return set()
    try:
        return set(ast.literal_eval(texto))
    except (ValueError, SyntaxError):
        return set()


df['fatores_set'] = df['factors'].apply(parse_fatores)

# ---------------------------------------------------------------------
# 2. Construção do grafo
# ---------------------------------------------------------------------
G = nx.Graph()

# O NÓ carrega a evidência como identidade (o que ele "diz"), mas mantemos
# row_id como chave única do nó (duas linhas podem ter evidência parecida,
# mas são posts diferentes)
for idx, row in df.iterrows():
    G.add_node(
        row['row_id'],
        evidence=row['evidence for suicide risk level'],
        risk=row['suicide risk']
    )

# A ARESTA vem inteiramente da coluna 'factors' — note que o texto de
# evidência NÃO participa dessa conta, só serve pra rotular o nó depois
K_FATORES_MINIMO = 3
row_ids = df['row_id'].tolist()
fatores_sets = df['fatores_set'].tolist()

for i, j in itertools.combinations(range(len(df)), 2):
    em_comum = fatores_sets[i] & fatores_sets[j]
    if len(em_comum) >= K_FATORES_MINIMO:
        G.add_edge(row_ids[i], row_ids[j], weight=len(em_comum), fatores_comuns=em_comum)

# ---------------------------------------------------------------------
# 3. Checagem de interpretabilidade (ótimo para a apresentação):
# imprime alguns pares conectados mostrando a EVIDÊNCIA de cada lado e
# os FATORES que os uniram — evidencia visualmente a lógica "nó != aresta"
# ---------------------------------------------------------------------
print("Exemplos de conexões (evidência de cada nó + fatores que os uniram):\n")
for u, v, dados in list(G.edges(data=True))[:5]:
    ev_u = str(G.nodes[u]['evidence'])[:80]
    ev_v = str(G.nodes[v]['evidence'])[:80]
    print(f"  [{u}] \"{ev_u}...\"")
    print(f"  [{v}] \"{ev_v}...\"")
    print(f"  -> conectados por {dados['weight']} fatores em comum: {dados['fatores_comuns']}\n")

# ---------------------------------------------------------------------
# 4. Cores por suicide risk
# ---------------------------------------------------------------------
ordem_categorias = ['Ideation', 'Indicator', 'Behavior', 'Attempt']
paleta_cores = ['#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
niveis_presentes = [c for c in ordem_categorias if c in df['suicide risk'].unique()]
mapa_cores = {risco: paleta_cores[i] for i, risco in enumerate(niveis_presentes)}
cores_dos_nos = [mapa_cores.get(G.nodes[n]['risk'], 'grey') for n in G.nodes()]

# ---------------------------------------------------------------------
# 5. Visualização
# ---------------------------------------------------------------------
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=42, k=0.15)

pesos = [G[u][v]['weight'] * 0.4 for u, v in G.edges()]

nx.draw_networkx_nodes(G, pos, node_color=cores_dos_nos, node_size=60, alpha=0.8)
nx.draw_networkx_edges(G, pos, width=pesos, alpha=0.2, edge_color='gray')
plt.axis('off')
plt.title(
    f"Nós = Evidência Textual | Arestas = Fatores de Risco em Comum (>= {K_FATORES_MINIMO})",
    fontsize=14
)

elementos_legenda = [
    Line2D([0], [0], marker='o', color='w', label=risco,
           markerfacecolor=cor, markersize=12)
    for risco, cor in mapa_cores.items()
]
plt.legend(handles=elementos_legenda, title="Suicide Risk", loc='best', fontsize=10)
plt.tight_layout()
plt.savefig('rede_3_evidence_factors.png', dpi=150)
plt.show()

print(f"Total de Nós (posts com evidência preenchida): {G.number_of_nodes()}")
print(f"Total de Arestas (>= {K_FATORES_MINIMO} fatores em comum): {G.number_of_edges()}")
