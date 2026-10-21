

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from networkx.algorithms import bipartite
import ast

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


def parse_fatores(texto):
    """
    A coluna 'factors' vem como uma STRING que parece uma lista Python,
    ex: "['hopelessness', 'coping strategy', 'hopelessness']".
    ast.literal_eval converte essa string de volta para uma lista real.
    Usamos set() para remover repetições (um mesmo fator pode aparecer
    várias vezes anotado no mesmo post).
    """
    if not isinstance(texto, str):
        return set()
    try:
        return set(ast.literal_eval(texto))
    except (ValueError, SyntaxError):
        return set()


df['fatores_set'] = df['factors'].apply(parse_fatores)

# ---------------------------------------------------------------------
# 2. Construção do grafo BIPARTIDO
# ---------------------------------------------------------------------
B = nx.Graph()

# Nós de post: guardamos bipartite=0 para identificar o "lado" do post
for idx, row in df.iterrows():
    B.add_node(row['row_id'], bipartite=0, tipo='post', risk=row['suicide risk'])

# Nós de fator: bipartite=1 identifica o "lado" do fator.
# Um mesmo fator (ex: "hopelessness") aparece em vários posts, mas existe
# como UM ÚNICO nó no grafo — é isso que faz a "ponte" entre posts.
todos_fatores = set()
for fatores in df['fatores_set']:
    todos_fatores.update(fatores)

for fator in todos_fatores:
    B.add_node(fator, bipartite=1, tipo='fator')

# Arestas post -> fator
for idx, row in df.iterrows():
    for fator in row['fatores_set']:
        B.add_edge(row['row_id'], fator)

print(f"Grafo bipartido: {sum(1 for _, d in B.nodes(data=True) if d['tipo']=='post')} nós de post, "
      f"{sum(1 for _, d in B.nodes(data=True) if d['tipo']=='fator')} nós de fator, "
      f"{B.number_of_edges()} arestas post-fator")

# ---------------------------------------------------------------------
# 3. Visualização do grafo BIPARTIDO
# ---------------------------------------------------------------------
# Fatores geralmente são muitos menos que posts (dezenas vs milhares),
# então plotamos com layout separado por "camadas" para ficar legível.
nos_post = [n for n, d in B.nodes(data=True) if d['tipo'] == 'post']
nos_fator = [n for n, d in B.nodes(data=True) if d['tipo'] == 'fator']

pos_bipartido = {}
pos_bipartido.update({n: (0, i) for i, n in enumerate(nos_post)})
pos_bipartido.update({n: (1, i * (len(nos_post) / max(len(nos_fator), 1))) for i, n in enumerate(nos_fator)})

plt.figure(figsize=(10, 14))
nx.draw_networkx_nodes(B, pos_bipartido, nodelist=nos_post, node_color='lightsteelblue',
                        node_size=8, alpha=0.5, label='Posts')
nx.draw_networkx_nodes(B, pos_bipartido, nodelist=nos_fator, node_color='crimson',
                        node_size=250, alpha=0.9, label='Fatores')
nx.draw_networkx_labels(B, pos_bipartido, labels={n: n for n in nos_fator}, font_size=7)
nx.draw_networkx_edges(B, pos_bipartido, alpha=0.05)
plt.axis('off')
plt.title("Grafo Bipartido: Posts <-> Fatores de Risco", fontsize=15)
plt.legend(scatterpoints=1, loc='upper right')
plt.tight_layout()
plt.savefig('rede_2a_bipartido.png', dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 4. Projeção ponderada (só entre posts)
# ---------------------------------------------------------------------
# networkx já tem uma função pronta pra isso: bipartite.weighted_projected_graph
# Ela cria uma aresta entre dois nós do MESMO lado (aqui, posts) sempre que
# eles compartilham >= 1 vizinho do outro lado (aqui, fatores), e o peso da
# aresta = número de vizinhos (fatores) em comum.
G_projetado = bipartite.weighted_projected_graph(B, nos_post)

# Filtro opcional: exigir um número MÍNIMO de fatores em comum para a
# aresta aparecer na visualização (equivalente ao "X > 10" das redes
# anteriores, mas aqui X é "número de fatores de risco compartilhados").
MIN_FATORES_EM_COMUM = 3
arestas_para_remover = [
    (u, v) for u, v, d in G_projetado.edges(data=True)
    if d['weight'] < MIN_FATORES_EM_COMUM
]
G_projetado.remove_edges_from(arestas_para_remover)

# ---------------------------------------------------------------------
# 5. Cores por suicide risk (mesmo esquema fixo das redes anteriores)
# ---------------------------------------------------------------------
ordem_categorias = ['Ideation', 'Indicator', 'Behavior', 'Attempt']
paleta_cores = ['#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
niveis_presentes = [c for c in ordem_categorias if c in df['suicide risk'].unique()]
mapa_cores = {risco: paleta_cores[i] for i, risco in enumerate(niveis_presentes)}

risco_por_post = dict(zip(df['row_id'], df['suicide risk']))
cores_dos_nos = [mapa_cores.get(risco_por_post.get(n), 'grey') for n in G_projetado.nodes()]

# ---------------------------------------------------------------------
# 6. Visualização da rede PROJETADA
# ---------------------------------------------------------------------
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G_projetado, seed=42, k=0.15)

# Espessura da aresta proporcional ao peso (número de fatores em comum)
pesos = [G_projetado[u][v]['weight'] * 0.3 for u, v in G_projetado.edges()]

nx.draw_networkx_nodes(G_projetado, pos, node_color=cores_dos_nos, node_size=60, alpha=0.8)
nx.draw_networkx_edges(G_projetado, pos, width=pesos, alpha=0.2, edge_color='gray')
plt.axis('off')
plt.title(
    f"Rede de Posts Projetada — Fatores de Risco em Comum (>= {MIN_FATORES_EM_COMUM})",
    fontsize=16
)

elementos_legenda = [
    Line2D([0], [0], marker='o', color='w', label=risco,
           markerfacecolor=cor, markersize=12)
    for risco, cor in mapa_cores.items()
]
plt.legend(handles=elementos_legenda, title="Suicide Risk", loc='best', fontsize=10)
plt.tight_layout()
plt.savefig('rede_2b_projetada.png', dpi=150)
plt.show()

print(f"Total de Nós (Posts) na rede projetada: {G_projetado.number_of_nodes()}")
print(f"Total de Arestas (>= {MIN_FATORES_EM_COMUM} fatores em comum): {G_projetado.number_of_edges()}")
