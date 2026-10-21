import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------
# 1. Carregamento e normalização dos dados
# ---------------------------------------------------------------------
from pathlib import Path

# Caminho ABSOLUTO baseado na localização deste script -- funciona
# independente de qual seja o diretório atual (cwd) do terminal.
caminho_arquivo = Path(__file__).resolve().parent / 'train_suicide_2026.xlsx'
df = pd.read_excel(caminho_arquivo)

# Normalização da coluna de risco (mesma correção do código anterior:
# tira espaços extras e padroniza maiúsculas/minúsculas)
df['suicide risk'] = (
    df['suicide risk'].astype(str).str.strip().str.lower().str.capitalize()
)

# ---------------------------------------------------------------------
# 2. Geração dos embeddings
# ---------------------------------------------------------------------
# 'all-MiniLM-L6-v2' é um modelo leve e rápido, bom equilíbrio entre
# qualidade semântica e velocidade — ótimo para datasets de ~1.500 posts.
modelo = SentenceTransformer('all-MiniLM-L6-v2')

# .encode() transforma cada post em um vetor numérico.
# show_progress_bar ajuda a acompanhar o processamento (pode demorar
# alguns minutos dependendo do hardware).
embeddings = modelo.encode(
    df['post'].astype(str).tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

# ---------------------------------------------------------------------
# 3. Matriz de similaridade de cosseno
# ---------------------------------------------------------------------
# matriz[i][j] = similaridade semântica entre o post i e o post j
matriz_similaridade = cosine_similarity(embeddings)

# ---------------------------------------------------------------------
# 4. Construção do grafo
# ---------------------------------------------------------------------
G = nx.Graph()

for idx, row in df.iterrows():
    G.add_node(row['row_id'], risk=row['suicide risk'], post_text=row['post'])

# LIMIAR DE SIMILARIDADE: diferente da contagem de palavras (que é um
# número inteiro sem limite superior claro), a similaridade de cosseno
# é sempre entre -1 e 1, o que facilita escolher um limiar interpretável.
# 0.5-0.6 costuma ser um bom ponto de partida para o modelo MiniLM;
# ajuste observando a densidade do grafo resultante.
LIMIAR_SIMILARIDADE = 0.55

n = len(df)
row_ids = df['row_id'].tolist()

for i in range(n):
    for j in range(i + 1, n):
        similaridade = matriz_similaridade[i][j]
        if similaridade > LIMIAR_SIMILARIDADE:
            G.add_edge(row_ids[i], row_ids[j], weight=float(similaridade))

# ---------------------------------------------------------------------
# 5. Mapeamento de cores (4 categorias fixas de suicide risk)
# ---------------------------------------------------------------------
ordem_categorias = ['Ideation', 'Indicator', 'Behavior', 'Attempt']
paleta_cores = ['#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
niveis_presentes = [c for c in ordem_categorias if c in df['suicide risk'].unique()]
mapa_cores = {risco: paleta_cores[i] for i, risco in enumerate(niveis_presentes)}
cores_dos_nos = [mapa_cores.get(G.nodes[n_id]['risk'], 'grey') for n_id in G.nodes()]

# ---------------------------------------------------------------------
# 6. Visualização
# ---------------------------------------------------------------------
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=42, k=0.15)

nx.draw_networkx_nodes(G, pos, node_color=cores_dos_nos, node_size=60, alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
plt.axis('off')
plt.title(
    f"Rede de Posts — Similaridade Semântica (cosseno > {LIMIAR_SIMILARIDADE})",
    fontsize=16
)

elementos_legenda = [
    Line2D([0], [0], marker='o', color='w', label=risco,
           markerfacecolor=cor, markersize=12)
    for risco, cor in mapa_cores.items()
]
plt.legend(handles=elementos_legenda, title="Suicide Risk", loc='best', fontsize=10)
plt.tight_layout()
plt.savefig('rede_1_embeddings_semanticos.png', dpi=150)
plt.show()

print(f"Total de Nós (Posts): {G.number_of_nodes()}")
print(f"Total de Arestas (similaridade semântica > {LIMIAR_SIMILARIDADE}): {G.number_of_edges()}")
