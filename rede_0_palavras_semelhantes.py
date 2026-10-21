import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import itertools
from matplotlib.lines import Line2D

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)


def processar_texto(texto):
    if not isinstance(texto, str):
        return set()
    tokens = word_tokenize(texto.lower())
    return {
        w for w in tokens
        if w not in stop_words and w not in punctuation and w.isalpha()
    }


from pathlib import Path

# Caminho ABSOLUTO baseado na localização deste script -- funciona
# independente de qual seja o diretório atual (cwd) do terminal.
caminho_arquivo = Path(__file__).resolve().parent / 'train_suicide_2026.xlsx'
df = pd.read_excel(caminho_arquivo)

# A coluna "suicide risk" tem variações como "Ideation", "ideation", "Ideation "
# o que fazia o código antigo tratar a mesma categoria como se fossem várias.
#  Normalizar ANTES de usar a coluna
df['suicide risk'] = (
    df['suicide risk']
    .astype(str)
    .str.strip()
    .str.lower()
    .str.capitalize()   # "ideation " -> "ideation" -> "Ideation"
)
# Resultado: apenas 4 categorias -> Ideation, Attempt, Behavior, Indicator

df['conjunto_palavras'] = df['post'].apply(processar_texto)

# 3. Inicialização do Grafo
G = nx.Graph()

# 4. Nós
for index, row in df.iterrows():
    node_id = row['row_id']
    risco = row['suicide risk']  # já normalizado
    texto_post = row['post']
    G.add_node(node_id, risk=risco, post_text=texto_post)

# 5. Arestas (X > 10 palavras em comum)
X = 10
for (i1, row1), (i2, row2) in itertools.combinations(df.iterrows(), 2):
    palavras_em_comum = row1['conjunto_palavras'].intersection(row2['conjunto_palavras'])
    if len(palavras_em_comum) > X:
        G.add_edge(row1['row_id'], row2['row_id'], weight=len(palavras_em_comum))

# 6. Mapeamento de Cores
ordem_categorias = ['Ideation', 'Indicator', 'Behavior', 'Attempt']
paleta_cores = ['#2ca02c', '#ff7f0e', '#9467bd', '#d62728']  

niveis_risco_presentes = [c for c in ordem_categorias if c in df['suicide risk'].unique()]
mapa_cores = {risco: paleta_cores[i] for i, risco in enumerate(niveis_risco_presentes)}

cores_dos_nos = [mapa_cores.get(G.nodes[n]['risk'], 'grey') for n in G.nodes()]

# 7. Visualização da Rede
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=42, k=0.15)

nx.draw_networkx_nodes(G, pos, node_color=cores_dos_nos, node_size=60, alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')

plt.axis('off')
plt.title(f"Rede de Semelhança de Posts (Conexão para > {X} palavras iguais)", fontsize=16)

elementos_legenda = [
    Line2D([0], [0], marker='o', color='w', label=risco,
           markerfacecolor=cor, markersize=12)
    for risco, cor in mapa_cores.items()
]
plt.legend(handles=elementos_legenda, title="Suicide Risk", loc='best', fontsize=10)

plt.tight_layout()
plt.show()

print(f"Total de Nós (Posts): {G.number_of_nodes()}")
print(f"Total de Arestas (Conexões > {X} palavras): {G.number_of_edges()}")
