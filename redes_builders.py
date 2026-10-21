#import os
from redes_builders import carregar_dados, construir_R4_usuario
import networkx as nx

# 1. Carrega os dados
caminho_planilha = '/content/train_suicide_2026.xlsx'
df_r4 = carregar_dados(caminho_planilha if os.path.exists(caminho_planilha) else None)

# 2. Constrói exclusivamente a rede R4
R4 = construir_R4_usuario(df_r4)

# 3. Exibe informações básicas da rede gerada
print(f"Rede R4 construída com sucesso!")
print(f"Número de nós (posts): {R4.number_of_nodes()}")
print(f"Número de arestas (sequências temporais): {R4.number_of_edges()}")
print(f"Número de componentes conexos (usuários): {nx.number_connected_components(R4)}")