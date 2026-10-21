"""
analise_completa.py
=====================
Roda a atividade técnica completa:

  PARTE 1 - Constrói R4 (rede por usuário, baseada na sequência de posts)
            e faz uma análise extra só possível com essa rede: a
            trajetória de risco de cada usuário ao longo dos posts.

  PARTE 2 - Roda as estatísticas (com foco em homofilia por cor) em
            TODAS as redes disponíveis (R0..R4) e monta uma tabela
            comparativa.

Redes que exigem pacotes não instalados no ambiente de teste (nltk para
R0, sentence-transformers para R1) são puladas automaticamente com um
aviso -- rode este script na sua máquina, onde ambos já estão instalados,
para ver o conjunto completo.
"""

from redes_builders import (
    carregar_dados,
    construir_R0_palavras,
    construir_R1_embeddings,
    construir_R2_5_projetado,
    construir_R3_evidencia,
    construir_R4_usuario,
)
from estatisticas_redes import relatorio_completo, tabela_comparativa

import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------
df = carregar_dados()  # sem argumento -> acha a planilha automaticamente na mesma pasta

# ---------------------------------------------------------------------
# PARTE 1 - Rede por usuário (R4) + análise de trajetória de risco
# ---------------------------------------------------------------------
R4 = construir_R4_usuario(df)


def analisar_trajetoria_risco(df):
    """
    Pergunta: dentro de um mesmo usuário, o nível de risco tende a
    AUMENTAR ao longo da sequência de posts (do primeiro pro último)?

    Mapeamos as 4 categorias numa escala ordinal de severidade clínica
    crescente (Indicator < Ideation < Behavior < Attempt) e medimos:

      (a) correlação de Spearman entre post_id (ordem) e severidade,
          calculada DENTRO de cada usuário e depois resumida (média);
      (b) taxa de "escalada": fração de usuários cujo ÚLTIMO post tem
          severidade >= a do PRIMEIRO post.
    """
    severidade = {'Indicator': 1, 'Ideation': 2, 'Behavior': 3, 'Attempt': 4}
    df = df.copy()
    df['severidade'] = df['suicide risk'].map(severidade)

    correlacoes = []
    escaladas = 0
    usuarios_validos = 0

    for user_id, grupo in df.groupby('anon_user_id'):
        grupo = grupo.sort_values('post_id')
        if len(grupo) < 3:
            continue  # correlação de Spearman precisa de variação mínima
        usuarios_validos += 1

        rho, _ = spearmanr(grupo['post_id'], grupo['severidade'])
        if pd.notna(rho):
            correlacoes.append(rho)

        primeiro = grupo['severidade'].iloc[0]
        ultimo = grupo['severidade'].iloc[-1]
        if ultimo >= primeiro:
            escaladas += 1

    print(f"\n{'=' * 70}\nTRAJETÓRIA DE RISCO POR USUÁRIO (só possível com R4)\n{'=' * 70}")
    print(f"Usuários analisados (com >= 3 posts): {usuarios_validos}")
    print(f"Correlação de Spearman média (post_id x severidade): "
          f"{sum(correlacoes) / len(correlacoes):.3f}" if correlacoes else "N/A")
    print(f"  -> positivo = risco tende a subir ao longo dos posts; "
          f"negativo = tende a cair; perto de 0 = sem tendência clara")
    print(f"Usuários cujo último post é >= severo que o primeiro: "
          f"{escaladas}/{usuarios_validos} ({100 * escaladas / usuarios_validos:.1f}%)")


analisar_trajetoria_risco(df)

# ---------------------------------------------------------------------
# PARTE 2 - Estatísticas (incl. homofilia por cor) em todas as redes
# ---------------------------------------------------------------------
R0 = construir_R0_palavras(df, X=10)
R1 = construir_R1_embeddings(df, limiar=0.55)
R2_5 = construir_R2_5_projetado(df, min_fatores=3)
R3 = construir_R3_evidencia(df, k_fatores=3)

relatorios = [
    relatorio_completo(R0, "R0 - Palavras em comum (> 10)"),
    relatorio_completo(R1, "R1 - Similaridade semântica (cosseno > 0.55)"),
    relatorio_completo(R2_5, "R2.5 - Fatores em comum (>= 3), projeção do bipartido"),
    relatorio_completo(R3, "R3 - Nós = evidência, arestas = fatores em comum (>= 3)"),
    relatorio_completo(R4, "R4 - Sequência de posts por usuário"),
]

print(f"\n\n{'=' * 70}\nTABELA COMPARATIVA\n{'=' * 70}")
tabela = tabela_comparativa(relatorios)
print(tabela.to_string(index=False))
tabela.to_csv('comparativo_redes.csv', index=False)
print("\n(tabela também salva em comparativo_redes.csv)")
