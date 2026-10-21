"""
estatisticas_redes.py
======================
Funções para extrair estatísticas de qualquer uma das redes (R0..R4),
com foco especial na pergunta da atividade:

    "Nós com a mesma cor (risk level) estão mais conectados entre si do
     que com nós de cores diferentes? Responda para cada cor."

MÉTRICAS USADAS PARA RESPONDER ISSO (explicação para a apresentação):

1. RAZÃO INTRA OBSERVADA (por cor c):
   De todas as arestas que tocam algum nó da cor c, qual fração liga DOIS
   nós da cor c (em vez de ligar c a outra cor)?
       razao_intra(c) = arestas_dentro_de_c / (arestas_dentro_de_c + arestas_de_c_para_fora)

2. RAZÃO INTRA ESPERADA (linha de base "sem padrão", por cor c):
   Se as arestas se formassem AO ACASO (sem nenhuma preferência por cor),
   qual fração dos vizinhos de um nó da cor c seria, em média, também da
   cor c? Resposta: a PROPORÇÃO que a cor c representa no total de nós.
       razao_esperada(c) = (nº de nós da cor c) / (nº total de nós)

3. ÍNDICE DE HOMOFILIA (por cor c):
       indice(c) = razao_intra(c) / razao_esperada(c)
   > 1  -> a cor c se conecta com ela mesma MAIS do que seria esperado ao
           acaso (homofilia: "farinha do mesmo saco")
   = 1  -> mistura aleatória, sem padrão
   < 1  -> a cor c se conecta com ela mesma MENOS do que o esperado
           (heterofilia: prefere se ligar a outras cores)

4. CONTRIBUIÇÃO DE MODULARIDADE (por cor c):
   Métrica padrão da ciência de redes (Newman) para dizer se um grupo
   forma uma "comunidade" de verdade. Positivo = comunidade coesa
   (mais arestas internas do que o esperado ao acaso); perto de zero ou
   negativo = grupo não forma comunidade.
       Q(c) = (L_c / L) - (k_c / (2L))^2
   onde L_c = arestas dentro de c, L = total de arestas no grafo,
   k_c = soma dos graus dos nós de c.

5. ASSORTATIVIDADE (UM número para o grafo INTEIRO, não por cor):
   nx.attribute_assortativity_coefficient -- versão "oficial" do
   NetworkX do mesmo conceito, agregando todas as cores num único
   coeficiente entre -1 e 1 (mesma leitura do índice de homofilia, mas
   sem quebrar por categoria).
"""

import networkx as nx
import pandas as pd


def resumo_basico(G):
    """Estatísticas gerais do grafo (tamanho, densidade, componentes, clustering)."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    componentes = list(nx.connected_components(G)) if n > 0 else []
    maior_componente = max((len(c) for c in componentes), default=0)

    return {
        'n_nos': n,
        'n_arestas': m,
        'densidade': nx.density(G) if n > 1 else 0.0,
        'grau_medio': (2 * m / n) if n > 0 else 0.0,
        'n_componentes': len(componentes),
        'tamanho_maior_componente': maior_componente,
        'clustering_medio': nx.average_clustering(G) if n > 0 else 0.0,
    }


def distribuicao_por_categoria(G, atributo='risk'):
    """Grau médio e clustering médio, quebrados por categoria (cor)."""
    categorias = set(nx.get_node_attributes(G, atributo).values())
    resultado = {}
    graus = dict(G.degree())
    clustering = nx.clustering(G)

    for c in categorias:
        nos_c = [n for n, d in G.nodes(data=True) if d.get(atributo) == c]
        if not nos_c:
            continue
        resultado[c] = {
            'n_nos': len(nos_c),
            'grau_medio': sum(graus[n] for n in nos_c) / len(nos_c),
            'clustering_medio': sum(clustering[n] for n in nos_c) / len(nos_c),
        }
    return resultado


def homofilia_por_categoria(G, atributo='risk'):
    """
    Núcleo da análise: para cada categoria (cor), calcula razão intra
    observada vs esperada, o índice de homofilia e a contribuição de
    modularidade. Ver explicação completa no docstring do módulo.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0 or m == 0:
        return {}

    atributos = nx.get_node_attributes(G, atributo)
    categorias = set(atributos.values())
    graus = dict(G.degree())

    resultado = {}
    for c in categorias:
        nos_c = [n for n, v in atributos.items() if v == c]
        n_c = len(nos_c)
        if n_c == 0:
            continue

        # arestas internas (dois nós da mesma cor c) e externas (um dos
        # dois nós é da cor c, o outro não)
        internas = 0
        externas = 0
        for u, v in G.edges():
            cor_u, cor_v = atributos.get(u), atributos.get(v)
            if cor_u == c and cor_v == c:
                internas += 1
            elif cor_u == c or cor_v == c:
                externas += 1

        total_tocando_c = internas + externas
        razao_intra_obs = (internas / total_tocando_c) if total_tocando_c > 0 else 0.0
        razao_intra_esp = n_c / n
        indice_homofilia = (razao_intra_obs / razao_intra_esp) if razao_intra_esp > 0 else float('nan')

        k_c = sum(graus[node] for node in nos_c)
        contrib_modularidade = (internas / m) - (k_c / (2 * m)) ** 2

        resultado[c] = {
            'n_nos': n_c,
            'arestas_internas': internas,
            'arestas_externas_tocando_c': externas,
            'razao_intra_observada': razao_intra_obs,
            'razao_intra_esperada_ao_acaso': razao_intra_esp,
            'indice_homofilia': indice_homofilia,
            'contribuicao_modularidade': contrib_modularidade,
        }
    return resultado


def assortatividade(G, atributo='risk'):
    """Coeficiente único (-1 a 1) para o grafo inteiro. None se não for calculável."""
    try:
        return nx.attribute_assortativity_coefficient(G, atributo)
    except Exception:
        return None


def modularidade_total(G, atributo='risk'):
    """Soma das contribuições de modularidade de todas as categorias = modularidade total da partição por cor."""
    homof = homofilia_por_categoria(G, atributo)
    return sum(v['contribuicao_modularidade'] for v in homof.values())


def relatorio_completo(G, nome_rede, atributo='risk', imprimir=True):
    """Roda todas as análises acima e (opcionalmente) imprime um relatório formatado."""
    if G is None:
        if imprimir:
            print(f"\n=== {nome_rede}: rede não construída (dependência ausente) ===")
        return None

    basico = resumo_basico(G)
    por_categoria = distribuicao_por_categoria(G, atributo)
    homof = homofilia_por_categoria(G, atributo)
    assort = assortatividade(G, atributo)
    modularidade = modularidade_total(G, atributo)

    if imprimir:
        print(f"\n{'=' * 70}\n{nome_rede}\n{'=' * 70}")
        print(f"Nós: {basico['n_nos']} | Arestas: {basico['n_arestas']} | "
              f"Densidade: {basico['densidade']:.5f}")
        print(f"Grau médio: {basico['grau_medio']:.2f} | "
              f"Componentes conexos: {basico['n_componentes']} "
              f"(maior: {basico['tamanho_maior_componente']} nós) | "
              f"Clustering médio: {basico['clustering_medio']:.4f}")
        print(f"Assortatividade por '{atributo}' (grafo inteiro): "
              f"{assort:.4f}" if assort is not None else "Assortatividade: não calculável")
        print(f"Modularidade total da partição por '{atributo}': {modularidade:.4f}")

        print(f"\n{'Cor':<12}{'N nós':>7}{'Grau méd':>10}{'Intra obs':>11}{'Intra esp':>11}{'Índice homof.':>15}{'Contrib. modul.':>17}")
        for cor, stats in homof.items():
            print(
                f"{cor:<12}{stats['n_nos']:>7}"
                f"{por_categoria.get(cor, {}).get('grau_medio', 0):>10.2f}"
                f"{stats['razao_intra_observada']:>11.3f}"
                f"{stats['razao_intra_esperada_ao_acaso']:>11.3f}"
                f"{stats['indice_homofilia']:>15.2f}"
                f"{stats['contribuicao_modularidade']:>17.4f}"
            )

    return {
        'rede': nome_rede,
        **basico,
        'assortatividade': assort,
        'modularidade_total': modularidade,
        'homofilia_por_categoria': homof,
        'distribuicao_por_categoria': por_categoria,
    }


def tabela_comparativa(relatorios):
    """Recebe uma lista de dicts (saída de relatorio_completo) e monta um DataFrame resumo."""
    linhas = []
    for r in relatorios:
        if r is None:
            continue
        linhas.append({
            'rede': r['rede'],
            'n_nos': r['n_nos'],
            'n_arestas': r['n_arestas'],
            'densidade': round(r['densidade'], 5),
            'grau_medio': round(r['grau_medio'], 2),
            'n_componentes': r['n_componentes'],
            'clustering_medio': round(r['clustering_medio'], 4),
            'assortatividade': round(r['assortatividade'], 4) if r['assortatividade'] is not None else None,
            'modularidade_total': round(r['modularidade_total'], 4),
        })
    return pd.DataFrame(linhas)
