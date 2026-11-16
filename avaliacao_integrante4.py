"""Integrante 4: avaliação e comparação dos classificadores.

Lê os arquivos de predição de cada modelo (formato comum, ver README) e gera:
  - matrizes de confusão de todos os modelos lado a lado (matrizes_comparacao.png)
  - métricas por classe (precisão, sensibilidade, especificidade, F1) e globais
  - visão de triagem (doente x normal): sensibilidade para "qualquer doença"
  - intervalos de confiança de 95% por bootstrap (o teste é pequeno)
  - matriz de confusão de cada modelo (contagens e % por linha)
  - gráfico comparativo e teste de McNemar entre pares de modelos
  - análise de erros por base de origem (MC = Montgomery, SZ = Shenzhen, NIH)
  - normal x doença dentro de cada base (separa o efeito da doença do efeito da origem)
  - resumo.md pronto para colar no relatório

Uso:
  python avaliacao_integrante4.py                      # procura resultados_*/predicoes_teste.csv
  python avaliacao_integrante4.py --modelo CNN=resultados_cnn_realcado/predicoes_teste.csv \
                                  --modelo SVM=resultados_svm/predicoes_teste.csv
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, roc_auc_score

CLASSES = ['normal', 'pneumonia', 'tuberculose']
CLASSE_NEGATIVA = 'normal'
CORES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
FONTES = {'MC': 'Montgomery', 'SZ': 'Shenzhen', 'NIH': 'NIH ChestX-ray14'}
N_BOOT = 2000


# ---------------------------------------------------------------- leitura
def carregar_predicoes(caminho):
    df = pd.read_csv(caminho)
    faltando = {'arquivo', 'classe_real', 'classe_predita'} - set(df.columns)
    if faltando:
        raise ValueError(f'{caminho}: faltam as colunas {sorted(faltando)}')
    for col in ('classe_real', 'classe_predita'):
        df[col] = df[col].astype(str).str.strip().str.lower()
    desconhecidas = set(df.classe_real) | set(df.classe_predita)
    desconhecidas -= set(CLASSES)
    if desconhecidas:
        raise ValueError(f'{caminho}: classes fora de {CLASSES}: {sorted(desconhecidas)}')
    # chave de junção: só o nome do arquivo (o caminho muda entre zip e pasta)
    df['chave'] = df['arquivo'].astype(str).str.replace('\\', '/', regex=False).str.split('/').str[-1]
    if df['chave'].duplicated().any():
        raise ValueError(f'{caminho}: há nomes de arquivo repetidos no teste')
    df['fonte'] = df['chave'].str.split('_').str[0].where(lambda s: s.isin(list(FONTES)), 'outra')
    return df.set_index('chave')


def descobrir_modelos(raiz):
    modelos = {}
    for csv in sorted(Path(raiz).glob('resultados_*/predicoes_teste.csv')):
        nome = csv.parent.name.removeprefix('resultados_')
        modelos[nome] = csv
    return modelos


# ---------------------------------------------------------------- métricas
def metricas_basicas(real, pred):
    """Métricas que serão usadas no bootstrap (só dependem de real/pred)."""
    m = {'acuracia': float(np.mean(real == pred))}
    f1s = []
    for c in CLASSES:
        tp = np.sum((real == c) & (pred == c)); fn = np.sum((real == c) & (pred != c))
        fp = np.sum((real != c) & (pred == c)); tn = np.sum((real != c) & (pred != c))
        sens = tp / (tp + fn) if tp + fn else np.nan
        prec = tp / (tp + fp) if tp + fp else np.nan
        espec = tn / (tn + fp) if tn + fp else np.nan
        f1 = 2 * prec * sens / (prec + sens) if (tp > 0) else 0.0
        m[f'sensibilidade_{c}'] = float(sens)
        m[f'precisao_{c}'] = float(prec)
        m[f'especificidade_{c}'] = float(espec)
        m[f'f1_{c}'] = float(f1)
        f1s.append(f1)
    m['f1_macro'] = float(np.mean(f1s))
    m['sensibilidade_macro'] = float(np.nanmean([m[f'sensibilidade_{c}'] for c in CLASSES]))
    m['precisao_macro'] = float(np.nanmean([m[f'precisao_{c}'] for c in CLASSES]))
    # visão de triagem: qualquer doença x normal
    doente_real, doente_pred = real != CLASSE_NEGATIVA, pred != CLASSE_NEGATIVA
    m['triagem_sensibilidade_doenca'] = float(np.mean(doente_pred[doente_real])) if doente_real.any() else np.nan
    m['triagem_especificidade'] = float(np.mean(~doente_pred[~doente_real])) if (~doente_real).any() else np.nan
    return m


def auc_ovr(df):
    cols = [f'prob_{c}' for c in CLASSES]
    if not set(cols) <= set(df.columns):
        return None
    try:
        y = df['classe_real'].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        return float(roc_auc_score(y, df[cols].to_numpy(), multi_class='ovr', average='macro', labels=[0, 1, 2]))
    except ValueError:
        return None


def intervalos_bootstrap(real, pred, chaves, semente=42):
    rng = np.random.default_rng(semente)
    n = len(real)
    amostras = {k: [] for k in chaves}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = metricas_basicas(real[idx], pred[idx])
        for k in chaves:
            amostras[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in amostras.items()}


def mcnemar(acerto_a, acerto_b):
    """Teste exato de McNemar: só contam os casos em que os modelos discordam."""
    b = int(np.sum(acerto_a & ~acerto_b))  # A acerta, B erra
    c = int(np.sum(~acerto_a & acerto_b))  # A erra, B acerta
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, float(p)


# ---------------------------------------------------------------- figuras
def figura_matriz(real, pred, titulo, destino):
    mat = confusion_matrix(real, pred, labels=CLASSES)
    pct = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(pct, cmap='Blues', vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(mat):
        cor = 'white' if pct[i, j] > 0.55 else '#1a1a19'
        ax.text(j, i, f'{v}\n({pct[i, j]:.0%})', ha='center', va='center', color=cor, fontsize=11)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Classe predita', ylabel='Classe real', title=titulo)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig(destino, dpi=160); plt.close(fig)


def figura_matrizes_lado_a_lado(dados, destino):
    """Todas as matrizes de confusão numa figura só (para slides e relatório)."""
    curtas = {'normal': 'Normal', 'pneumonia': 'Pneum.', 'tuberculose': 'Tuberc.'}
    fig, eixos = plt.subplots(1, len(dados), figsize=(4.2 * len(dados), 4.6), squeeze=False)
    for ax, (nome, df) in zip(eixos[0], dados.items()):
        mat = confusion_matrix(df.classe_real, df.classe_predita, labels=CLASSES)
        pct = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
        ax.imshow(pct, cmap='Blues', vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, str(v), ha='center', va='center', fontsize=20, color='white' if pct[i, j] > 0.55 else '#1a1a19')
        acc = float((df.classe_real == df.classe_predita).mean())
        ax.set_title(f'{nome}  ({acc:.1%})'.replace('.', ','), fontsize=17, pad=10)
        ax.set_xticks(range(3), [curtas[c] for c in CLASSES], fontsize=13)
        ax.set_yticks(range(3), [curtas[c] for c in CLASSES], fontsize=13)
        ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
    eixos[0][0].set_ylabel('Classe real', fontsize=14)
    for ax in eixos[0]: ax.set_xlabel('Classe predita', fontsize=14)
    fig.tight_layout(); fig.savefig(destino, dpi=120); plt.close(fig)


def figura_comparacao(tabela, destino):
    metricas = [('sensibilidade_normal', 'Sens. normal'), ('sensibilidade_pneumonia', 'Sens. pneumonia'),
                ('sensibilidade_tuberculose', 'Sens. tuberculose'), ('triagem_sensibilidade_doenca', 'Sens. qualquer doença'),
                ('f1_macro', 'F1 macro'), ('acuracia', 'Acurácia')]
    modelos = list(tabela.index)
    x = np.arange(len(metricas))
    largura = 0.8 / len(modelos)
    fig, ax = plt.subplots(figsize=(11, 5.0))
    for k, modelo in enumerate(modelos):
        valores = [tabela.loc[modelo, m] for m, _ in metricas]
        pos = x - 0.4 + largura * (k + 0.5)
        ax.bar(pos, valores, largura - 0.02, color=CORES[k % len(CORES)], label=modelo, edgecolor='white', linewidth=1)
        for p, v in zip(pos, valores):
            if not np.isnan(v):
                ax.text(p, v + 0.015, f'{v:.2f}', ha='center', va='bottom', fontsize=8, color='#444')
    ax.set_xticks(x, [r for _, r in metricas])
    ax.set_ylim(0, 1.08); ax.set_ylabel('Valor (0 a 1)')
    ax.set_title('Comparação dos classificadores no conjunto de teste')
    ax.yaxis.grid(True, color='#e5e5e5'); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, ncol=len(modelos), loc='upper center', bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout(); fig.savefig(destino, dpi=160); plt.close(fig)


# ---------------------------------------------------------------- principal
def avaliar_modelo(nome, df, pasta):
    real, pred = df['classe_real'].to_numpy(), df['classe_predita'].to_numpy()
    m = metricas_basicas(real, pred)
    m['auc_roc_ovr_macro'] = auc_ovr(df)
    m['n_imagens'] = int(len(df))
    m['suporte'] = {c: int(np.sum(real == c)) for c in CLASSES}
    chaves_ic = ['acuracia', 'f1_macro', 'sensibilidade_pneumonia', 'sensibilidade_tuberculose', 'triagem_sensibilidade_doenca']
    m['ic95'] = intervalos_bootstrap(real, pred, chaves_ic)
    (pasta / f'metricas_{nome}.json').write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding='utf8')
    figura_matriz(real, pred, f'Matriz de confusão - {nome}', pasta / f'matriz_{nome}.png')
    return m


def analise_por_fonte(dados):
    linhas = []
    for nome, df in dados.items():
        for (fonte, classe), g in df.groupby(['fonte', 'classe_real']):
            linha = {'modelo': nome, 'fonte': FONTES.get(fonte, fonte), 'classe_real': classe, 'n': len(g),
                     'acertos': int((g.classe_real == g.classe_predita).sum())}
            for c in CLASSES:
                linha[f'predito_{c}'] = int((g.classe_predita == c).sum())
            linhas.append(linha)
    t = pd.DataFrame(linhas)
    t['taxa_acerto'] = (t.acertos / t.n).round(3)
    return t


def analise_intra_base(dados):
    """Normal x doença DENTRO de cada base: tira o efeito da origem da imagem.
    Acurácia balanceada perto de 0,5 = o modelo não separa as classes quando a base é a mesma."""
    linhas = []
    for nome, df in dados.items():
        for fonte, g in df.groupby('fonte'):
            doentes = g[g.classe_real != CLASSE_NEGATIVA]
            normais = g[g.classe_real == CLASSE_NEGATIVA]
            if doentes.empty or normais.empty:
                continue
            doenca = doentes.classe_real.iloc[0]
            sens = float((doentes.classe_predita == doenca).mean())
            espec = float((normais.classe_predita == CLASSE_NEGATIVA).mean())
            linhas.append({'modelo': nome, 'fonte': FONTES.get(fonte, fonte), 'doenca': doenca,
                           'n_normal': len(normais), 'n_doente': len(doentes), 'sensibilidade': round(sens, 3),
                           'especificidade': round(espec, 3), 'acuracia_balanceada': round((sens + espec) / 2, 3)})
    return pd.DataFrame(linhas)


def escrever_resumo(tabela, metricas, mc, fontes, conjunto, pasta, intra=None):
    def ic(m, k):
        lo, hi = m['ic95'][k]
        return f'{m[k]:.1%} (IC95% {lo:.1%}–{hi:.1%})'
    L = ['# Resumo da avaliação (gerado por avaliacao_integrante4.py)', '',
         f'Conjunto de teste comum a todos os modelos: **{conjunto} imagens**.', '',
         '## Tabela comparativa', '',
         '| Modelo | Acurácia | F1 macro | Sens. normal | Sens. pneumonia | Sens. tuberculose | Sens. qualquer doença | Espec. triagem | AUC macro |',
         '|---|---|---|---|---|---|---|---|---|']
    for nome, m in metricas.items():
        auc = f"{m['auc_roc_ovr_macro']:.3f}" if m['auc_roc_ovr_macro'] is not None else '–'
        L.append(f"| {nome} | {m['acuracia']:.1%} | {m['f1_macro']:.3f} | {m['sensibilidade_normal']:.1%} | "
                 f"{m['sensibilidade_pneumonia']:.1%} | {m['sensibilidade_tuberculose']:.1%} | "
                 f"{m['triagem_sensibilidade_doenca']:.1%} | {m['triagem_especificidade']:.1%} | {auc} |")
    L += ['', '## Intervalos de confiança (bootstrap, 2000 reamostragens)', '']
    for nome, m in metricas.items():
        L.append(f"- **{nome}**: acurácia {ic(m, 'acuracia')}; sensibilidade pneumonia {ic(m, 'sensibilidade_pneumonia')}; "
                 f"sensibilidade tuberculose {ic(m, 'sensibilidade_tuberculose')}")
    if len(mc):
        L += ['', '## Teste de McNemar (pares de modelos, mesmas imagens)', '',
              '| Modelo A | Modelo B | Só A acerta | Só B acerta | p-valor |', '|---|---|---|---|---|']
        for _, r in mc.iterrows():
            L.append(f"| {r.modelo_a} | {r.modelo_b} | {r.so_a_acerta} | {r.so_b_acerta} | {r.p_valor:.4f} |")
        L.append('\np < 0,05 indica diferença de acertos estatisticamente significativa entre os dois modelos.')
    if intra is not None and len(intra):
        L += ['', '## Normal x doença dentro de cada base (acurácia balanceada; 0,5 = acaso)', '',
              intra.pivot(index='modelo', columns='fonte', values='acuracia_balanceada').to_markdown()]
    L += ['', '## Erros por base de origem', '', fontes.to_markdown(index=False)]
    (pasta / 'resumo.md').write_text('\n'.join(L), encoding='utf8')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--modelo', action='append', default=[], metavar='NOME=CSV')
    ap.add_argument('--raiz', default='.', help='pasta onde procurar resultados_*/predicoes_teste.csv')
    ap.add_argument('--saida', default='resultados_avaliacao')
    args = ap.parse_args()

    modelos = dict(s.split('=', 1) for s in args.modelo) if args.modelo else descobrir_modelos(args.raiz)
    if not modelos:
        ap.error('nenhum arquivo de predições encontrado; use --modelo NOME=caminho.csv')
    pasta = Path(args.saida); pasta.mkdir(parents=True, exist_ok=True)
    dados = {nome: carregar_predicoes(csv) for nome, csv in modelos.items()}

    # compara somente nas imagens que TODOS os modelos classificaram
    comuns = sorted(set.intersection(*(set(d.index) for d in dados.values())))
    for nome, d in dados.items():
        fora = len(d) - len(comuns)
        print(f'{nome}: {len(d)} imagens' + (f' ({fora} fora do conjunto comum)' if fora else ''))
    if not comuns:
        raise SystemExit('Os modelos não têm nenhuma imagem de teste em comum: a comparação não é válida.')
    dados = {n: d.loc[comuns] for n, d in dados.items()}
    reais = {n: tuple(d.classe_real) for n, d in dados.items()}
    if len(set(reais.values())) > 1:
        raise SystemExit('Rótulos reais divergem entre os modelos para as mesmas imagens: confira as bases.')

    metricas = {n: avaliar_modelo(n, d, pasta) for n, d in dados.items()}
    colunas = ['acuracia', 'f1_macro', 'precisao_macro', 'sensibilidade_macro'] + \
              [f'{p}_{c}' for c in CLASSES for p in ('precisao', 'sensibilidade', 'especificidade', 'f1')] + \
              ['triagem_sensibilidade_doenca', 'triagem_especificidade', 'auc_roc_ovr_macro']
    tabela = pd.DataFrame({n: {k: m[k] for k in colunas} for n, m in metricas.items()}).T.astype(float)
    tabela.round(4).to_csv(pasta / 'tabela_comparativa.csv', index_label='modelo')
    figura_comparacao(tabela, pasta / 'comparacao_metricas.png')
    figura_matrizes_lado_a_lado(dados, pasta / 'matrizes_comparacao.png')

    linhas = []
    for a, b in combinations(dados, 2):
        ok_a = (dados[a].classe_real == dados[a].classe_predita).to_numpy()
        ok_b = (dados[b].classe_real == dados[b].classe_predita).to_numpy()
        so_a, so_b, p = mcnemar(ok_a, ok_b)
        linhas.append({'modelo_a': a, 'modelo_b': b, 'so_a_acerta': so_a, 'so_b_acerta': so_b, 'p_valor': p})
    mc = pd.DataFrame(linhas, columns=['modelo_a', 'modelo_b', 'so_a_acerta', 'so_b_acerta', 'p_valor'])
    mc.to_csv(pasta / 'mcnemar.csv', index=False)

    fontes = analise_por_fonte(dados)
    fontes.to_csv(pasta / 'analise_por_fonte.csv', index=False)
    intra = analise_intra_base(dados)
    intra.to_csv(pasta / 'analise_intra_base.csv', index=False)
    escrever_resumo(tabela, metricas, mc, fontes, len(comuns), pasta, intra)

    print('\n' + tabela[['acuracia', 'f1_macro', 'sensibilidade_pneumonia', 'sensibilidade_tuberculose',
                         'triagem_sensibilidade_doenca']].round(3).to_string())
    print(f'\nArquivos salvos em {pasta.resolve()}')


if __name__ == '__main__':
    main()
