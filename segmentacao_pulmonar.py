"""Integrante 2: segmentação pulmonar com PDI clássico (limiarização + morfologia) e classificação
só com a região pulmonar.

Etapas:
  1) segmentar_pulmao(img): máscara binária 512x512 dos dois pulmões, só com PDI clássico
     (suavização, Otsu, inversão, remoção do fundo/borda, abertura, fechamento, 2 maiores
     componentes conexos, preenchimento de buracos). Ver os parâmetros documentados abaixo.
  2) figura qualitativa com exemplos das três bases (resultados_segmentacao/exemplos.png).
  3) as MESMAS características de classico_3classes.py (HOG, LBP, GLCM de integracao.py), extraídas
     só da região pulmonar, e o MESMO protocolo: treino no split treino, escolha da configuração
     pelo F1 macro na validação, uma única avaliação no teste (160 imagens), class_weight='balanced'
     no SVM e no RF. Predições em resultados_<svm|rf|knn>_seg/predicoes_teste.csv.

A máscara é calculada sempre na versão 'basico' (a 'realcado' passou por CLAHE, que achata o contraste
global e atrapalha o limiar) e aplicada à imagem da versão escolhida (basico ou realcado).

Uso:  python segmentacao_pulmonar.py --dados segunda_leitura_processado.zip
      (ou a pasta que contém processado/, por exemplo TrabalhoPDIav3)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

# ---------------------------------------------------------------- parâmetros da segmentação
# Todos em pixels da imagem 512x512.
SIGMA_SUAVIZACAO = 3        # gaussiana antes do limiar: tira ruído e a textura fina das costelas/vasos
LIMIAR_CORPO = 10           # pixels <= 10 são as barras pretas do preenchimento (quadrar a imagem): ficam fora do Otsu
FRACAO_CENTRO = (0.15, 0.80, 0.15, 0.85)   # janela (linha_ini, linha_fim, col_ini, col_fim) onde o Otsu é calculado
USAR_ADAPTATIVO = True
BLOCO_ADAPTATIVO = 101      # janela da média local (~20% da largura): maior que uma costela, menor que um pulmão
C_ADAPTATIVO = 0            # o pixel precisa ficar abaixo da média local menos C
RAIO_ABERTURA = 5           # elemento estruturante elíptico 11x11: abertura remove ruído e corta pontes
                            # finas entre o pulmão e o ar de fora do corpo (axila, pescoço)
RAIO_ABERTURA_FORTE = 12    # 2ª tentativa (25x25) quando o pulmão continua ligado ao fundo e some ao tirar a borda
RAIO_FECHAMENTO = 10        # elemento elíptico 21x21: fechamento devolve ao pulmão vasos, costelas e opacidades pequenas
AREA_MINIMA = 0.01          # componente com menos de 1% da imagem não é pulmão
RAZAO_SEGUNDO = 0.15        # o 2º pulmão precisa ter pelo menos 15% da área do 1º (senão é bolha gástrica, traqueia...)


def _disco(raio):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * raio + 1, 2 * raio + 1))


def _tirar_borda(binaria):
    """Remove os componentes conexos que tocam a borda da imagem (fundo, ar fora do corpo, barras pretas)."""
    n, rotulos = cv2.connectedComponents(binaria.astype(np.uint8), connectivity=8)
    na_borda = np.unique(np.concatenate([rotulos[0], rotulos[-1], rotulos[:, 0], rotulos[:, -1]]))
    return binaria & ~np.isin(rotulos, na_borda)


def _maiores_componentes(binaria, k=2):
    """Mantém os k maiores componentes conexos (pulmão direito e esquerdo) que passam nos critérios de área."""
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(binaria.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    ordem = np.argsort(areas)[::-1][:k] + 1
    manter = [r for r in ordem if stats[r, cv2.CC_STAT_AREA] >= AREA_MINIMA * binaria.size]
    if len(manter) == 2 and stats[manter[1], cv2.CC_STAT_AREA] < RAZAO_SEGUNDO * stats[manter[0], cv2.CC_STAT_AREA]:
        manter = manter[:1]
    return np.isin(rotulos, manter)


def segmentar_pulmao(img_cinza, etapas=False):
    """Segmenta os pulmões de uma radiografia de tórax 512x512 em tons de cinza.

    Devolve uma máscara uint8 0/255 (512x512). Com etapas=True devolve também um dict com as
    imagens intermediárias (para figuras) e qual caminho foi usado ('normal', 'abertura_forte' ou 'falha').
    """
    passos = {}
    # 1) suavização gaussiana (sigma 3): o limiar fica menos sensível a ruído e às bordas das costelas
    suave = cv2.GaussianBlur(img_cinza, (0, 0), SIGMA_SUAVIZACAO)
    passos['suavizada'] = suave

    # 2) limiar de Otsu calculado só na região central do tórax e sem as barras pretas do preenchimento.
    #    Na imagem inteira o Otsu costuma separar "ar fora do corpo x resto"; no centro os dois modos
    #    do histograma são pulmão (escuro) e mediastino/costelas (claro), que é o que interessa.
    h, w = suave.shape
    centro = np.zeros_like(suave, bool)
    centro[int(h * FRACAO_CENTRO[0]):int(h * FRACAO_CENTRO[1]), int(w * FRACAO_CENTRO[2]):int(w * FRACAO_CENTRO[3])] = True
    corpo = suave > LIMIAR_CORPO
    limiar, _ = cv2.threshold(suave[centro & corpo].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3) inversão: o pulmão (cheio de ar) é ESCURO no raio-X, então ficamos com o que está abaixo do limiar
    escuro_otsu = suave <= limiar
    # 3b) limiar adaptativo (média local numa janela grande): o pulmão é mais escuro que a vizinhança,
    #     enquanto a parede torácica é mais clara que a vizinhança (fica entre pulmão e ar externo).
    #     O E lógico com o Otsu corta a ligação pulmão-fundo pela lateral e pelo ápice.
    escuro_adapt = cv2.adaptiveThreshold(suave, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                         BLOCO_ADAPTATIVO, C_ADAPTATIVO) > 0
    escuro = escuro_otsu & escuro_adapt if USAR_ADAPTATIVO else escuro_otsu
    passos['otsu_invertida'] = escuro

    tentativas = []
    for nome, raio in (('normal', RAIO_ABERTURA), ('abertura_forte', RAIO_ABERTURA_FORTE)):
        # 4) abertura (erosão seguida de dilatação): tira ruído e desfaz ligações finas com o fundo
        aberta = cv2.morphologyEx(escuro.astype(np.uint8), cv2.MORPH_OPEN, _disco(raio)).astype(bool)
        # 5) remoção do fundo: tudo que é escuro e toca a borda é ar de fora do corpo ou barra preta
        sem_fundo = _tirar_borda(aberta)
        # 6) os 2 maiores componentes conexos = pulmão direito e esquerdo
        pulmoes = _maiores_componentes(sem_fundo, k=2)
        tentativas.append((nome, aberta, sem_fundo, pulmoes, cv2.connectedComponents(pulmoes.astype(np.uint8))[0] - 1))
        if tentativas[-1][4] == 2:
            break
    # fica com a 1ª tentativa que achou os dois pulmões; se nenhuma achou, com a que achou um
    nome, aberta, sem_fundo, pulmoes, _ = max(tentativas, key=lambda t: t[4])
    caminho = nome if pulmoes.any() else 'falha'

    if not pulmoes.any():
        # 6b) último recurso, para recortes apertados em que o pulmão encosta na borda da imagem:
        #     não remove a borda, mas só aceita componentes com centroide dentro da janela central
        aberta = cv2.morphologyEx(escuro.astype(np.uint8), cv2.MORPH_OPEN, _disco(RAIO_ABERTURA))
        n, rotulos, stats, centroides = cv2.connectedComponentsWithStats(aberta, connectivity=8)
        h, w = escuro.shape
        dentro = [r for r in range(1, n)
                  if FRACAO_CENTRO[0] * h <= centroides[r][1] <= FRACAO_CENTRO[1] * h
                  and FRACAO_CENTRO[2] * w <= centroides[r][0] <= FRACAO_CENTRO[3] * w]
        sem_fundo = np.isin(rotulos, dentro)
        pulmoes = _maiores_componentes(sem_fundo, k=2)
        caminho = 'sem_remover_borda' if pulmoes.any() else 'falha'
    passos['aberta'], passos['sem_fundo'], passos['componentes'] = aberta, sem_fundo, pulmoes

    if not pulmoes.any():
        # sem pulmão encontrado: devolve a máscara inteira (a imagem fica como está) e marca a falha
        mascara = np.full(img_cinza.shape, 255, np.uint8)
    else:
        # 7) fechamento (dilatação seguida de erosão): preenche reentrâncias da borda do pulmão
        #    (vasos do hilo, costelas, opacidades) e 8) preenchimento de buracos internos
        fechada = cv2.morphologyEx(pulmoes.astype(np.uint8), cv2.MORPH_CLOSE, _disco(RAIO_FECHAMENTO))
        cheia = ndi.binary_fill_holes(fechada)
        passos['fechada'] = fechada
        # o fechamento pode juntar de novo um pedaço pequeno: fica só com os 2 maiores outra vez
        mascara = (_maiores_componentes(cheia, k=2) * 255).astype(np.uint8)
    passos['caminho'] = caminho
    return (mascara, passos) if etapas else mascara


# ---------------------------------------------------------------- aplicação da máscara
MARGEM_RECORTE = 10   # pixels a mais em volta do retângulo dos pulmões no modo 'recortar'
MODOS = ['zerar', 'recortar']


def aplicar_mascara(img, mascara, modo):
    """'zerar': mantém o tamanho 512x512 e põe 0 em tudo que está fora do pulmão.
    'recortar': recorta o retângulo que envolve os dois pulmões (com margem), sem zerar nada dentro dele;
    tira bordas, marcações e equipamento, mas mantém a textura contínua (mediastino incluído)."""
    if modo == 'zerar':
        return np.where(mascara > 0, img, 0).astype(np.uint8)
    linhas, colunas = np.where(mascara > 0)
    h, w = img.shape
    y0, y1 = max(linhas.min() - MARGEM_RECORTE, 0), min(linhas.max() + MARGEM_RECORTE + 1, h)
    x0, x1 = max(colunas.min() - MARGEM_RECORTE, 0), min(colunas.max() + MARGEM_RECORTE + 1, w)
    return img[y0:y1, x0:x1]


# ---------------------------------------------------------------- dados
def _nome_zip(caminho):
    """Caminho no formato do zip (processado/<versao>/<split>/<classe>/<arquivo>), venha do zip ou da pasta."""
    partes = Path(str(caminho).replace('\\', '/')).parts
    return '/'.join(partes[partes.index('processado'):])


def calcular_mascaras(fonte):
    """Segmenta todas as imagens 'basico' (treino, validação e teste). Chave = nome do arquivo."""
    from classico_3classes import SPLITS, listar_imagens
    mascaras, linhas = {}, []
    for split in SPLITS:
        itens, ler = listar_imagens(fonte, 'basico', split)
        for nome, classe in itens:
            mascara, passos = segmentar_pulmao(ler(nome), etapas=True)
            chave = Path(nome).name
            mascaras[chave] = mascara
            n_comp = cv2.connectedComponents(mascara)[0] - 1
            linhas.append({'arquivo': _nome_zip(nome).replace('/basico/', '/{versao}/'), 'fonte': chave.split('_')[0],
                           'split': split, 'classe': ['normal', 'pneumonia', 'tuberculose'][classe], 'area_fracao': round(float((mascara > 0).mean()), 4),
                           'componentes': n_comp, 'caminho': passos['caminho']})
        print(f'  segmentadas {split:9s} {len(itens)} imagens')
    return mascaras, pd.DataFrame(linhas)


def extrair_tudo_seg(fonte, mascaras, cache):
    """HOG, LBP e GLCM (funções de integracao.py) da região pulmonar, para cada versão, split e modo."""
    from classico_3classes import SPLITS, VERSOES, listar_imagens
    from integracao import extrair_caracteristicas
    if cache.exists():
        return np.load(cache, allow_pickle=True)['tabela'].item()
    tabela = {}
    for versao in VERSOES:
        for split in SPLITS:
            itens, ler = listar_imagens(fonte, versao, split)
            feats = {(m, k): [] for m in MODOS for k in ('hog', 'lbp', 'glcm')}
            for nome, _ in itens:
                img = ler(nome)
                for modo in MODOS:
                    regiao = aplicar_mascara(img, mascaras[Path(nome).name], modo)
                    for k in ('hog', 'lbp', 'glcm'):
                        # extrair_caracteristicas redimensiona para 128x128 e, na versão 'basico',
                        # aplica gaussiana 3x3 + CLAHE, exatamente como em classico_3classes.py
                        feats[(modo, k)].append(extrair_caracteristicas(regiao, [k], versao))
            for modo in MODOS:
                tabela[(modo, versao, split)] = {'arquivos': [_nome_zip(n) for n, _ in itens],
                                                 'y': np.array([c for _, c in itens]),
                                                 **{k: np.array(feats[(modo, k)]) for k in ('hog', 'lbp', 'glcm')}}
            print(f'  características {versao:8s} {split:9s} {len(itens)} imagens')
    np.savez_compressed(cache, tabela=np.array(tabela, dtype=object))
    return tabela


def montar(tabela, modo, versao, split, descritores):
    t = tabela[(modo, versao, split)]
    return np.hstack([t[d] for d in descritores]), t['y']


# ---------------------------------------------------------------- figuras
def figura_etapas(img, destino):
    """Uma imagem passando por cada etapa da segmentação (para o relatório)."""
    import matplotlib.pyplot as plt
    mascara, p = segmentar_pulmao(img, etapas=True)
    quadros = [('1. original (basico)', img), ('2. gaussiana (sigma 3)', p['suavizada']),
               ('3. Otsu central E adaptativo,\ninvertido (pulmão escuro)', p['otsu_invertida']),
               (f'4. abertura (elipse {2 * RAIO_ABERTURA + 1}x{2 * RAIO_ABERTURA + 1})', p['aberta']),
               ('5. sem regiões na borda', p['sem_fundo']), ('6. 2 maiores componentes', p['componentes']),
               (f'7. fechamento (elipse {2 * RAIO_FECHAMENTO + 1}x{2 * RAIO_FECHAMENTO + 1})', p.get('fechada', mascara)),
               ('8. buracos preenchidos (máscara final)', mascara)]
    fig, eixos = plt.subplots(2, 4, figsize=(16, 9.5), layout='constrained')
    for ax, (titulo, q) in zip(eixos.flat, quadros):
        ax.imshow(q, cmap='gray'); ax.set_title(titulo, fontsize=10); ax.axis('off')
    fig.suptitle('Segmentação pulmonar por limiarização e morfologia matemática', fontsize=13)
    fig.savefig(destino, dpi=110); plt.close(fig)


def figura_exemplos(fonte, mascaras, destino, por_fonte=2):
    """Exemplos do teste, 2 por base: imagem, máscara prevista, contorno, região usada em cada modo."""
    import matplotlib.pyplot as plt
    from classico_3classes import listar_imagens
    itens, ler = listar_imagens(fonte, 'realcado', 'teste')
    rng = np.random.default_rng(42)
    escolhidos = []
    for prefixo in ('MC', 'SZ', 'NIH'):
        da_fonte = [n for n, _ in itens if Path(n).name.startswith(prefixo + '_')]
        escolhidos += [da_fonte[i] for i in sorted(rng.choice(len(da_fonte), por_fonte, replace=False))]
    fig, eixos = plt.subplots(len(escolhidos), 4, figsize=(13, 3.3 * len(escolhidos)))
    for linha, nome in zip(eixos, escolhidos):
        img, mascara = ler(nome), mascaras[Path(nome).name]
        contorno = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cs, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contorno, cs, -1, (255, 40, 40), 3)
        for ax, (titulo, q) in zip(linha, [(Path(nome).name, img), ('máscara prevista', mascara),
                                           ('contorno', contorno), ('modo zerar', aplicar_mascara(img, mascara, 'zerar'))]):
            ax.imshow(q, cmap=None if q.ndim == 3 else 'gray'); ax.set_title(titulo, fontsize=9); ax.axis('off')
    fig.tight_layout(); fig.savefig(destino, dpi=80); plt.close(fig)


# ---------------------------------------------------------------- principal
def main():
    import matplotlib
    matplotlib.use('Agg')
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score

    from classico_3classes import COMBINACOES, VERSOES, classificadores, listar_imagens
    from integracao import salvar_predicoes

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dados', default=None,
                    help='zip do Integrante 1 ou pasta que contém processado/ (padrão: o zip, se existir, senão TrabalhoPDIav3)')
    ap.add_argument('--cache', default='caracteristicas_seg_cache.npz')
    ap.add_argument('--saida', default='resultados_segmentacao')
    args = ap.parse_args()
    fonte = args.dados or ('segunda_leitura_processado.zip' if Path('segunda_leitura_processado.zip').exists()
                           else 'TrabalhoPDIav3')
    saida = Path(args.saida); saida.mkdir(parents=True, exist_ok=True)

    # 1) segmentação de todas as imagens (versão basico)
    print(f'Segmentando as imagens de {fonte}...')
    mascaras, estat = calcular_mascaras(fonte)
    estat.to_csv(saida / 'mascaras_por_imagem.csv', index=False)
    resumo_seg = estat.groupby('fonte').agg(n=('area_fracao', 'size'), area_media=('area_fracao', 'mean'),
                                            area_desvio=('area_fracao', 'std'), area_min=('area_fracao', 'min'),
                                            area_max=('area_fracao', 'max'),
                                            com_2_pulmoes=('componentes', lambda s: int((s == 2).sum())),
                                            com_1_pulmao=('componentes', lambda s: int((s == 1).sum())),
                                            falhas=('caminho', lambda s: int((s == 'falha').sum()))).round(4)
    resumo_seg.to_csv(saida / 'resumo_mascaras.csv')
    print(resumo_seg.to_string())

    # 2) figuras
    _, ler = listar_imagens(fonte, 'basico', 'teste')
    figura_etapas(ler(_nome_exemplo(fonte)), saida / 'etapas.png')
    figura_exemplos(fonte, mascaras, saida / 'exemplos.png')

    # 3) características só da região pulmonar
    print('Extraindo características da região pulmonar (só na primeira execução)...')
    tabela = extrair_tudo_seg(fonte, mascaras, Path(args.cache))

    # 4) grade escolhida pela VALIDAÇÃO (F1 macro): modo x versão x descritores
    linhas = []
    for chave, (nome_clf, fabrica) in classificadores().items():
        for modo in MODOS:
            for versao in VERSOES:
                for nome_comb, desc in COMBINACOES.items():
                    X_tr, y_tr = montar(tabela, modo, versao, 'treino', desc)
                    X_va, y_va = montar(tabela, modo, versao, 'validacao', desc)
                    p = fabrica().fit(X_tr, y_tr).predict(X_va)
                    linhas.append({'classificador': chave, 'nome': nome_clf, 'modo': modo, 'versao': versao,
                                   'descritores': nome_comb, 'f1_macro_validacao': f1_score(y_va, p, average='macro'),
                                   'acuracia_validacao': accuracy_score(y_va, p)})
                    print(f'{nome_clf:14s} {modo:8s} {versao:8s} {nome_comb:13s} F1 val {linhas[-1]["f1_macro_validacao"]:.3f}')
    grade = pd.DataFrame(linhas)
    grade.round(4).sort_values(['classificador', 'f1_macro_validacao'], ascending=[True, False]) \
        .to_csv(saida / 'grade_validacao.csv', index=False)

    # 5) uma única avaliação no teste com a configuração escolhida
    for chave, (nome_clf, fabrica) in classificadores().items():
        sub = grade[grade.classificador == chave]
        best = sub.loc[sub.f1_macro_validacao.idxmax()]
        desc = COMBINACOES[best.descritores]
        X_tr, y_tr = montar(tabela, best.modo, best.versao, 'treino', desc)
        X_te, y_te = montar(tabela, best.modo, best.versao, 'teste', desc)
        modelo = fabrica().fit(X_tr, y_tr)
        pred, prob = modelo.predict(X_te), modelo.predict_proba(X_te)
        pasta = Path(f'resultados_{chave}_seg')
        salvar_predicoes(pasta / 'predicoes_teste.csv', tabela[(best.modo, best.versao, 'teste')]['arquivos'], y_te, pred, prob)
        cfg = {'classificador': nome_clf, 'segmentacao': 'segmentacao_pulmonar.segmentar_pulmao (máscara da versão basico)',
               'modo': best.modo, 'versao': best.versao, 'descritores': best.descritores,
               'f1_macro_validacao': round(float(best.f1_macro_validacao), 4),
               'acuracia_validacao': round(float(best.acuracia_validacao), 4),
               'f1_macro_teste': round(float(f1_score(y_te, pred, average='macro')), 4),
               'acuracia_teste': round(float(accuracy_score(y_te, pred)), 4), 'n_teste': int(len(y_te))}
        (pasta / 'configuracao.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf8')
        print(f'\n>> {nome_clf}: escolhido {best.modo} + {best.versao} + {best.descritores} '
              f'(F1 val {best.f1_macro_validacao:.3f}) -> teste: acurácia {cfg["acuracia_teste"]:.3f}, '
              f'F1 macro {cfg["f1_macro_teste"]:.3f}')

    # 6) diagnóstico de viés, igual ao de classico_3classes.py (LBP+GLCM, realcado), agora só no pulmão:
    #    as características ainda dizem de qual BASE a imagem veio?
    fonte_de = lambda arqs: np.array([Path(a).name.split('_')[0] for a in arqs])
    vies = {'tarefa': 'prever a base de origem (MC, SZ, NIH) com LBP+GLCM realcado, só na região pulmonar'}
    for modo in MODOS:
        X_tr, _ = montar(tabela, modo, 'realcado', 'treino', ['lbp', 'glcm'])
        X_te, _ = montar(tabela, modo, 'realcado', 'teste', ['lbp', 'glcm'])
        f_tr = fonte_de(tabela[(modo, 'realcado', 'treino')]['arquivos'])
        f_te = fonte_de(tabela[(modo, 'realcado', 'teste')]['arquivos'])
        rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1).fit(X_tr, f_tr)
        vies[f'acuracia_teste_{modo}'] = round(float(accuracy_score(f_te, rf.predict(X_te))), 4)
    vies['acuracia_teste_sem_segmentacao'] = json.loads(Path('resultados_classico_vies_fonte.json').read_text(encoding='utf8'))['acuracia_teste'] \
        if Path('resultados_classico_vies_fonte.json').exists() else None
    vies['acaso_classe_majoritaria'] = round(float(pd.Series(f_te).value_counts(normalize=True).max()), 4)
    (saida / 'vies_fonte.json').write_text(json.dumps(vies, indent=2, ensure_ascii=False), encoding='utf8')
    print(f'\nDiagnóstico de viés (acerto da BASE de origem): {vies}')


def _nome_exemplo(fonte):
    """Uma imagem de Shenzhen do teste para a figura das etapas (a primeira em ordem alfabética)."""
    from classico_3classes import listar_imagens
    itens, _ = listar_imagens(fonte, 'basico', 'teste')
    return next(n for n, _ in itens if Path(n).name.startswith('SZ_'))


if __name__ == '__main__':
    main()
