"""Integrante 4: abordagem tradicional nas 3 classes, no MESMO teste da CNN.

Reaproveita o pré-processamento e os extratores do Integrante 2
(experimento_TOTAL_dataset_yeshua.py: HOG, LBP, GLCM em 128x128) e acrescenta o protocolo
necessário para comparar com a CNN de forma justa:
  - treino em processado/<versao>/treino, ESCOLHA da configuração pela validação (F1 macro)
    e uma única avaliação no teste completo (160 imagens, sem descartar nenhuma);
  - pesos balanceados por classe (o treino tem 368 normais e só 97 pneumonias);
  - predições salvas no formato comum (resultados_<modelo>/predicoes_teste.csv);
  - o melhor modelo salvo em modelos/modelo_classico.joblib para o protótipo.

Uso:  python classico_3classes.py --dados segunda_leitura_processado.zip
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from integracao import CLASSES, extrair_caracteristicas, salvar_modelo_classico, salvar_predicoes

VERSOES = ['basico', 'realcado']
SPLITS = ['treino', 'validacao', 'teste']
COMBINACOES = {'HOG': ['hog'], 'LBP': ['lbp'], 'GLCM': ['glcm'],
               'LBP+GLCM': ['lbp', 'glcm'], 'HOG+LBP+GLCM': ['hog', 'lbp', 'glcm']}


def classificadores():
    return {
        'svm': ('SVM (RBF)', lambda: make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, class_weight='balanced',
                                                                          probability=True, random_state=42))),
        'rf': ('Random Forest', lambda: RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                                               random_state=42, n_jobs=-1)),
        'knn': ('KNN (k=5)', lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
    }


def listar_imagens(fonte, versao, split):
    """Devolve [(nome_no_zip_ou_caminho, classe_idx)] e uma função para ler cada imagem."""
    fonte = Path(fonte)
    itens = []
    if fonte.is_file():
        z = zipfile.ZipFile(fonte)
        nomes = sorted(z.namelist())
        for i, c in enumerate(CLASSES):
            pref = f'processado/{versao}/{split}/{c}/'
            itens += [(n, i) for n in nomes if n.startswith(pref) and n.lower().endswith(('.png', '.jpg', '.jpeg'))]
        ler = lambda n: cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_GRAYSCALE)
    else:
        for i, c in enumerate(CLASSES):
            pasta = fonte / 'processado' / versao / split / c
            itens += [(str(p), i) for p in sorted(pasta.glob('*')) if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        ler = lambda n: cv2.imread(n, cv2.IMREAD_GRAYSCALE)
    if not itens:
        raise SystemExit(f'Nenhuma imagem em {fonte} / {versao} / {split}')
    return itens, ler


def extrair_tudo(fonte, cache):
    """Calcula HOG, LBP e GLCM uma vez por imagem e guarda em cache (.npz)."""
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d['tabela'].item()
    tabela = {}
    for versao in VERSOES:
        for split in SPLITS:
            itens, ler = listar_imagens(fonte, versao, split)
            feats = {k: [] for k in ('hog', 'lbp', 'glcm')}
            for nome, _ in itens:
                img = ler(nome)
                for k in feats:
                    feats[k].append(extrair_caracteristicas(img, [k], versao))
            tabela[(versao, split)] = {'arquivos': [n for n, _ in itens], 'y': np.array([c for _, c in itens]),
                                       **{k: np.array(v) for k, v in feats.items()}}
            print(f'  {versao:8s} {split:9s} {len(itens)} imagens')
    np.savez_compressed(cache, tabela=np.array(tabela, dtype=object))
    return tabela


def montar(tabela, versao, split, descritores):
    t = tabela[(versao, split)]
    return np.hstack([t[d] for d in descritores]), t['y']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dados', default='segunda_leitura_processado.zip', help='zip do Integrante 1 ou pasta extraída')
    ap.add_argument('--cache', default='caracteristicas_cache.npz')
    args = ap.parse_args()

    print('Extraindo características (só na primeira execução)...')
    tabela = extrair_tudo(args.dados, Path(args.cache))

    linhas = []
    for chave, (nome_clf, fabrica) in classificadores().items():
        for versao in VERSOES:
            for nome_comb, desc in COMBINACOES.items():
                X_tr, y_tr = montar(tabela, versao, 'treino', desc)
                X_va, y_va = montar(tabela, versao, 'validacao', desc)
                modelo = fabrica().fit(X_tr, y_tr)
                p = modelo.predict(X_va)
                linhas.append({'classificador': chave, 'nome': nome_clf, 'versao': versao, 'descritores': nome_comb,
                               'f1_macro_validacao': f1_score(y_va, p, average='macro'),
                               'acuracia_validacao': accuracy_score(y_va, p)})
                print(f'{nome_clf:14s} {versao:8s} {nome_comb:13s} F1 val {linhas[-1]["f1_macro_validacao"]:.3f}')
    grade = pd.DataFrame(linhas)

    melhores = {}
    for chave, (nome_clf, fabrica) in classificadores().items():
        sub = grade[grade.classificador == chave]
        best = sub.loc[sub.f1_macro_validacao.idxmax()]
        desc = COMBINACOES[best.descritores]
        X_tr, y_tr = montar(tabela, best.versao, 'treino', desc)
        X_te, y_te = montar(tabela, best.versao, 'teste', desc)
        modelo = fabrica().fit(X_tr, y_tr)
        pred = modelo.predict(X_te)
        prob = modelo.predict_proba(X_te)
        arquivos = tabela[(best.versao, 'teste')]['arquivos']
        salvar_predicoes(f'resultados_{chave}/predicoes_teste.csv', arquivos, y_te, pred, prob)
        cfg = {'classificador': nome_clf, 'versao': best.versao, 'descritores': best.descritores,
               'f1_macro_validacao': round(float(best.f1_macro_validacao), 4),
               'f1_macro_teste': round(float(f1_score(y_te, pred, average='macro')), 4),
               'acuracia_teste': round(float(accuracy_score(y_te, pred)), 4)}
        Path(f'resultados_{chave}/configuracao.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf8')
        melhores[chave] = (best, modelo, desc)
        print(f'\n>> {nome_clf}: escolhido {best.versao} + {best.descritores} (F1 val {best.f1_macro_validacao:.3f}) '
              f'-> teste: acurácia {cfg["acuracia_teste"]:.3f}, F1 macro {cfg["f1_macro_teste"]:.3f}')

    grade.round(4).sort_values(['classificador', 'f1_macro_validacao'], ascending=[True, False]) \
        .to_csv('resultados_classico_grade_validacao.csv', index=False)

    # diagnóstico de viés: as mesmas características conseguem dizer de qual BASE a imagem veio?
    fonte_de = lambda arqs: np.array([Path(a).name.split('_')[0] for a in arqs])
    X_tr, _ = montar(tabela, 'realcado', 'treino', ['lbp', 'glcm'])
    X_te, _ = montar(tabela, 'realcado', 'teste', ['lbp', 'glcm'])
    f_tr, f_te = fonte_de(tabela[('realcado', 'treino')]['arquivos']), fonte_de(tabela[('realcado', 'teste')]['arquivos'])
    rf_fonte = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1).fit(X_tr, f_tr)
    acc_fonte = accuracy_score(f_te, rf_fonte.predict(X_te))
    Path('resultados_classico_vies_fonte.json').write_text(json.dumps(
        {'tarefa': 'prever a base de origem (MC, SZ, NIH) com LBP+GLCM realcado', 'acuracia_teste': round(float(acc_fonte), 4),
         'acaso_classe_majoritaria': round(float(pd.Series(f_te).value_counts(normalize=True).max()), 4)}, indent=2), encoding='utf8')
    print(f'\nDiagnóstico de viés: LBP+GLCM acertam a BASE de origem em {acc_fonte:.1%} das imagens de teste')

    # modelo para o protótipo: o de maior F1 na validação entre SVM e RF
    chave = max(('svm', 'rf'), key=lambda k: melhores[k][0].f1_macro_validacao)
    best, modelo, desc = melhores[chave]
    salvar_modelo_classico('modelos/modelo_classico.joblib', modelo, desc,
                           f'{best.nome} ({best.descritores}, {best.versao})', versao=best.versao)
    print(f'\nModelo do protótipo: {best.nome} ({best.descritores}, {best.versao}) -> modelos/modelo_classico.joblib')


if __name__ == '__main__':
    main()
