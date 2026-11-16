"""Integrante 4: teste em base diferente da usada no treino (objetivo específico 4 da Etapa 1).

Tuberculose x normal é a única tarefa que existe em duas bases (Shenzhen e Montgomery), então:
  - treino em Shenzhen (treino + validação) e teste em Shenzhen (teste)  -> desempenho "na mesma base"
  - treino em Shenzhen (todas as imagens)    e teste em Montgomery (todas) -> desempenho "em base nova"
Modelos: as três configurações tradicionais escolhidas em classico_3classes.py e,
no lugar da CNN (sem acesso aos pesos ImageNet originais para retreinar), uma regressão logística
sobre as características da ResNet18 até layer3 (camadas que ficaram congeladas no treino do Integrante 3).

Uso:  python generalizacao_entre_bases.py --dados segunda_leitura_processado.zip
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classico_3classes import COMBINACOES, classificadores, extrair_tudo, listar_imagens

TB, NORMAL = 2, 0


def coletar(tabela, versao, descritores, fonte, splits):
    X, y, arqs = [], [], []
    for s in splits:
        t = tabela[(versao, s)]
        mask = np.array([Path(a).name.startswith(fonte + '_') for a in t['arquivos']]) & np.isin(t['y'], [NORMAL, TB])
        X.append(np.hstack([t[d] for d in descritores])[mask]); y.append(t['y'][mask])
        arqs += [a for a, m in zip(t['arquivos'], mask) if m]
    return np.vstack(X), (np.concatenate(y) == TB).astype(int), arqs


def metricas(y, p):
    return {'acuracia_balanceada': round(balanced_accuracy_score(y, p), 3),
            'sensibilidade_tb': round(recall_score(y, p, pos_label=1), 3),
            'especificidade': round(recall_score(y, p, pos_label=0), 3), 'n': int(len(y))}


def caracteristicas_resnet(fonte_dados, versao, fonte, splits, ckpt):
    """Saída média de layer3 da ResNet18 (pesos ImageNet que ficaram congelados no treino da CNN)."""
    import torch
    from PIL import Image
    from cnn_integrante3 import criar_modelo, criar_transformacoes
    modelo = criar_modelo(pretreinado=False)
    modelo.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True)['state_dict'])
    modelo.eval()
    corpo = torch.nn.Sequential(modelo.conv1, modelo.bn1, modelo.relu, modelo.maxpool, modelo.layer1, modelo.layer2, modelo.layer3)
    t = criar_transformacoes(224)[1]
    X, y = [], []
    for s in splits:
        itens, ler = listar_imagens(fonte_dados, versao, s)
        for nome, c in itens:
            if Path(nome).name.startswith(fonte + '_') and c in (NORMAL, TB):
                with torch.no_grad():
                    f = corpo(t(Image.fromarray(ler(nome)).convert('RGB'))[None]).mean(dim=(2, 3))[0].numpy()
                X.append(f); y.append(int(c == TB))
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dados', default='segunda_leitura_processado.zip')
    ap.add_argument('--cache', default='caracteristicas_cache.npz')
    ap.add_argument('--cnn', default='resultados_cnn_realcado/melhor_modelo.pt')
    args = ap.parse_args()
    tabela = extrair_tudo(args.dados, Path(args.cache))
    escolhas = {k: json.loads(Path(f'resultados_{k}/configuracao.json').read_text(encoding='utf8')) for k in ('svm', 'rf', 'knn')}
    linhas = []
    for chave, (nome, fabrica) in classificadores().items():
        cfg = escolhas[chave]
        desc, versao = COMBINACOES[cfg['descritores']], cfg['versao']
        X_a, y_a, _ = coletar(tabela, versao, desc, 'SZ', ['treino', 'validacao'])
        X_b, y_b, _ = coletar(tabela, versao, desc, 'SZ', ['teste'])
        mesma = metricas(y_b, fabrica().fit(X_a, y_a).predict(X_b))
        X_t, y_t, _ = coletar(tabela, versao, desc, 'SZ', ['treino', 'validacao', 'teste'])
        X_m, y_m, _ = coletar(tabela, versao, desc, 'MC', ['treino', 'validacao', 'teste'])
        nova = metricas(y_m, fabrica().fit(X_t, y_t).predict(X_m))
        linhas.append({'modelo': f"{nome} ({cfg['descritores']})", **{f'mesma_base_{k}': v for k, v in mesma.items()},
                       **{f'base_nova_{k}': v for k, v in nova.items()}})
        print(f"{linhas[-1]['modelo']:28s} Shenzhen {mesma['acuracia_balanceada']:.3f} -> Montgomery {nova['acuracia_balanceada']:.3f}")

    if Path(args.cnn).exists():
        fab = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000))
        X_a, y_a = caracteristicas_resnet(args.dados, 'realcado', 'SZ', ['treino', 'validacao'], args.cnn)
        X_b, y_b = caracteristicas_resnet(args.dados, 'realcado', 'SZ', ['teste'], args.cnn)
        X_m, y_m = caracteristicas_resnet(args.dados, 'realcado', 'MC', ['treino', 'validacao', 'teste'], args.cnn)
        mesma = metricas(y_b, fab().fit(X_a, y_a).predict(X_b))
        nova = metricas(y_m, fab().fit(np.vstack([X_a, X_b]), np.concatenate([y_a, y_b])).predict(X_m))
        linhas.append({'modelo': 'ResNet18 (ImageNet) + reg. logística', **{f'mesma_base_{k}': v for k, v in mesma.items()},
                       **{f'base_nova_{k}': v for k, v in nova.items()}})
        print(f"{linhas[-1]['modelo']:28s} Shenzhen {mesma['acuracia_balanceada']:.3f} -> Montgomery {nova['acuracia_balanceada']:.3f}")

    t = pd.DataFrame(linhas)
    Path('resultados_avaliacao').mkdir(exist_ok=True)
    t.to_csv('resultados_avaliacao/generalizacao_entre_bases.csv', index=False)
    print('\nSalvo em resultados_avaliacao/generalizacao_entre_bases.csv')


if __name__ == '__main__':
    main()
