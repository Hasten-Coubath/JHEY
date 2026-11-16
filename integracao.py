"""Integrante 4: funções de integração usadas pelo protótipo e pelos outros integrantes.

1) Formato comum de predições -> salvar_predicoes(...)
   Todo classificador deve gerar resultados_<nome>/predicoes_teste.csv com as colunas:
       arquivo, classe_real, classe_predita, prob_normal, prob_pneumonia, prob_tuberculose
   (as colunas prob_* são opcionais, mas sem elas não dá para calcular AUC).

2) Contrato do modelo clássico -> salvar_modelo_classico(...) / carregar_modelo_classico(...)
   Um .joblib com o classificador já treinado (de preferência um Pipeline com o StandardScaler)
   e a lista de descritores usados, para o protótipo extrair as mesmas características.

3) Pré-processamento do Integrante 1 e extração de características do Integrante 2 para uma imagem nova.
"""
import csv
from pathlib import Path

import cv2
import numpy as np

CLASSES = ['normal', 'pneumonia', 'tuberculose']


# ------------------------------------------------------------ formato comum
def salvar_predicoes(destino, arquivos, reais, preditas, probabilidades=None):
    """Grava o CSV de predições no formato que avaliacao_integrante4.py lê.

    arquivos: nomes/caminhos das imagens de teste
    reais, preditas: rótulos em texto ('normal', 'pneumonia', 'tuberculose') ou índices 0..2
    probabilidades: matriz (n, 3) na ordem de CLASSES, ou None
    """
    para_texto = lambda v: CLASSES[int(v)] if not isinstance(v, str) else v.lower()
    destino = Path(destino); destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open('w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(['arquivo', 'classe_real', 'classe_predita'] + ([f'prob_{c}' for c in CLASSES] if probabilidades is not None else []))
        for i, (a, r, p) in enumerate(zip(arquivos, reais, preditas)):
            extra = list(map(float, probabilidades[i])) if probabilidades is not None else []
            w.writerow([str(a).replace('\\', '/'), para_texto(r), para_texto(p)] + extra)


# ------------------------------------------------------------ pré-processamento (Integrante 1)
# Parâmetros de processado/estatisticas.json: tamanho 512, percentis 0,5 e 99,5,
# CLAHE clip 2,0 grade 8, mediana 5. Conferido contra as imagens do zip:
# realcado = mediana5(CLAHE(basico)) reproduz o arquivo do zip (erro médio < 1 nível de cinza).
TAMANHO_BASE = 512


def preprocessar_basico(img_cinza, tamanho=TAMANHO_BASE):
    """Normaliza pelos percentis 0,5-99,5, completa até ficar quadrada (barras pretas) e redimensiona."""
    img = img_cinza.astype(np.float32)
    lo, hi = np.percentile(img, (0.5, 99.5))
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    img = (img * 255).astype(np.uint8)
    h, w = img.shape
    lado = max(h, w)
    quadrada = np.zeros((lado, lado), np.uint8)
    quadrada[(lado - h) // 2:(lado - h) // 2 + h, (lado - w) // 2:(lado - w) // 2 + w] = img
    return cv2.resize(quadrada, (tamanho, tamanho), interpolation=cv2.INTER_AREA)


def realcar(img_basico):
    """Versão 'realcado' a partir da 'basico': CLAHE (clip 2,0, grade 8x8) seguido de mediana 5x5."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.medianBlur(clahe.apply(img_basico), 5)


def preprocessar_realcado(img_cinza, tamanho=TAMANHO_BASE):
    return realcar(preprocessar_basico(img_cinza, tamanho))


# ------------------------------------------------------------ características (código do Integrante 2)
# Mesmos parâmetros de experimento_TOTAL_dataset_yeshua.py
_TAM_CLASSICO = 128
_hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
_clahe_classico = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocessar_classico(img_processada, versao):
    """Igual a preprocessar_imagem() do Integrante 2: 128x128; na versão 'basico' aplica gaussiana 3x3 + CLAHE."""
    img = cv2.resize(img_processada, (_TAM_CLASSICO, _TAM_CLASSICO))
    if versao == 'basico':
        img = _clahe_classico.apply(cv2.GaussianBlur(img, (3, 3), 0))
    return img


def extrair_hog(img):
    return _hog.compute(img).flatten()


def extrair_lbp(img, P=8, R=1):
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(img, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype('float')
    return hist / (hist.sum() + 1e-6)


def extrair_glcm(img):
    from skimage.feature import graycomatrix, graycoprops
    glcm = graycomatrix(img, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2], levels=256, symmetric=True, normed=True)
    return np.hstack([graycoprops(glcm, p).ravel() for p in ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation')])


EXTRATORES = {'hog': extrair_hog, 'lbp': extrair_lbp, 'glcm': extrair_glcm}


def extrair_caracteristicas(img_processada, descritores, versao):
    """img_processada = imagem 'basico' ou 'realcado' (512 px), como sai do Integrante 1."""
    img = preprocessar_classico(img_processada, versao)
    return np.hstack([EXTRATORES[d.lower()](img) for d in descritores])


# ------------------------------------------------------------ contrato do modelo clássico
def salvar_modelo_classico(destino, modelo, descritores, nome, versao='basico', classes=CLASSES):
    """Exemplo (Integrante 2):
        pipe = make_pipeline(StandardScaler(), SVC(probability=True, class_weight='balanced'))
        pipe.fit(X_treino, y_treino)
        salvar_modelo_classico('modelos/modelo_classico.joblib', pipe, ['glcm', 'lbp'], 'SVM (GLCM+LBP)', versao='basico')
    y_treino deve usar os índices 0, 1, 2 na ordem de CLASSES.
    """
    import joblib
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'modelo': modelo, 'descritores': list(descritores), 'nome': nome, 'versao': versao,
                 'classes': list(classes)}, destino)


def carregar_modelo_classico(caminho):
    import joblib
    pacote = joblib.load(caminho)
    faltando = {'modelo', 'descritores', 'classes'} - set(pacote)
    if faltando:
        raise ValueError(f'{caminho} não segue o contrato: faltam {sorted(faltando)}')
    return pacote


def prever_classico(pacote, imagens):
    """imagens: dict {'basico': img512, 'realcado': img512}; usa a versão com que o modelo foi treinado."""
    versao = pacote.get('versao', 'basico')
    x = extrair_caracteristicas(imagens[versao], pacote['descritores'], versao).reshape(1, -1)
    modelo, classes = pacote['modelo'], pacote['classes']
    pred = modelo.predict(x)[0]
    pred = classes[int(pred)] if not isinstance(pred, str) else pred
    probs = None
    if hasattr(modelo, 'predict_proba'):
        try:
            p = modelo.predict_proba(x)[0]
            probs = {classes[int(c)] if not isinstance(c, str) else c: float(v) for c, v in zip(modelo.classes_, p)}
        except Exception:
            probs = None
    return pred, probs
