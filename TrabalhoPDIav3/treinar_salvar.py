import os
import random
import cv2
import joblib
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ==========================================
# 1. CONFIGURAÇÕES (AJUSTE CONFORME O EXPERIMENTO)
# ==========================================
LARGURA, ALTURA = 128, 128
PASTA_SAIDA = "modelos_salvos"

# Escolha "binario" (pasta chest_xray) ou "multiclasse" (pasta processado)
MODO_EXPERIMENTO = "binario"  # Opções: "binario" ou "multiclasse"
TIPO_DADOS = "basico"         # Usado no modo multiclasse: "basico" ou "realcado"

BALANCEADO = False            # True para limitar amostras por classe; False para usar tudo
MAX_TREINO_POR_CLASSE = 300
MAX_TESTE_POR_CLASSE = 150

if MODO_EXPERIMENTO == "binario":
    CAMINHO_TREINO = os.path.join("chest_xray", "train")
    CAMINHO_TESTE = os.path.join("chest_xray", "test")
    CATEGORIAS = {'NORMAL': 0, 'PNEUMONIA': 1}
    NOME_CLASSES = ['NORMAL', 'PNEUMONIA']
    APLICAR_CLAHE = True
else:
    base_dir = os.path.join(os.getcwd(), "processado", TIPO_DADOS)
    CAMINHO_TREINO = os.path.join(base_dir, "treino")
    CAMINHO_TESTE = os.path.join(base_dir, "teste")
    CATEGORIAS = {'normal': 0, 'pneumonia': 1, 'tuberculose': 2}
    NOME_CLASSES = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSE']
    APLICAR_CLAHE = (TIPO_DADOS == "basico")

# ==========================================
# 2. PRÉ-PROCESSAMENTO E EXTRAÇÃO DE CARACTERÍSTICAS
# ==========================================
hog_desc = cv2.HOGDescriptor((LARGURA, ALTURA), (16, 16), (8, 8), (8, 8), 9)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocessar_imagem(caminho_img):
    img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img_res = cv2.resize(img, (LARGURA, ALTURA))
    if APLICAR_CLAHE:
        img_suave = cv2.GaussianBlur(img_res, (3, 3), 0)
        return clahe.apply(img_suave)
    return img_res

def extrair_hog(img):
    return hog_desc.compute(img).flatten()

def extrair_lbp(img, P=8, R=1):
    lbp = local_binary_pattern(img, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    return hist.astype("float") / (hist.sum() + 1e-6)

def extrair_glcm(img):
    glcm = graycomatrix(
        img, distances=[1, 3], angles=[0, np.pi/4, np.pi/2],
        levels=256, symmetric=True, normed=True
    )
    return np.hstack([
        graycoprops(glcm, 'contrast').ravel(),
        graycoprops(glcm, 'dissimilarity').ravel(),
        graycoprops(glcm, 'homogeneity').ravel(),
        graycoprops(glcm, 'energy').ravel(),
        graycoprops(glcm, 'correlation').ravel()
    ])

def carregar_dados(caminho_divisao, balanceado=False, max_por_classe=300):
    dados = {tec: {r: [] for r in CATEGORIAS.values()} for tec in ['HOG', 'LBP', 'GLCM']}
    print(f"Carregando imagens de: {caminho_divisao}...")

    for cat_nome, rotulo in CATEGORIAS.items():
        pasta_cat = os.path.join(caminho_divisao, cat_nome)
        if not os.path.exists(pasta_cat):
            print(f"Aviso: Pasta não encontrada -> {pasta_cat}")
            continue

        arquivos = [
            os.path.join(pasta_cat, arq)
            for arq in os.listdir(pasta_cat)
            if arq.lower().endswith(('.jpeg', '.jpg', '.png'))
        ]
        if balanceado:
            random.seed(42)
            random.shuffle(arquivos)
            arquivos = arquivos[:max_por_classe]

        for caminho in arquivos:
            img = preprocessar_imagem(caminho)
            if img is not None:
                dados['HOG'][rotulo].append(extrair_hog(img))
                dados['LBP'][rotulo].append(extrair_lbp(img))
                dados['GLCM'][rotulo].append(extrair_glcm(img))

    if balanceado:
        qtds = [len(dados['HOG'][r]) for r in CATEGORIAS.values() if len(dados['HOG'][r]) > 0]
        menor_qtd = min(qtds) if qtds else 0
    else:
        menor_qtd = None

    resultado = {}
    for tec in ['HOG', 'LBP', 'GLCM']:
        X, y = [], []
        for rotulo in CATEGORIAS.values():
            amostras = dados[tec][rotulo][:menor_qtd] if balanceado else dados[tec][rotulo]
            X.extend(amostras)
            y.extend([rotulo] * len(amostras))
        resultado[tec] = (np.array(X), np.array(y))

    return resultado

# ==========================================
# 3. TREINAMENTO E EXPORTAÇÃO DOS MODELOS
# ==========================================
if __name__ == "__main__":
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    treino = carregar_dados(CAMINHO_TREINO, balanceado=BALANCEADO, max_por_classe=MAX_TREINO_POR_CLASSE)
    teste = carregar_dados(CAMINHO_TESTE, balanceado=BALANCEADO, max_por_classe=MAX_TESTE_POR_CLASSE)

    peso_classe = None if BALANCEADO else 'balanced'

    # probability=True no SVM permite exibir a confiança (%) na predição
    classificadores = {
        'SVM': SVC(kernel='rbf', C=1.0, class_weight=peso_classe, probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight=peso_classe, random_state=42)
    }

    print(f"\nIniciando treinamento e salvamento em '{PASTA_SAIDA}/'...")

    for nome_clf, modelo_base in classificadores.items():
        for tec in ['HOG', 'LBP', 'GLCM']:
            X_tr, y_tr = treino[tec]
            X_te, y_te = teste[tec]

            if len(X_tr) == 0:
                continue

            # Clona os parâmetros para criar uma instância limpa a cada iteração
            clf = modelo_base.__class__(**modelo_base.get_params())

            if nome_clf == 'RandomForest':
                scaler = None
                dados_tr, dados_te = X_tr, X_te
            else:
                scaler = StandardScaler()
                dados_tr = scaler.fit_transform(X_tr)
                dados_te = scaler.transform(X_te)

            clf.fit(dados_tr, y_tr)
            acc = accuracy_score(y_te, clf.predict(dados_te)) if len(X_te) > 0 else 0.0

            nome_arquivo = f"{MODO_EXPERIMENTO}_{nome_clf}_{tec}.joblib"
            caminho_salvar = os.path.join(PASTA_SAIDA, nome_arquivo)

            # Salva o modelo junto com o scaler e os metadados necessários para inferência
            pacote = {
                'modelo': clf,
                'scaler': scaler,
                'tecnica': tec,
                'nome_clf': nome_clf,
                'classes': NOME_CLASSES,
                'aplicar_clahe': APLICAR_CLAHE,
                'largura': LARGURA,
                'altura': ALTURA
            }
            joblib.dump(pacote, caminho_salvar)
            print(f"[SALVO] {nome_arquivo:<35} | Acurácia Teste: {acc*100:.2f}%")
