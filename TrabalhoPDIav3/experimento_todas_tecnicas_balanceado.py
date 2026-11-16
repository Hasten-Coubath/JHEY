import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import random

# Configurações Globais
LARGURA, ALTURA = 128, 128
CAMINHO_KAGGLE = "chest_xray"  # Ajuste para o caminho da sua pasta do dataset
MAX_TREINO_POR_CLASSE = 600  # Amostras balanceadas para treino (ex: 300 Normais, 300 Pneumonia)
MAX_TESTE_POR_CLASSE = 300   # Amostras balanceadas para teste (ex: 150 Normais, 150 Pneumonia)

# Inicializadores PDI
hog_desc = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocessar_imagem(caminho_img):
    img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img_res = cv2.resize(img, (LARGURA, ALTURA))
    img_suave = cv2.GaussianBlur(img_res, (3, 3), 0)
    return clahe.apply(img_suave)

def extrair_hog(img):
    return hog_desc.compute(img).flatten()

def extrair_lbp(img, P=8, R=1):
    lbp = local_binary_pattern(img, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    return hist.astype("float") / (hist.sum() + 1e-6)

def extrair_glcm(img):
    glcm = graycomatrix(img, distances=[1, 3], angles=[0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
    return np.hstack([
        graycoprops(glcm, 'contrast').ravel(),
        graycoprops(glcm, 'dissimilarity').ravel(),
        graycoprops(glcm, 'homogeneity').ravel(),
        graycoprops(glcm, 'energy').ravel(),
        graycoprops(glcm, 'correlation').ravel()
    ])

def carregar_dados_balanceados(subpasta, max_por_classe):
    caminho_divisao = os.path.join(CAMINHO_KAGGLE, subpasta)
    categorias = {'NORMAL': 0, 'PNEUMONIA': 1}
    
    dados = {
        'HOG': {0: [], 1: []},
        'LBP': {0: [], 1: []},
        'GLCM': {0: [], 1: []}
    }
    
    print(f"Carregando e balanceando dados de: {subpasta}...")
    for cat_nome, rotulo in categorias.items():
        pasta_cat = os.path.join(caminho_divisao, cat_nome)
        if not os.path.exists(pasta_cat): continue
        
        arquivos = [os.path.join(pasta_cat, arq) for arq in os.listdir(pasta_cat) if arq.endswith(('.jpeg', '.jpg', '.png'))]
        random.shuffle(arquivos)
        arquivos = arquivos[:max_por_classe]
        
        for caminho in arquivos:
            img = preprocessar_imagem(caminho)
            if img is not None:
                dados['HOG'][rotulo].append(extrair_hog(img))
                dados['LBP'][rotulo].append(extrair_lbp(img))
                dados['GLCM'][rotulo].append(extrair_glcm(img))
                
    menor_qtd = min(len(dados['HOG'][0]), len(dados['HOG'][1]))
    print(f"-> {subpasta}: NORMAL = {len(dados['HOG'][0])} | PNEUMONIA = {len(dados['HOG'][1])} -> Usando {menor_qtd} de cada.")
    
    res = {}
    for tec in ['HOG', 'LBP', 'GLCM']:
        X = dados[tec][0][:menor_qtd] + dados[tec][1][:menor_qtd]
        y = [0] * menor_qtd + [1] * menor_qtd
        res[tec] = (np.array(X), np.array(y))
        
    return res

# 1. Carregar Datasets Balanceados
treino_por_tec = carregar_dados_balanceados('train', MAX_TREINO_POR_CLASSE)
teste_por_tec = carregar_dados_balanceados('test', MAX_TESTE_POR_CLASSE)

# 2. Definir Classificadores
classificadores = {
    'SVM (RBF)': SVC(kernel='rbf', C=1.0, random_state=42),
    'KNN (K=5)': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# 3. Executar Experimentos e Plotar
for nome_clf, clf in classificadores.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Comparação de Técnicas de PDI com {nome_clf} (Dados Balanceados)', fontsize=16)
    
    print(f"\n================ AVALIAÇÃO: {nome_clf} ===============")
    
    for i, tec in enumerate(['HOG', 'LBP', 'GLCM']):
        X_tr, y_tr = treino_por_tec[tec]
        X_te, y_te = teste_por_tec[tec]
        
        # Padronização (Essencial para SVM e KNN, opcional para Random Forest)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        
        # Treinamento e Predição
        dados_tr = X_tr if nome_clf == 'Random Forest' else X_tr_scaled
        dados_te = X_te if nome_clf == 'Random Forest' else X_te_scaled
        
        clf.fit(dados_tr, y_tr)
        y_pred = clf.predict(dados_te)
        
        acc = accuracy_score(y_te, y_pred)
        rec = recall_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred)
        cm = confusion_matrix(y_te, y_pred)
        
        print(f"[{tec:4s}] -> Acurácia: {acc*100:.2f}% | Sensibilidade (Recall): {rec*100:.2f}% | F1-Score: {f1*100:.2f}%")
        
        # Plot Matriz de Confusão
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
        axes[i].set_title(f"{tec} + {nome_clf}\nAcc: {acc*100:.1f}% | Recall: {rec*100:.1f}%")
        axes[i].set_xlabel('Predito')
        axes[i].set_ylabel('Real')
        
    plt.tight_layout()
    plt.show()