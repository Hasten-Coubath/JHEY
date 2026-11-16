import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

LARGURA, ALTURA = 128, 128
CAMINHO_KAGGLE = "chest_xray"

hog_desc = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocessar_imagem(caminho_img):
    img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img_res = cv2.resize(img, (LARGURA, ALTURA))
    img_suave = cv2.GaussianBlur(img_res, (3, 3), 0)
    return clahe.apply(img_suave)

def extrair_hog(img): return hog_desc.compute(img).flatten()

def extrair_lbp(img, P=8, R=1):
    lbp = local_binary_pattern(img, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    return hist / (hist.sum() + 1e-6)

def extrair_glcm(img):
    glcm = graycomatrix(img, distances=[1, 3], angles=[0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
    return np.hstack([
        graycoprops(glcm, 'contrast').ravel(),
        graycoprops(glcm, 'dissimilarity').ravel(),
        graycoprops(glcm, 'homogeneity').ravel(),
        graycoprops(glcm, 'energy').ravel(),
        graycoprops(glcm, 'correlation').ravel()
    ])

def carregar_dados_todas_tecnicas(subpasta):
    X_hog, X_lbp, X_glcm, y = [], [], [], []
    caminho_divisao = os.path.join(CAMINHO_KAGGLE, subpasta)
    categorias = {'NORMAL': 0, 'PNEUMONIA': 1}
    for cat_nome, rotulo in categorias.items():
        pasta_cat = os.path.join(caminho_divisao, cat_nome)
        if not os.path.exists(pasta_cat): continue
        for arq in os.listdir(pasta_cat):
            if arq.endswith(('.jpeg', '.jpg', '.png')):
                caminho = os.path.join(pasta_cat, arq)
                img = preprocessar_imagem(caminho)
                if img is not None:
                    X_hog.append(extrair_hog(img))
                    X_lbp.append(extrair_lbp(img))
                    X_glcm.append(extrair_glcm(img))
                    y.append(rotulo)
    return np.array(X_hog), np.array(X_lbp), np.array(X_glcm), np.array(y)

X_tr_hog, X_tr_lbp, X_tr_glcm, y_train = carregar_dados_todas_tecnicas('train')
X_te_hog, X_te_lbp, X_te_glcm, y_test = carregar_dados_todas_tecnicas('test')

tecnicas = {'HOG': (X_tr_hog, X_te_hog), 'LBP': (X_tr_lbp, X_te_lbp), 'GLCM': (X_tr_glcm, X_te_glcm)}
resultados = {}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Comparação de Técnicas de PDI com KNN (K=5)', fontsize=16)

for i, (nome_tec, (X_tr, X_te)) in enumerate(tecnicas.items()):
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    # Classificador KNN
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_tr_scaled, y_train)
    y_pred = knn.predict(X_te_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    resultados[nome_tec] = {'Acc': acc, 'Recall': rec, 'F1': f1}
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[i],
                xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
    axes[i].set_title(f"{nome_tec} + KNN\nAcc: {acc*100:.1f}% | Recall: {rec*100:.1f}%")
    axes[i].set_xlabel('Predito')
    axes[i].set_ylabel('Real')

plt.tight_layout()
plt.show()

print("\n=== RESUMO DE DESEMPENHO (KNN) ===")
for tec, metrics in resultados.items():
    print(f"{tec:5s} -> Acurácia: {metrics['Acc']*100:.2f}% | Sensibilidade: {metrics['Recall']*100:.2f}% | F1-Score: {metrics['F1']*100:.2f}%")



"""
=== RESUMO DE DESEMPENHO (KNN) ===
HOG   -> Acurácia: 79.97% | Sensibilidade: 92.82% | F1-Score: 85.28%
LBP   -> Acurácia: 70.03% | Sensibilidade: 89.74% | F1-Score: 78.92%
GLCM  -> Acurácia: 72.28% | Sensibilidade: 91.54% | F1-Score: 80.50%
"""