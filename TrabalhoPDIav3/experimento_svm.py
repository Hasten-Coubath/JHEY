import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

# Configurações Globais
LARGURA, ALTURA = 128, 128
CAMINHO_KAGGLE = "chest_xray"  # Ajuste para a sua pasta

# Descritores PDI
hog_desc = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocessar_imagem(caminho_img):
    img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img_res = cv2.resize(img, (LARGURA, ALTURA))
    img_suave = cv2.GaussianBlur(img_res, (3, 3), 0)
    return clahe.apply(img_suave)

def extrair_hog(img):
    return hog_desc.compute(img).flatten()

def extrair_lbp(img, P=8, R=1):
    lbp = local_binary_pattern(img, P, R, method="uniform")
    # Histograma de padrões uniformes LBP
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extrair_glcm(img):
    # GLCM precisa de imagem uint8 quantizada (reduzir níveis ajuda no desempenho)
    glcm = graycomatrix(img, distances=[1, 3], angles=[0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').ravel()
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    energy = graycoprops(glcm, 'energy').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()
    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])

def carregar_dados_todas_tecnicas(subpasta):
    X_hog, X_lbp, X_glcm, y = [], [], [], []
    caminho_divisao = os.path.join(CAMINHO_KAGGLE, subpasta)
    categorias = {'NORMAL': 0, 'PNEUMONIA': 1}
    
    print(f"Carregando e extraindo características ({subpasta})...")
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

# 1. Carregar Dados
X_tr_hog, X_tr_lbp, X_tr_glcm, y_train = carregar_dados_todas_tecnicas('train')
X_te_hog, X_te_lbp, X_te_glcm, y_test = carregar_dados_todas_tecnicas('test')

tecnicas = {
    'HOG': (X_tr_hog, X_te_hog),
    'LBP': (X_tr_lbp, X_te_lbp),
    'GLCM': (X_tr_glcm, X_te_glcm)
}

resultados = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Comparação de Técnicas de PDI com SVM', fontsize=16)

# 2. Loop de Avaliação com SVM
for i, (nome_tec, (X_tr, X_te)) in enumerate(tecnicas.items()):
    # Normalização dos dados (fundamental pro SVM)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    # Classificador SVM
    model = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42)
    model.fit(X_tr_scaled, y_train)
    y_pred = model.predict(X_te_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    resultados[nome_tec] = {'Acc': acc, 'Recall': rec, 'F1': f1}
    
    # Matriz de Confusão Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
    axes[i].set_title(f"{nome_tec} + SVM\nAcc: {acc*100:.1f}% | Recall: {rec*100:.1f}%")
    axes[i].set_xlabel('Predito')
    axes[i].set_ylabel('Real')

plt.tight_layout()
plt.show()

# Exibir Resumo no Terminal
print("\n=== RESUMO DE DESEMPENHO (SVM) ===")
for tec, metrics in resultados.items():
    print(f"{tec:5s} -> Acurácia: {metrics['Acc']*100:.2f}% | Sensibilidade (Recall): {metrics['Recall']*100:.2f}% | F1-Score: {metrics['F1']*100:.2f}%")



"""
=== RESUMO DE DESEMPENHO (SVM) ===
HOG   -> Acurácia: 75.80% | Sensibilidade (Recall): 98.97% | F1-Score: 83.64%
LBP   -> Acurácia: 72.60% | Sensibilidade (Recall): 84.36% | F1-Score: 79.37%
GLCM  -> Acurácia: 74.84% | Sensibilidade (Recall): 84.36% | F1-Score: 80.74%

"""