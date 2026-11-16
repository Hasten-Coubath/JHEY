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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random

# ==========================================
# 1. CONFIGURAÇÕES GLOBAIS E CAMINHOS
# ==========================================
LARGURA, ALTURA = 128, 128

# Você pode alternar para 'realcado' alterando a string abaixo se desejar
TIPO_DADOS = "basico" # Opções: "basico" ou "realcado"
BASE_DIR = f"/home/emanuel/Documentos/TrabalhoPDIav3/processado/{TIPO_DADOS}"

CAMINHOS = {
    'train': os.path.join(BASE_DIR, "treino"),
    'test': os.path.join(BASE_DIR, "teste"),
    'val': os.path.join(BASE_DIR, "validacao")
}

CATEGORIAS = {'normal': 0, 'pneumonia': 1, 'tuberculose': 2}
NOME_CLASSES = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSE']

# Parâmetros de PDI
hog_desc = cv2.HOGDescriptor((LARGURA, ALTURA), (16, 16), (8, 8), (8, 8), 9)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ==========================================
# 2. FUNÇÕES DE PRÉ-PROCESSAMENTO E EXTRACÃO
# ==========================================
def preprocessar_imagem(caminho_img):
    img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        return None
    img_res = cv2.resize(img, (LARGURA, ALTURA))
    # Se os dados forem 'basico', aplicamos CLAHE e suavização. 
    # Se já forem 'realcado', você pode pular o filtro se preferir.
    if TIPO_DADOS == "basico":
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
    glcm = graycomatrix(img, distances=[1, 3], angles=[0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
    return np.hstack([
        graycoprops(glcm, 'contrast').ravel(),
        graycoprops(glcm, 'dissimilarity').ravel(),
        graycoprops(glcm, 'homogeneity').ravel(),
        graycoprops(glcm, 'energy').ravel(),
        graycoprops(graycoprops(glcm, 'correlation'), 'correlation').ravel() if 'correlation' in globals() else graycoprops(glcm, 'correlation').ravel()
    ])

# ==========================================
# 3. CARREGAMENTO DOS DADOS (COM OU SEM BALANCEAMENTO)
# ==========================================
def carregar_dados(divisao, balanceado=False, max_por_classe=300):
    caminho_divisao = CAMINHOS[divisao]
    
    dados = {
        'HOG': {0: [], 1: [], 2: []},
        'LBP': {0: [], 1: [], 2: []},
        'GLCM': {0: [], 1: [], 2: []}
    }
    
    print(f"Carregando dados de [{divisao}] ({'Balanceado' f' - Máx {max_por_classe}/classe' if balanceado else 'Completo'})...")
    
    for cat_nome, rotulo in CATEGORIAS.items():
        pasta_cat = os.path.join(caminho_divisao, cat_nome)
        if not os.path.exists(pasta_cat):
            print(f"Aviso: Pasta não encontrada -> {pasta_cat}")
            continue
            
        arquivos = [os.path.join(pasta_cat, arq) for arq in os.listdir(pasta_cat) if arq.endswith(('.jpeg', '.jpg', '.png'))]
        random.shuffle(arquivos)
        
        if balanceado:
            arquivos = arquivos[:max_por_classe]
            
        for caminho in arquivos:
            img = preprocessar_imagem(caminho)
            if img is not None:
                dados['HOG'][rotulo].append(extrair_hog(img))
                dados['LBP'][rotulo].append(extrair_lbp(img))
                dados['GLCM'][rotulo].append(extrair_glcm(img))
                
    if balanceado:
        # Encontra a menor quantidade entre as classes disponíveis para igualar
        qtds = [len(dados['HOG'][r]) for r in CATEGORIAS.values() if len(dados['HOG'][r]) > 0]
        menor_qtd = min(qtds) if qtds else 0
        print(f"-> {divisao} balanceado: Usando {menor_qtd} amostras por classe.")
        
        res = {}
        for tec in ['HOG', 'LBP', 'GLCM']:
            X = []
            y = []
            for rotulo in CATEGORIAS.values():
                amostras = dados[tec][rotulo][:menor_qtd]
                X.extend(amostras)
                y.extend([rotulo] * len(amostras))
            res[tec] = (np.array(X), np.array(y))
        return res
    else:
        res = {}
        for tec in ['HOG', 'LBP', 'GLCM']:
            X, y = [], []
            for rotulo in CATEGORIAS.values():
                amostras = dados[tec][rotulo]
                X.extend(amostras)
                y.extend([rotulo] * len(amostras))
            res[tec] = (np.array(X), np.array(y))
        return res

# ==========================================
# 4. EXECUTAR EXPERIMENTOS (LOOP PRINCIPAL)
# ==========================================
def rodar_experimento(balanceado=False):
    # Carrega treino e teste
    treino_por_tec = carregar_dados('train', balanceado=balanceado, max_por_classe=300)
    teste_por_tec = carregar_dados('test', balanceado=balanceado, max_por_classe=150)
    
    classificadores = {
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, random_state=42),
        'KNN (K=5)': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    for nome_clf, clf in classificadores.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        modo_txt = "Dataset Balanceado" if balanceado else "Dataset Completo"
        fig.suptitle(f'Multiclasse (3 classes) - {nome_clf} ({modo_txt})', fontsize=15)
        
        print(f"\n================ AVALIAÇÃO: {nome_clf} ({modo_txt}) ================")
        
        for i, tec in enumerate(['HOG', 'LBP', 'GLCM']):
            X_tr, y_tr = treino_por_tec[tec]
            X_te, y_te = teste_por_tec[tec]
            
            if len(X_tr) == 0 or len(X_te) == 0:
                print(f"[{tec}] Dados vazios para esta técnica. Pulando...")
                continue
                
            # Normalização (fundamental para SVM e KNN)
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)
            
            dados_tr = X_tr if nome_clf == 'Random Forest' else X_tr_scaled
            dados_te = X_te if nome_clf == 'Random Forest' else X_te_scaled
            
            # Treino e Predição
            clf.fit(dados_tr, y_tr)
            y_pred = clf.predict(dados_te)
            
            acc = accuracy_score(y_te, y_pred)
            print(f"[{tec:4s}] -> Acurácia Global: {acc*100:.2f}%")
            
            # Matriz de Confusão Multiclasse (3x3)
            cm = confusion_matrix(y_te, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=NOME_CLASSES, yticklabels=NOME_CLASSES)
            axes[i].set_title(f"{tec} + {nome_clf}\nAcc: {acc*100:.1f}%")
            axes[i].set_xlabel('Predito')
            axes[i].set_ylabel('Real')
            
        plt.tight_layout()
        plt.show()

# ==========================================
# 5. CHAMADA DOS TESTES
# ==========================================
if __name__ == "__main__":
    print(">>> INICIANDO TESTES COM O DATASET COMPLETO (DESBALANCEADO) <<<")
    rodar_experimento(balanceado=False)
    
    print("\n>>> INICIANDO TESTES COM O DATASET BALANCEADO <<<")
    rodar_experimento(balanceado=True)