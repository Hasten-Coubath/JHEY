import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ==========================================
# 1. CAMINHOS DO MODELO E DA NOVA IMAGEM
# ==========================================
CAMINHO_MODELO = "modelos_salvos/binario_SVM_HOG.joblib"
CAMINHO_IMAGEM = "caminho/para/nova_radiografia.jpeg"

# ==========================================
# 2. FUNÇÕES DE EXTRAÇÃO (IDÊNTICAS AO TREINO)
# ==========================================
def preparar_imagem(caminho_img, largura, altura, aplicar_clahe):
    img_original = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho_img}")

    img_res = cv2.resize(img_original, (largura, altura))
    if aplicar_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_suave = cv2.GaussianBlur(img_res, (3, 3), 0)
        img_proc = clahe.apply(img_suave)
    else:
        img_proc = img_res

    return img_original, img_proc

def extrair_caracteristicas(img, tecnica, largura, altura):
    if tecnica == 'HOG':
        hog_desc = cv2.HOGDescriptor((largura, altura), (16, 16), (8, 8), (8, 8), 9)
        return hog_desc.compute(img).flatten()

    elif tecnica == 'LBP':
        P, R = 8, 1
        lbp = local_binary_pattern(img, P, R, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        return hist.astype("float") / (hist.sum() + 1e-6)

    elif tecnica == 'GLCM':
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
    else:
        raise ValueError(f"Técnica desconhecida: {tecnica}")

# ==========================================
# 3. INFERÊNCIA E PLOTAGEM DO RESULTADO
# ==========================================
def classificar_e_plotar(caminho_modelo, caminho_imagem):
    # 1. Carregar pacote do modelo treinado
    pacote = joblib.load(caminho_modelo)
    modelo = pacote['modelo']
    scaler = pacote['scaler']
    tecnica = pacote['tecnica']
    nome_clf = pacote['nome_clf']
    classes = pacote['classes']
    aplicar_clahe = pacote['aplicar_clahe']
    largura = pacote['largura']
    altura = pacote['altura']

    # 2. Pré-processar imagem e extrair vetor de características
    img_original, img_proc = preparar_imagem(caminho_imagem, largura, altura, aplicar_clahe)
    vetor_features = extrair_caracteristicas(img_proc, tecnica, largura, altura).reshape(1, -1)

    # 3. Aplicar normalização (se o modelo exigir)
    if scaler is not None:
        vetor_features = scaler.transform(vetor_features)

    # 4. Predição
    indice_predito = modelo.predict(vetor_features)[0]
    classe_predita = classes[indice_predito]

    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(vetor_features)[0]
        confianca = probabilidades[indice_predito] * 100
        texto_titulo = f"Predição: {classe_predita} ({confianca:.1f}%)"
    else:
        probabilidades = None
        texto_titulo = f"Predição: {classe_predita}"

    # 5. Plotagem
    cor_destaque = '#2ecc71' if classe_predita == 'NORMAL' else '#e74c3c'

    if probabilidades is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={'width_ratios': [1.2, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))

    fig.suptitle(f"Modelo: {nome_clf} + {tecnica}", fontsize=13, fontweight='bold')

    # Imagem de Raio-X
    ax1.imshow(img_original, cmap='gray')
    ax1.set_title(texto_titulo, fontsize=13, fontweight='bold', color=cor_destaque, pad=10)
    ax1.axis('off')

    # Gráfico de Probabilidades por Classe
    if probabilidades is not None:
        cores_barras = ['#2ecc71' if c == 'NORMAL' else '#e74c3c' for c in classes]
        y_pos = np.arange(len(classes))
        barras = ax2.barh(y_pos, probabilidades * 100, color=cores_barras, height=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(classes, fontsize=11, fontweight='bold')
        ax2.invert_yaxis()
        ax2.set_xlim(0, 112)
        ax2.set_xlabel("Probabilidade (%)", fontsize=10)
        ax2.set_title("Confiança por Classe", fontsize=11)
        ax2.grid(axis='x', linestyle='--', alpha=0.5)

        for barra, prob in zip(barras, probabilidades):
            ax2.text(
                barra.get_width() + 2,
                barra.get_y() + barra.get_height() / 2,
                f"{prob*100:.1f}%",
                va='center', fontsize=10, fontweight='bold'
            )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    classificar_e_plotar(CAMINHO_MODELO, CAMINHO_IMAGEM)
