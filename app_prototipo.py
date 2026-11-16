"""Integrante 4: protótipo integrado de apoio à triagem de raio-X de tórax.

Executar:  streamlit run app_prototipo.py
Ferramenta acadêmica de apoio à decisão. NÃO substitui o laudo do médico radiologista.
"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from cnn_integrante3 import CLASSES, criar_modelo, criar_transformacoes, gradcam
from integracao import carregar_modelo_classico, preprocessar_basico, prever_classico, realcar

RAIZ = Path(__file__).parent
CNN_PADRAO = RAIZ / 'resultados_cnn_realcado' / 'melhor_modelo.pt'
CLASSICO_PADRAO = RAIZ / 'modelos' / 'modelo_classico.joblib'
MEDIA, DESVIO = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
ROTULOS = {'normal': 'Normal', 'pneumonia': 'Pneumonia', 'tuberculose': 'Tuberculose'}


@st.cache_resource
def carregar_cnn(caminho):
    ckpt = torch.load(caminho, map_location='cpu', weights_only=True)
    modelo = criar_modelo(pretreinado=False)
    modelo.load_state_dict(ckpt['state_dict'])
    modelo.eval()
    return modelo, ckpt.get('tamanho', 224), ckpt.get('versao', '?')


@st.cache_resource
def carregar_classico(caminho):
    return carregar_modelo_classico(caminho)


def para_tensor(img_cinza, tamanho):
    """Mesma transformação de avaliação usada no treino da CNN (Integrante 3)."""
    tensor = criar_transformacoes(tamanho)[1](Image.fromarray(img_cinza).convert('RGB'))
    rgb = np.clip(tensor.numpy().transpose(1, 2, 0) * DESVIO + MEDIA, 0, 1)
    return tensor, rgb


def sobrepor(rgb, mapa, alpha=0.4):
    mapa = cv2.resize(mapa, (rgb.shape[1], rgb.shape[0]))
    cor = cv2.applyColorMap(np.uint8(mapa * 255), cv2.COLORMAP_JET)[:, :, ::-1] / 255.0
    return np.clip((1 - alpha) * rgb + alpha * cor, 0, 1)


def barras_probabilidade(probs):
    for c in CLASSES:
        v = probs.get(c)
        if v is None:
            continue
        st.write(f'{ROTULOS[c]}: **{v:.1%}**')
        st.progress(float(v))


st.set_page_config(page_title='Segunda Leitura - triagem de raio-X', layout='wide')
st.title('Segunda Leitura: apoio à triagem de raio-X de tórax')
st.caption('Protótipo acadêmico (P.D.I. – UNIFOR). Sugere normal, pneumonia ou tuberculose e mostra as regiões '
           'que mais pesaram na decisão. **Não substitui o laudo do médico radiologista.**')

with st.sidebar:
    st.header('Configuração')
    caminho_cnn = st.text_input('Modelo CNN (.pt)', str(CNN_PADRAO))
    caminho_cls = st.text_input('Modelo clássico (.joblib)', str(CLASSICO_PADRAO))
    tipo = st.radio('A imagem enviada é', ['original (sem tratamento)', "versão 'basico' do Integrante 1",
                                          "versão 'realcado' do Integrante 1"], index=0)
    alpha = st.slider('Intensidade do Grad-CAM', 0.1, 0.8, 0.4, 0.05)

arquivo = st.file_uploader('Envie uma radiografia de tórax frontal (PNG ou JPG)', type=['png', 'jpg', 'jpeg'])
if arquivo is None:
    st.info('Envie uma imagem para começar.')
    st.stop()

original = np.array(Image.open(arquivo).convert('L'))
if tipo.startswith('original'):
    basico = preprocessar_basico(original)
    processada = realcar(basico)
elif 'basico' in tipo:
    basico, processada = original, realcar(original)
else:  # já realçada: não dá para voltar à 'basico'; usa a própria imagem nas duas versões
    basico = processada = original
imagens = {'basico': basico, 'realcado': processada}

c1, c2 = st.columns(2)
c1.image(original, caption='Imagem enviada', width='stretch', clamp=True)
c2.image(processada, caption='Pré-processada (normalização, CLAHE e mediana 5x5)', width='stretch', clamp=True)

st.divider()
col_cnn, col_cls = st.columns(2)
pred_cnn = pred_cls = None

with col_cnn:
    st.subheader('CNN (ResNet18 + transferência de aprendizado)')
    if not Path(caminho_cnn).exists():
        st.warning(f'Modelo não encontrado: {caminho_cnn}')
    else:
        modelo, tamanho, versao = carregar_cnn(caminho_cnn)
        tensor, rgb = para_tensor(processada, tamanho)
        with torch.no_grad():
            p = torch.softmax(modelo(tensor.unsqueeze(0)), dim=1)[0].numpy()
        pred_cnn = CLASSES[int(p.argmax())]
        st.metric('Sugestão', ROTULOS[pred_cnn], f'confiança {p.max():.1%}', delta_color='off')
        barras_probabilidade(dict(zip(CLASSES, map(float, p))))
        mapa = gradcam(modelo, tensor, int(p.argmax()), torch.device('cpu'))
        st.image(sobrepor(rgb, mapa, alpha), caption='Grad-CAM: regiões que mais influenciaram a sugestão',
                 width='stretch', clamp=True)

with col_cls:
    st.subheader('Abordagem tradicional de PDI')
    if not Path(caminho_cls).exists():
        st.info('Modelo clássico não encontrado. Gere com `python classico_3classes.py`.')
    else:
        pacote = carregar_classico(caminho_cls)
        st.caption(f"{pacote.get('nome', 'Modelo clássico')} | descritores: {', '.join(pacote['descritores']).upper()}")
        pred_cls, probs = prever_classico(pacote, imagens)
        st.metric('Sugestão', ROTULOS.get(pred_cls, pred_cls))
        if probs:
            barras_probabilidade(probs)

st.divider()
sugestoes = [s for s in (pred_cnn, pred_cls) if s]
if sugestoes:
    if any(s != 'normal' for s in sugestoes):
        st.error('Prioridade de leitura: **ALTA**. Ao menos um modelo sugere alteração pulmonar.')
    else:
        st.success('Prioridade de leitura: **rotina**. Nenhum modelo sugere alteração.')
    if len(sugestoes) == 2 and pred_cnn != pred_cls:
        st.warning('Os dois modelos discordam: o caso merece atenção do radiologista.')
st.caption('A sugestão é apenas uma ordem de prioridade para a fila de laudos. A decisão clínica é do profissional de saúde.')
