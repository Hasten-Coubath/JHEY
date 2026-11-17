"""Integrante 4: protótipo integrado de apoio à triagem de raio-X de tórax.

Executar:  streamlit run app_prototipo.py
Ferramenta acadêmica de apoio à decisão. NÃO substitui o laudo do médico radiologista.

Organização da tela (padrões de interfaces de apoio à decisão clínica):
  - o que importa primeiro: prioridade de leitura e concordância entre os modelos no topo;
  - visualizador de imagem com fundo escuro (como nas estações de radiologia), em abas:
    original, pré-processada, Grad-CAM e segmentação pulmonar;
  - estados sempre com texto + ícone (não só cor) e probabilidades apresentadas como estimativas;
  - transparência: desempenho dos modelos no teste e limitações conhecidas ao alcance de um clique.
"""
import html
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from cnn_integrante3 import CLASSES, criar_modelo, criar_transformacoes, gradcam
from integracao import carregar_modelo_classico, preprocessar_basico, prever_classico, realcar
from segmentacao_pulmonar import segmentar_pulmao

RAIZ = Path(__file__).parent
CNN_PADRAO = RAIZ / 'resultados_cnn_realcado' / 'melhor_modelo.pt'
CLASSICO_PADRAO = RAIZ / 'modelos' / 'modelo_classico.joblib'
TABELA_METRICAS = RAIZ / 'resultados_avaliacao' / 'tabela_comparativa.csv'
MEDIA, DESVIO = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
ROTULOS = {'normal': 'Normal', 'pneumonia': 'Pneumonia', 'tuberculose': 'Tuberculose'}
TIPOS_IMAGEM = ['Radiografia original (sem tratamento)', 'Pré-processada: versão básica',
                'Pré-processada: versão realçada']

ESTILO = """
<style>
#MainMenu, footer, [data-testid="stDecoration"] {display: none;}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1320px;}
section[data-testid="stSidebar"] {border-right: 1px solid #E2E8F0;}
section[data-testid="stSidebar"] h3 {font-size: .78rem; text-transform: uppercase; letter-spacing: .07em;
  color: #64748B; margin: .6rem 0 .2rem 0;}

.sl-topo {display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
  background: linear-gradient(135deg, #0B2540 0%, #12395E 100%); color: #FFFFFF;
  padding: 18px 24px; border-radius: 14px; margin-bottom: 18px;}
.sl-marca {display: flex; align-items: center; gap: 14px;}
.sl-logo {width: 44px; height: 44px; border-radius: 11px; background: #0E7490; display: flex;
  align-items: center; justify-content: center; font-weight: 700; font-size: 1rem; letter-spacing: .03em;}
.sl-titulo {font-size: 1.3rem; font-weight: 700; margin: 0; line-height: 1.2;}
.sl-subtitulo {font-size: .85rem; color: #CBD5E1; margin: 2px 0 0 0;}
.sl-selo {font-size: .74rem; color: #E2E8F0; border: 1px solid rgba(226,232,240,.35); padding: 5px 12px;
  border-radius: 999px; white-space: nowrap;}

div[class*="st-key-cartao"] {background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
  padding: 18px 20px 14px 20px; box-shadow: 0 1px 2px rgba(15,23,42,.04);}
.sl-rotulo {font-size: .72rem; text-transform: uppercase; letter-spacing: .07em; color: #64748B;
  font-weight: 600; margin-bottom: 6px;}
.sl-cartao-titulo {font-size: 1rem; font-weight: 650; color: #0F172A; margin: 0;}
.sl-cartao-detalhe {font-size: .78rem; color: #64748B; margin: 2px 0 12px 0;}

.sl-prioridade {display: flex; gap: 16px; align-items: flex-start; border-radius: 14px; padding: 16px 20px;
  border: 1px solid; margin-bottom: 16px;}
.sl-prioridade .sl-icone {flex: 0 0 34px; height: 34px; border-radius: 50%; display: flex;
  align-items: center; justify-content: center; font-weight: 800; font-size: 1.05rem; color: #FFFFFF;}
.sl-prioridade h4 {margin: 0; font-size: 1.05rem; font-weight: 700;}
.sl-prioridade p {margin: 3px 0 0 0; font-size: .87rem;}
.sl-alta {background: #FEF3F2; border-color: #FECDCA; color: #7A271A;}
.sl-alta .sl-icone {background: #B42318;}
.sl-rotina {background: #ECFDF3; border-color: #ABEFC6; color: #054F31;}
.sl-rotina .sl-icone {background: #067647;}
.sl-chips {display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;}
.sl-chip {font-size: .78rem; background: rgba(255,255,255,.75); border: 1px solid rgba(15,23,42,.12);
  border-radius: 999px; padding: 3px 11px; color: #1E293B;}
.sl-chip b {font-weight: 650;}
.sl-aviso {font-size: .82rem; background: #FFFAEB; border: 1px solid #FEDF89; color: #7A2E0E;
  border-radius: 10px; padding: 8px 12px; margin-top: 10px;}

.sl-sugestao {display: flex; align-items: baseline; justify-content: space-between; gap: 10px; flex-wrap: wrap;
  border-bottom: 1px solid #F1F5F9; padding-bottom: 12px; margin-bottom: 12px;}
.sl-sugestao .valor {font-size: 1.55rem; font-weight: 700; color: #0F172A; line-height: 1.1;}
.sl-sugestao .conf {font-size: .8rem; color: #0E7490; background: #ECFEFF; border: 1px solid #A5F3FC;
  border-radius: 999px; padding: 3px 10px; white-space: nowrap;}
.sl-prob {margin-bottom: 10px;}
.sl-prob-topo {display: flex; justify-content: space-between; font-size: .86rem; color: #334155;
  margin-bottom: 4px;}
.sl-barra {height: 8px; background: #EEF2F6; border-radius: 4px; overflow: hidden;}
.sl-barra > div {height: 100%; border-radius: 4px;}
.sl-nota {font-size: .75rem; color: #64748B; margin-top: 6px;}

div[class*="st-key-cartao_visor"] img {background: #000000; border-radius: 10px;}
.sl-rodape {font-size: .76rem; color: #64748B; border-top: 1px solid #E2E8F0; padding-top: 12px;
  margin-top: 22px; display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap;}
</style>
"""


# ---------------------------------------------------------------- modelos
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


@st.cache_data
def carregar_metricas(caminho):
    return pd.read_csv(caminho, index_col='modelo') if Path(caminho).exists() else None


def para_tensor(img_cinza, tamanho):
    """Mesma transformação de avaliação usada no treino da CNN (Integrante 3)."""
    tensor = criar_transformacoes(tamanho)[1](Image.fromarray(img_cinza).convert('RGB'))
    rgb = np.clip(tensor.numpy().transpose(1, 2, 0) * DESVIO + MEDIA, 0, 1)
    return tensor, rgb


def sobrepor(rgb, mapa, alpha=0.4):
    mapa = cv2.resize(mapa, (rgb.shape[1], rgb.shape[0]))
    cor = cv2.applyColorMap(np.uint8(mapa * 255), cv2.COLORMAP_JET)[:, :, ::-1] / 255.0
    return np.clip((1 - alpha) * rgb + alpha * cor, 0, 1)


def contorno_pulmonar(img_cinza, mascara):
    """Contorno da máscara pulmonar (segmentacao_pulmonar.py) sobre a imagem, em azul-petróleo claro."""
    cor = cv2.cvtColor(img_cinza, cv2.COLOR_GRAY2RGB)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cor, contornos, -1, (34, 211, 238), 3)
    return cor


# ---------------------------------------------------------------- blocos de HTML
def md(texto):
    st.markdown(texto, unsafe_allow_html=True)


def html_cabecalho():
    return ('<div class="sl-topo"><div class="sl-marca"><div class="sl-logo">SL</div><div>'
            '<p class="sl-titulo">Segunda Leitura</p>'
            '<p class="sl-subtitulo">Apoio à triagem de radiografias de tórax: normal, pneumonia ou tuberculose</p>'
            '</div></div><span class="sl-selo">Protótipo acadêmico · não substitui o laudo médico</span></div>')


def html_resultado(pred, probs, confianca=None):
    """Sugestão do modelo e barras de probabilidade (a classe sugerida em destaque)."""
    partes = ['<div class="sl-sugestao"><div><div class="sl-rotulo">Sugestão</div>',
              f'<div class="valor">{ROTULOS.get(pred, pred)}</div></div>']
    if confianca is not None:
        partes.append(f'<span class="conf">probabilidade {confianca:.0%}</span>')
    partes.append('</div>')
    for c in CLASSES:
        v = (probs or {}).get(c)
        if v is None:
            continue
        destaque = c == pred
        partes.append(f'<div class="sl-prob"><div class="sl-prob-topo"><span style="font-weight:{650 if destaque else 400}">'
                      f'{ROTULOS[c]}</span><span>{v:.1%}</span></div><div class="sl-barra">'
                      f'<div style="width:{v * 100:.1f}%;background:{"#0E7490" if destaque else "#94A3B8"}"></div></div></div>')
    return ''.join(partes)


def html_prioridade(sugestoes):
    """Prioridade de leitura: ALTA se algum modelo sugere alteração; aviso extra quando os modelos discordam."""
    alta = any(s != 'normal' for s in sugestoes.values())
    classe, icone = ('sl-alta', '!') if alta else ('sl-rotina', '✓')
    titulo = 'Prioridade de leitura: ALTA' if alta else 'Prioridade de leitura: rotina'
    texto = ('Ao menos um modelo sugere alteração pulmonar. Recomenda-se antecipar a leitura deste exame.' if alta
             else 'Nenhum modelo sugere alteração pulmonar. O exame segue a ordem normal da fila de laudos.')
    chips = ''.join(f'<span class="sl-chip">{html.escape(nome)}: <b>{ROTULOS.get(s, s)}</b></span>'
                    for nome, s in sugestoes.items())
    aviso = ''
    if len(set(sugestoes.values())) > 1:
        aviso = ('<div class="sl-aviso"><b>Modelos discordam.</b> As abordagens chegaram a sugestões diferentes; '
                 'o caso merece atenção redobrada do radiologista.</div>')
    return (f'<div class="sl-prioridade {classe}"><div class="sl-icone">{icone}</div><div style="flex:1">'
            f'<h4>{titulo}</h4><p>{texto}</p><div class="sl-chips">{chips}</div>{aviso}</div></div>')


def cabecalho_cartao(titulo, detalhe):
    md(f'<p class="sl-cartao-titulo">{html.escape(titulo)}</p><p class="sl-cartao-detalhe">{html.escape(detalhe)}</p>')


def chave_classico(nome):
    nome = nome.lower()
    return 'svm' if nome.startswith('svm') else 'rf' if nome.startswith('random') else 'knn' if nome.startswith('knn') else None


# ---------------------------------------------------------------- página
st.set_page_config(page_title='Segunda Leitura · triagem de raio-X de tórax', page_icon='🩻', layout='wide')
md(ESTILO)
md(html_cabecalho())

with st.sidebar:
    st.markdown('### Exame')
    arquivo = st.file_uploader('Radiografia de tórax frontal (PA ou AP)', type=['png', 'jpg', 'jpeg'],
                               help='Formatos PNG ou JPG. A imagem é processada só nesta sessão.')
    tipo = st.selectbox('Tipo de imagem', TIPOS_IMAGEM, index=0,
                        help='Use "original" para imagens vindas direto do equipamento ou do PACS. '
                             'As versões pré-processadas são as da base do projeto.')
    st.markdown('### Visualização')
    alpha = st.slider('Intensidade do mapa de calor (Grad-CAM)', 0.1, 0.8, 0.4, 0.05)
    with st.expander('Configurações avançadas'):
        caminho_cnn = st.text_input('Modelo CNN (.pt)', str(CNN_PADRAO))
        caminho_cls = st.text_input('Modelo clássico (.joblib)', str(CLASSICO_PADRAO))
    st.caption('Uso exclusivamente acadêmico (P.D.I. – UNIFOR). As sugestões apoiam a priorização da fila de '
               'laudos; a decisão clínica é sempre do profissional de saúde.')

if arquivo is None:
    with st.container(key='cartao_inicio'):
        md('<div class="sl-rotulo">Como usar</div>')
        c1, c2, c3 = st.columns(3)
        c1.markdown('**1. Envie o exame**  \nNa barra lateral, escolha uma radiografia de tórax frontal em PNG ou JPG.')
        c2.markdown('**2. Confira a imagem**  \nVeja a imagem tratada, o mapa de calor da CNN e o contorno pulmonar.')
        c3.markdown('**3. Leia a sugestão**  \nA prioridade de leitura resume os dois modelos. Discordâncias são sinalizadas.')
    st.info('Nenhum exame carregado. Envie uma radiografia pela barra lateral para iniciar a análise.', icon='ℹ️')
    st.stop()

# ---------------------------------------------------------------- pré-processamento
original = np.array(Image.open(arquivo).convert('L'))
if tipo == TIPOS_IMAGEM[0]:
    basico = preprocessar_basico(original)
    processada = realcar(basico)
elif tipo == TIPOS_IMAGEM[1]:
    basico, processada = original, realcar(original)
else:  # já realçada: não dá para voltar à 'basico'; usa a própria imagem nas duas versões
    basico = processada = original
imagens = {'basico': basico, 'realcado': processada}
mascara = segmentar_pulmao(basico)

# ---------------------------------------------------------------- inferência
res_cnn = res_cls = None
erros = []
if Path(caminho_cnn).exists():
    modelo, tamanho, versao_cnn = carregar_cnn(caminho_cnn)
    tensor, rgb = para_tensor(processada, tamanho)
    with torch.no_grad():
        p = torch.softmax(modelo(tensor.unsqueeze(0)), dim=1)[0].numpy()
    idx = int(p.argmax())
    res_cnn = {'pred': CLASSES[idx], 'probs': dict(zip(CLASSES, map(float, p))), 'conf': float(p.max()),
               'gradcam': sobrepor(rgb, gradcam(modelo, tensor, idx, torch.device('cpu')), alpha)}
else:
    erros.append(f'Modelo CNN não encontrado em {caminho_cnn}.')
if Path(caminho_cls).exists():
    pacote = carregar_classico(caminho_cls)
    pred_cls, probs_cls = prever_classico(pacote, imagens)
    res_cls = {'pred': pred_cls, 'probs': probs_cls, 'conf': max(probs_cls.values()) if probs_cls else None,
               'nome': pacote.get('nome', 'Modelo clássico').replace('realcado', 'realçado').replace('basico', 'básico'),
               'descritores': pacote['descritores']}
else:
    erros.append('Modelo clássico não encontrado. Gere com `python classico_3classes.py`.')
for e in erros:
    st.warning(e)

sugestoes = {}
if res_cnn:
    sugestoes['CNN'] = res_cnn['pred']
if res_cls:
    sugestoes['PDI clássico'] = res_cls['pred']
if sugestoes:
    md(html_prioridade(sugestoes))

# ---------------------------------------------------------------- visualizador + resultados
col_img, col_res = st.columns([1.35, 1], gap='medium')

with col_img:
    with st.container(key='cartao_visor'):
        cabecalho_cartao('Visualizador', f'{arquivo.name} · {original.shape[1]}×{original.shape[0]} px')
        abas = st.tabs(['Pré-processada', 'Original', 'Mapa de calor (Grad-CAM)', 'Segmentação pulmonar'])
        with abas[0]:
            st.image(processada, caption='Normalização de intensidade, CLAHE e filtro de mediana 5×5',
                     width='stretch', clamp=True)
        with abas[1]:
            st.image(original, caption='Imagem como foi enviada', width='stretch', clamp=True)
        with abas[2]:
            if res_cnn:
                st.image(res_cnn['gradcam'], caption='Regiões que mais influenciaram a sugestão da CNN '
                         '(vermelho = maior influência)', width='stretch', clamp=True)
            else:
                st.caption('Disponível quando o modelo CNN estiver carregado.')
        with abas[3]:
            area = (mascara > 0).mean()
            st.image(contorno_pulmonar(basico, mascara),
                     caption=f'Contorno pulmonar por limiarização e morfologia matemática · '
                             f'{area:.0%} da imagem', width='stretch')

with col_res:
    if res_cnn:
        with st.container(key='cartao_cnn'):
            cabecalho_cartao('Rede neural convolucional', 'ResNet18 com transferência de aprendizado')
            md(html_resultado(res_cnn['pred'], res_cnn['probs'], res_cnn['conf']))
    if res_cls:
        st.write('')
        with st.container(key='cartao_classico'):
            cabecalho_cartao('Processamento digital de imagens (clássico)',
                             f"{res_cls['nome']} · descritores {', '.join(res_cls['descritores']).upper()}")
            md(html_resultado(res_cls['pred'], res_cls['probs'], res_cls['conf']))
    md('<p class="sl-nota">As porcentagens são estimativas dos modelos, não probabilidades clínicas calibradas.</p>')

# ---------------------------------------------------------------- transparência
st.write('')
with st.expander('Desempenho dos modelos e limitações conhecidas'):
    metricas = carregar_metricas(str(TABELA_METRICAS))
    linhas = {'CNN (ResNet18)': 'cnn_realcado'}
    if res_cls and chave_classico(res_cls['nome']):
        linhas[f"PDI clássico ({res_cls['nome']})"] = chave_classico(res_cls['nome'])
    if metricas is not None and all(k in metricas.index for k in linhas.values()):
        tabela = pd.DataFrame({nome: metricas.loc[k, ['acuracia', 'f1_macro', 'sensibilidade_pneumonia',
                                                       'sensibilidade_tuberculose']] for nome, k in linhas.items()}).T
        tabela.columns = ['Acurácia', 'F1 macro', 'Sens. pneumonia', 'Sens. tuberculose']
        st.markdown('**Resultado no conjunto de teste (160 imagens, nunca usadas no treino)**')
        st.dataframe(tabela.style.format({'Acurácia': '{:.1%}', 'F1 macro': '{:.3f}', 'Sens. pneumonia': '{:.1%}',
                                          'Sens. tuberculose': '{:.1%}'}), width='stretch')
    st.markdown(
        '- **Origem das imagens:** os modelos aprendem também características da base de origem (equipamento, '
        'bordas, marcações). Dentro de uma mesma base, separar pneumonia de normal fica perto do acaso.\n'
        '- **Bases novas:** treinados numa base e testados em outra, os modelos caem para perto do acaso.\n'
        '- **Uso pretendido:** ordenar a fila de leitura em ambiente acadêmico. Não é dispositivo médico e não foi '
        'validado clinicamente.')

md(f'<div class="sl-rodape"><span>Análise gerada em {datetime.now():%d/%m/%Y %H:%M} · {html.escape(arquivo.name)}</span>'
   '<span>Segunda Leitura · P.D.I. – UNIFOR · A decisão clínica é do profissional de saúde.</span></div>')
