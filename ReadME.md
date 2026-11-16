# Segunda Leitura: classificação auxiliar de raio-X de tórax

Trabalho da disciplina P.D.I. (UNIFOR, Profa. Lyndainês Santos). Protótipo de apoio à triagem que sugere
**normal**, **pneumonia** ou **tuberculose** em radiografias de tórax frontais e mostra as regiões que
pesaram na decisão. Ferramenta acadêmica: **não substitui o laudo do radiologista**.

## Estrutura

| Arquivo | Integrante | O que faz |
|---|---|---|
| `segunda_leitura_processado.zip` (fora do git) | 1 | Base organizada em `processado/<basico|realcado>/<treino|validacao|teste>/<classe>/` |
| `experimento_svm.py`, `experimento_rf.py`, `experimento_knn.py` | 2 | Primeiros testes (base Kaggle, normal x pneumonia) |
| `experimento_TOTAL_dataset_yeshua.py` | 2 | HOG / LBP / GLCM + SVM, KNN e Random Forest nas 3 classes |
| `classico_3classes.py` | 4 (sobre o código do 2) | Mesmos extratores, escolha pela validação, teste nas 160 imagens da CNN, predições no formato comum |
| `cnn_integrante3.py` | 3 | ResNet18 com transferência de aprendizado, treino e Grad-CAM |
| `avaliacao_integrante4.py` | 4 | Métricas, matrizes de confusão, comparação, IC95%, McNemar, erros por base |
| `integracao.py` | 4 | Formato comum de predições, contrato do modelo clássico, pré-processamento |
| `app_prototipo.py` | 4 | Protótipo Streamlit que junta CNN + Grad-CAM + modelo clássico |
| `generalizacao_entre_bases.py` | 4 | Teste em base nova: treina em Shenzhen, testa em Montgomery (tuberculose x normal) |
| `capturas/` | 4 | Capturas de tela do protótipo rodando |

## Como rodar

```bash
pip install -r requirements.txt

# 1) CNN (Integrante 3)
python cnn_integrante3.py --dados segunda_leitura_processado.zip --versao realcado

# 2) Abordagem tradicional nas 3 classes (gera resultados_svm/, resultados_rf/, resultados_knn/ e modelos/)
python classico_3classes.py --dados segunda_leitura_processado.zip

# 3) Avaliação e comparação (Integrante 4): procura todos os resultados_*/predicoes_teste.csv
python avaliacao_integrante4.py
#    ou escolhendo os modelos:
python avaliacao_integrante4.py --modelo CNN=resultados_cnn_realcado/predicoes_teste.csv \
                                --modelo SVM=resultados_svm/predicoes_teste.csv

# 4) Teste em base diferente da do treino (objetivo específico 4)
python generalizacao_entre_bases.py --dados segunda_leitura_processado.zip

# 5) Protótipo
streamlit run app_prototipo.py
```

A avaliação grava em `resultados_avaliacao/`: `tabela_comparativa.csv`, `comparacao_metricas.png`,
`matriz_<modelo>.png`, `metricas_<modelo>.json`, `mcnemar.csv`, `analise_por_fonte.csv`,
`analise_intra_base.csv`, `matrizes_comparacao.png`, `generalizacao_entre_bases.csv` e `resumo.md`.

## Contrato para os classificadores (importante para o Integrante 2)

Para a comparação ser justa, **todos os modelos precisam ser testados nas mesmas imagens**
(`processado/realcado/teste/` do zip, 3 classes). Cada modelo grava:

**1. `resultados_<nome>/predicoes_teste.csv`**

```
arquivo,classe_real,classe_predita,prob_normal,prob_pneumonia,prob_tuberculose
processado/realcado/teste/normal/MC_MCUCXR_0003_0.png,normal,normal,0.91,0.02,0.07
```

As colunas `prob_*` são opcionais (sem elas não sai a AUC). Use `integracao.salvar_predicoes(...)`.

**2. `modelos/modelo_classico.joblib`** (para aparecer no protótipo)

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from integracao import salvar_modelo_classico, extrair_caracteristicas

pipe = make_pipeline(StandardScaler(), SVC(probability=True, class_weight='balanced'))
pipe.fit(X_treino, y_treino)            # y: 0=normal, 1=pneumonia, 2=tuberculose
salvar_modelo_classico('modelos/modelo_classico.joblib', pipe, ['glcm', 'lbp'], 'SVM (GLCM+LBP)')
```

`X_treino` deve ser gerado com `integracao.extrair_caracteristicas(img_cinza, descritores)`, que usa o
mesmo pré-processamento e os mesmos extratores dos scripts `experimento_*.py`.

## Resultados no teste (160 imagens)

| Modelo | Configuração (escolhida pela validação) | Acurácia | F1 macro | Sens. pneumonia | Sens. tuberculose |
|---|---|---|---|---|---|
| CNN ResNet18 | realçado | 77,5% | 0,709 | 41,7% | 87,9% |
| KNN (k=5) | realçado + GLCM | 72,5% | 0,689 | 54,2% | 79,3% |
| Random Forest | realçado + GLCM | 70,6% | 0,678 | 58,3% | 81,0% |
| SVM (RBF) | realçado + LBP+GLCM | 65,6% | 0,648 | 95,8% | 86,2% |

Atenção: dentro do NIH (normal x pneumonia) nenhum modelo passa de 0,58 de acurácia balanceada (0,5 = acaso),
e as características LBP+GLCM acertam a base de origem em 92,5% das imagens. Ver `resultados_avaliacao/resumo.md`.

## Pendências

- [ ] Integrante 2: segmentação pulmonar e operações morfológicas, previstas na proposta
      (há máscaras de referência só para Montgomery, em `mascaras/` do zip).

## Generalização para uma base nova (tuberculose x normal, acurácia balanceada)

| Modelo | Treino e teste em Shenzhen | Treino em Shenzhen, teste em Montgomery |
|---|---|---|
| ResNet18 (ImageNet) + regressão logística | 0,884 | 0,571 |
| Random Forest (GLCM) | 0,840 | 0,584 |
| KNN (GLCM) | 0,833 | 0,482 |
| SVM (LBP+GLCM) | 0,862 | 0,450 |

Numa base que nunca viram, todos caem para perto do acaso (0,50).
