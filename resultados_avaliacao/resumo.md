# Resumo da avaliação (gerado por avaliacao_integrante4.py)

Conjunto de teste comum a todos os modelos: **160 imagens**.

## Tabela comparativa

| Modelo | Acurácia | F1 macro | Sens. normal | Sens. pneumonia | Sens. tuberculose | Sens. qualquer doença | Espec. triagem | AUC macro |
|---|---|---|---|---|---|---|---|---|
| CNN | 77.5% | 0.709 | 80.8% | 41.7% | 87.9% | 75.6% | 80.8% | 0.912 |
| KNN | 72.5% | 0.689 | 73.1% | 54.2% | 79.3% | 75.6% | 73.1% | 0.859 |
| RF | 70.6% | 0.678 | 66.7% | 58.3% | 81.0% | 76.8% | 66.7% | 0.879 |
| SVM | 65.6% | 0.648 | 41.0% | 95.8% | 86.2% | 91.5% | 41.0% | 0.877 |

## Intervalos de confiança (bootstrap, 2000 reamostragens)

- **CNN**: acurácia 77.5% (IC95% 71.2%–83.8%); sensibilidade pneumonia 41.7% (IC95% 22.2%–62.2%); sensibilidade tuberculose 87.9% (IC95% 78.8%–95.8%)
- **KNN**: acurácia 72.5% (IC95% 65.6%–79.4%); sensibilidade pneumonia 54.2% (IC95% 31.8%–73.9%); sensibilidade tuberculose 79.3% (IC95% 67.7%–88.9%)
- **RF**: acurácia 70.6% (IC95% 63.7%–77.5%); sensibilidade pneumonia 58.3% (IC95% 37.9%–78.6%); sensibilidade tuberculose 81.0% (IC95% 70.6%–90.3%)
- **SVM**: acurácia 65.6% (IC95% 57.5%–73.1%); sensibilidade pneumonia 95.8% (IC95% 85.7%–100.0%); sensibilidade tuberculose 86.2% (IC95% 77.4%–94.5%)

## Teste de McNemar (pares de modelos, mesmas imagens)

| Modelo A | Modelo B | Só A acerta | Só B acerta | p-valor |
|---|---|---|---|---|
| CNN | KNN | 32 | 24 | 0.3497 |
| CNN | RF | 32 | 21 | 0.1690 |
| CNN | SVM | 39 | 20 | 0.0183 |
| KNN | RF | 14 | 11 | 0.6900 |
| KNN | SVM | 32 | 21 | 0.1690 |
| RF | SVM | 26 | 18 | 0.2912 |

p < 0,05 indica diferença de acertos estatisticamente significativa entre os dois modelos.

## Normal x doença dentro de cada base (acurácia balanceada; 0,5 = acaso)

| modelo   |   Montgomery |   NIH ChestX-ray14 |   Shenzhen |
|:---------|-------------:|-------------------:|-----------:|
| CNN      |        0.775 |              0.513 |      0.906 |
| KNN      |        0.562 |              0.575 |      0.832 |
| RF       |        0.45  |              0.531 |      0.841 |
| SVM      |        0.475 |              0.501 |      0.762 |

## Erros por base de origem

| modelo   | fonte            | classe_real   |   n |   acertos |   predito_normal |   predito_pneumonia |   predito_tuberculose |   taxa_acerto |
|:---------|:-----------------|:--------------|----:|----------:|-----------------:|--------------------:|----------------------:|--------------:|
| CNN      | Montgomery       | normal        |  10 |         8 |                8 |                   0 |                     2 |         0.8   |
| CNN      | Montgomery       | tuberculose   |   8 |         6 |                2 |                   0 |                     6 |         0.75  |
| CNN      | NIH ChestX-ray14 | normal        |  23 |        14 |               14 |                   9 |                     0 |         0.609 |
| CNN      | NIH ChestX-ray14 | pneumonia     |  24 |        10 |               13 |                  10 |                     1 |         0.417 |
| CNN      | Shenzhen         | normal        |  45 |        41 |               41 |                   0 |                     4 |         0.911 |
| CNN      | Shenzhen         | tuberculose   |  50 |        45 |                5 |                   0 |                    45 |         0.9   |
| KNN      | Montgomery       | normal        |  10 |         5 |                5 |                   0 |                     5 |         0.5   |
| KNN      | Montgomery       | tuberculose   |   8 |         5 |                3 |                   0 |                     5 |         0.625 |
| KNN      | NIH ChestX-ray14 | normal        |  23 |        14 |               14 |                   6 |                     3 |         0.609 |
| KNN      | NIH ChestX-ray14 | pneumonia     |  24 |        13 |               10 |                  13 |                     1 |         0.542 |
| KNN      | Shenzhen         | normal        |  45 |        38 |               38 |                   2 |                     5 |         0.844 |
| KNN      | Shenzhen         | tuberculose   |  50 |        41 |                7 |                   2 |                    41 |         0.82  |
| RF       | Montgomery       | normal        |  10 |         4 |                4 |                   0 |                     6 |         0.4   |
| RF       | Montgomery       | tuberculose   |   8 |         4 |                4 |                   0 |                     4 |         0.5   |
| RF       | NIH ChestX-ray14 | normal        |  23 |        11 |               11 |                   9 |                     3 |         0.478 |
| RF       | NIH ChestX-ray14 | pneumonia     |  24 |        14 |                9 |                  14 |                     1 |         0.583 |
| RF       | Shenzhen         | normal        |  45 |        37 |               37 |                   2 |                     6 |         0.822 |
| RF       | Shenzhen         | tuberculose   |  50 |        43 |                6 |                   1 |                    43 |         0.86  |
| SVM      | Montgomery       | normal        |  10 |         2 |                2 |                   0 |                     8 |         0.2   |
| SVM      | Montgomery       | tuberculose   |   8 |         6 |                2 |                   0 |                     6 |         0.75  |
| SVM      | NIH ChestX-ray14 | normal        |  23 |         1 |                1 |                  20 |                     2 |         0.043 |
| SVM      | NIH ChestX-ray14 | pneumonia     |  24 |        23 |                0 |                  23 |                     1 |         0.958 |
| SVM      | Shenzhen         | normal        |  45 |        29 |               29 |                   6 |                    10 |         0.644 |
| SVM      | Shenzhen         | tuberculose   |  50 |        44 |                5 |                   1 |                    44 |         0.88  |