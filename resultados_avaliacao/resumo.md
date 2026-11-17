# Resumo da avaliação (gerado por avaliacao_integrante4.py)

Conjunto de teste comum a todos os modelos: **160 imagens**.

## Tabela comparativa

| Modelo | Acurácia | F1 macro | Sens. normal | Sens. pneumonia | Sens. tuberculose | Sens. qualquer doença | Espec. triagem | AUC macro |
|---|---|---|---|---|---|---|---|---|
| cnn_realcado | 77.5% | 0.709 | 80.8% | 41.7% | 87.9% | 75.6% | 80.8% | 0.912 |
| knn | 72.5% | 0.689 | 73.1% | 54.2% | 79.3% | 75.6% | 73.1% | 0.859 |
| knn_seg | 68.1% | 0.627 | 74.4% | 37.5% | 72.4% | 63.4% | 74.4% | 0.820 |
| rf | 70.6% | 0.678 | 66.7% | 58.3% | 81.0% | 76.8% | 66.7% | 0.879 |
| rf_seg | 66.9% | 0.630 | 66.7% | 50.0% | 74.1% | 70.7% | 66.7% | 0.853 |
| svm | 65.6% | 0.648 | 41.0% | 95.8% | 86.2% | 91.5% | 41.0% | 0.877 |
| svm_seg | 68.1% | 0.671 | 56.4% | 91.7% | 74.1% | 87.8% | 56.4% | 0.862 |

## Intervalos de confiança (bootstrap, 2000 reamostragens)

- **cnn_realcado**: acurácia 77.5% (IC95% 71.2%–83.8%); sensibilidade pneumonia 41.7% (IC95% 22.2%–62.2%); sensibilidade tuberculose 87.9% (IC95% 78.8%–95.8%)
- **knn**: acurácia 72.5% (IC95% 65.6%–79.4%); sensibilidade pneumonia 54.2% (IC95% 31.8%–73.9%); sensibilidade tuberculose 79.3% (IC95% 67.7%–88.9%)
- **knn_seg**: acurácia 68.1% (IC95% 61.3%–75.0%); sensibilidade pneumonia 37.5% (IC95% 17.6%–57.1%); sensibilidade tuberculose 72.4% (IC95% 60.9%–83.6%)
- **rf**: acurácia 70.6% (IC95% 63.7%–77.5%); sensibilidade pneumonia 58.3% (IC95% 37.9%–78.6%); sensibilidade tuberculose 81.0% (IC95% 70.6%–90.3%)
- **rf_seg**: acurácia 66.9% (IC95% 60.0%–73.8%); sensibilidade pneumonia 50.0% (IC95% 28.6%–70.6%); sensibilidade tuberculose 74.1% (IC95% 62.1%–84.5%)
- **svm**: acurácia 65.6% (IC95% 57.5%–73.1%); sensibilidade pneumonia 95.8% (IC95% 85.7%–100.0%); sensibilidade tuberculose 86.2% (IC95% 77.4%–94.5%)
- **svm_seg**: acurácia 68.1% (IC95% 60.6%–75.6%); sensibilidade pneumonia 91.7% (IC95% 78.6%–100.0%); sensibilidade tuberculose 74.1% (IC95% 63.0%–84.5%)

## Teste de McNemar (pares de modelos, mesmas imagens)

| Modelo A | Modelo B | Só A acerta | Só B acerta | p-valor |
|---|---|---|---|---|
| cnn_realcado | knn | 32 | 24 | 0.3497 |
| cnn_realcado | knn_seg | 31 | 16 | 0.0400 |
| cnn_realcado | rf | 32 | 21 | 0.1690 |
| cnn_realcado | rf_seg | 32 | 15 | 0.0186 |
| cnn_realcado | svm | 39 | 20 | 0.0183 |
| cnn_realcado | svm_seg | 32 | 17 | 0.0444 |
| knn | knn_seg | 30 | 23 | 0.4101 |
| knn | rf | 14 | 11 | 0.6900 |
| knn | rf_seg | 29 | 20 | 0.2529 |
| knn | svm | 32 | 21 | 0.1690 |
| knn | svm_seg | 29 | 22 | 0.4011 |
| knn_seg | rf | 24 | 28 | 0.6778 |
| knn_seg | rf_seg | 17 | 15 | 0.8601 |
| knn_seg | svm | 31 | 27 | 0.6940 |
| knn_seg | svm_seg | 19 | 19 | 1.0000 |
| rf | rf_seg | 26 | 20 | 0.4614 |
| rf | svm | 26 | 18 | 0.2912 |
| rf | svm_seg | 25 | 21 | 0.6587 |
| rf_seg | svm | 27 | 25 | 0.8899 |
| rf_seg | svm_seg | 11 | 13 | 0.8388 |
| svm | svm_seg | 15 | 19 | 0.6076 |

p < 0,05 indica diferença de acertos estatisticamente significativa entre os dois modelos.

## Normal x doença dentro de cada base (acurácia balanceada; 0,5 = acaso)

| modelo       |   Montgomery |   NIH ChestX-ray14 |   Shenzhen |
|:-------------|-------------:|-------------------:|-----------:|
| cnn_realcado |        0.775 |              0.513 |      0.906 |
| knn          |        0.562 |              0.575 |      0.832 |
| knn_seg      |        0.487 |              0.514 |      0.801 |
| rf           |        0.45  |              0.531 |      0.841 |
| rf_seg       |        0.438 |              0.489 |      0.8   |
| svm          |        0.475 |              0.501 |      0.762 |
| svm_seg      |        0.263 |              0.545 |      0.82  |

## Erros por base de origem

| modelo       | fonte            | classe_real   |   n |   acertos |   predito_normal |   predito_pneumonia |   predito_tuberculose |   taxa_acerto |
|:-------------|:-----------------|:--------------|----:|----------:|-----------------:|--------------------:|----------------------:|--------------:|
| cnn_realcado | Montgomery       | normal        |  10 |         8 |                8 |                   0 |                     2 |         0.8   |
| cnn_realcado | Montgomery       | tuberculose   |   8 |         6 |                2 |                   0 |                     6 |         0.75  |
| cnn_realcado | NIH ChestX-ray14 | normal        |  23 |        14 |               14 |                   9 |                     0 |         0.609 |
| cnn_realcado | NIH ChestX-ray14 | pneumonia     |  24 |        10 |               13 |                  10 |                     1 |         0.417 |
| cnn_realcado | Shenzhen         | normal        |  45 |        41 |               41 |                   0 |                     4 |         0.911 |
| cnn_realcado | Shenzhen         | tuberculose   |  50 |        45 |                5 |                   0 |                    45 |         0.9   |
| knn          | Montgomery       | normal        |  10 |         5 |                5 |                   0 |                     5 |         0.5   |
| knn          | Montgomery       | tuberculose   |   8 |         5 |                3 |                   0 |                     5 |         0.625 |
| knn          | NIH ChestX-ray14 | normal        |  23 |        14 |               14 |                   6 |                     3 |         0.609 |
| knn          | NIH ChestX-ray14 | pneumonia     |  24 |        13 |               10 |                  13 |                     1 |         0.542 |
| knn          | Shenzhen         | normal        |  45 |        38 |               38 |                   2 |                     5 |         0.844 |
| knn          | Shenzhen         | tuberculose   |  50 |        41 |                7 |                   2 |                    41 |         0.82  |
| knn_seg      | Montgomery       | normal        |  10 |         6 |                6 |                   1 |                     3 |         0.6   |
| knn_seg      | Montgomery       | tuberculose   |   8 |         3 |                5 |                   0 |                     3 |         0.375 |
| knn_seg      | NIH ChestX-ray14 | normal        |  23 |        15 |               15 |                   7 |                     1 |         0.652 |
| knn_seg      | NIH ChestX-ray14 | pneumonia     |  24 |         9 |               14 |                   9 |                     1 |         0.375 |
| knn_seg      | Shenzhen         | normal        |  45 |        37 |               37 |                   0 |                     8 |         0.822 |
| knn_seg      | Shenzhen         | tuberculose   |  50 |        39 |               11 |                   0 |                    39 |         0.78  |
| rf           | Montgomery       | normal        |  10 |         4 |                4 |                   0 |                     6 |         0.4   |
| rf           | Montgomery       | tuberculose   |   8 |         4 |                4 |                   0 |                     4 |         0.5   |
| rf           | NIH ChestX-ray14 | normal        |  23 |        11 |               11 |                   9 |                     3 |         0.478 |
| rf           | NIH ChestX-ray14 | pneumonia     |  24 |        14 |                9 |                  14 |                     1 |         0.583 |
| rf           | Shenzhen         | normal        |  45 |        37 |               37 |                   2 |                     6 |         0.822 |
| rf           | Shenzhen         | tuberculose   |  50 |        43 |                6 |                   1 |                    43 |         0.86  |
| rf_seg       | Montgomery       | normal        |  10 |         5 |                5 |                   2 |                     3 |         0.5   |
| rf_seg       | Montgomery       | tuberculose   |   8 |         3 |                2 |                   3 |                     3 |         0.375 |
| rf_seg       | NIH ChestX-ray14 | normal        |  23 |        11 |               11 |                  12 |                     0 |         0.478 |
| rf_seg       | NIH ChestX-ray14 | pneumonia     |  24 |        12 |               12 |                  12 |                     0 |         0.5   |
| rf_seg       | Shenzhen         | normal        |  45 |        36 |               36 |                   0 |                     9 |         0.8   |
| rf_seg       | Shenzhen         | tuberculose   |  50 |        40 |               10 |                   0 |                    40 |         0.8   |
| svm          | Montgomery       | normal        |  10 |         2 |                2 |                   0 |                     8 |         0.2   |
| svm          | Montgomery       | tuberculose   |   8 |         6 |                2 |                   0 |                     6 |         0.75  |
| svm          | NIH ChestX-ray14 | normal        |  23 |         1 |                1 |                  20 |                     2 |         0.043 |
| svm          | NIH ChestX-ray14 | pneumonia     |  24 |        23 |                0 |                  23 |                     1 |         0.958 |
| svm          | Shenzhen         | normal        |  45 |        29 |               29 |                   6 |                    10 |         0.644 |
| svm          | Shenzhen         | tuberculose   |  50 |        44 |                5 |                   1 |                    44 |         0.88  |
| svm_seg      | Montgomery       | normal        |  10 |         4 |                4 |                   6 |                     0 |         0.4   |
| svm_seg      | Montgomery       | tuberculose   |   8 |         1 |                0 |                   7 |                     1 |         0.125 |
| svm_seg      | NIH ChestX-ray14 | normal        |  23 |         4 |                4 |                  19 |                     0 |         0.174 |
| svm_seg      | NIH ChestX-ray14 | pneumonia     |  24 |        22 |                2 |                  22 |                     0 |         0.917 |
| svm_seg      | Shenzhen         | normal        |  45 |        36 |               36 |                   0 |                     9 |         0.8   |
| svm_seg      | Shenzhen         | tuberculose   |  50 |        42 |                8 |                   0 |                    42 |         0.84  |