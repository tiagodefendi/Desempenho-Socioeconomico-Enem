# Desempenho Socioeconômico no ENEM

Trabalho desenvolvido para a disciplina de Tópicos em Engenharia de Software voltada para Ciência de Dados (Mestrado — UTFPR).

---

## Proposta do Artigo

**Título:** Desempenho Socioeconômico no ENEM: Uma Análise Regional de Disparidades e Fatores Explicativos (2012–2024)

**Objetivo:** Investigar como fatores socioeconômicos regionais influenciam o desempenho dos candidatos no ENEM ao longo de 13 anos, identificando padrões de desigualdade entre macrorregiões brasileiras e avaliando o poder explicativo de indicadores individuais e contextuais.

**Hipóteses investigadas:**
- **H1:** Candidatos do Norte e Nordeste apresentam desempenho sistematicamente inferior ao do Sul e Sudeste em todas as áreas do ENEM.
- **H2:** A renda familiar (Q006) tem correlação positiva com o desempenho, mas seu efeito varia por macrorregião.
- **H3:** O IDHM do estado do candidato explica variação adicional no desempenho além da renda individual.

**Variável alvo:** `FAIXA_3C` — 3 classes derivadas de `NU_NOTA_MT`:

| Classe | Faixa | % candidatos |
|--------|-------|-------------|
| Baixo | < 500 | 46,8% |
| Médio | 500–730 | 47,5% |
| Alto | ≥ 730 | 5,7% |

**Métrica principal:** Macro F1 (mais justa para classes desbalanceadas)

**Ferramenta de integração SQL:** DuckDB (requisito da disciplina) — queries Q1–Q4 no notebook 04.

---

## Ablation Study de Features

O projeto compara três conjuntos de features para avaliar o ganho incremental do IDHM:

| Variante | Features | Qtd |
|----------|----------|-----|
| **M1** | Variáveis individuais do candidato: `REGIAO_NUM`, `NU_ANO`, `TP_SEXO_NUM`, `TP_COR_RACA_NUM`, `TP_ESCOLA_NUM`, `Q001–Q006_NUM`, `Q_OCUP_PAI/MAE_NUM`, `Q_N_PESSOAS_NUM`, `Q_EMPREGADA_NUM`, `Q_CARRO/MOTO/GELADEIRA/LAVADORA/TV/COMPUTADOR/INTERNET/CELULAR_NUM`, `Q_TIPO_ESCOLA_EM_NUM`, `Q_BOLSA_FAM_NUM` | 28 |
| **M2** | M1 + IDHM estadual geral: `idhm`, `idhm_educacao`, `idhm_renda`, `idhm_longevidade`, `renda_percapita`, `tx_analfabetismo`, `tx_envelhecimento`, `esperanca_vida`, `mortalidade_infantil` | 37 |
| **M3** | M2 + IDHM personalizado por identidade do candidato: `idhm_cand_raca`, `idhm_cand_sexo` | 39 |

> `idhm_cand_raca`: candidato branco → `idhm_branco`; preto/pardo → `idhm_negro`  
> `idhm_cand_sexo`: sexo M → `idhm_homem`; sexo F → `idhm_mulher`

---

## Dados

### Dataset Principal — Microdados ENEM (2012–2024)

Base de dados constituída pelos microdados do Exame Nacional do Ensino Médio (ENEM), disponibilizados pelo INEP/Governo Federal via Lei de Acesso à Informação (LAI).

- **Período:** 2012–2024 (13 anos)
- **Volume bruto:** ~76,3 milhões de inscrições
- **Volume após filtro de presença:** 51.321.197 candidatos
- **Filtro aplicado:** presença nos quatro dias de prova e nota de Matemática válida

Variáveis base: ano, município/UF da escola, sexo, cor/raça, tipo de escola, escolaridade dos pais (Q001/Q002), renda familiar (Q006), notas por área.

**Variáveis estendidas (2019–2024):** Q003–Q023 — ocupação dos pais, número de pessoas no domicílio, bens domésticos (carro, computador, internet, TV, geladeira, máquina de lavar), empregada doméstica, tipo de escola no EM, Bolsa Família. Disponíveis para ~46% do dataset.

> Acesso aos microdados: [dados.gov.br — Microdados ENEM](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem)

### Dataset Socioeconômico — Atlas Brasil / PNUD (2012–2024)

Indicadores socioeconômicos anuais por estado, extraídos do [Atlas Brasil](https://www.atlasbrasil.org.br/) (PNUD/IPEA/FJP).

Indicadores incluídos: IDHM Geral e subíndices (Educação, Renda, Longevidade), Renda per capita, Taxa de analfabetismo, Taxa de envelhecimento, Esperança de vida, Mortalidade infantil, IDHM desagregado por raça (branco/negro) e gênero (homem/mulher).

> **Limitação:** dados no nível estadual (27 UFs × 13 anos). O pipeline está preparado para receber dados municipais quando disponíveis.

### Tabela de Códigos Municipais — IBGE (2024)

5.571 municípios — bridge entre código IBGE dos microdados e nomes do Atlas Brasil.

---

## Pipeline de Dados

O processamento é realizado em 17 notebooks e 1 script Python em `src/`, executados sequencialmente. Ambiente: kernel **Python (enem)** (venv em `.venv/`).

```
src/
├── 01_lookup_municipios.ipynb         # lookup IBGE de municípios
├── 02_limpeza_enem.ipynb              # limpeza + Q_* harmonizadas (2012–2024)
├── 03_limpeza_atlas.ipynb             # limpeza Atlas Brasil → parquet
├── 04_join_final.ipynb                # join DuckDB + queries Q1–Q4
├── 05_eda.ipynb                       # análise exploratória + figuras H1/H2/H3
│
├── ── Abordagem Inicial ──────────────────────────────────────────────────
├── 06_modelagem.ipynb                 # KNN grid search
├── 07_algoritmo_genetico.ipynb        # AG + LinearSVC
├── 08_svm.ipynb                       # LinearSVC grid search
├── 09_ag_svm.ipynb                    # AG + LinearSVC (versão consolidada)
├── 10_svm_ag_cv.ipynb                 # LinearSVC+AG + 5-fold CV
├── 11_ensemble.ipynb                  # Ensemble: SVM + LightGBM + KNN
├── 12_mlp_lgb_smote.ipynb             # MLP + LightGBM+AG + SMOTE
│
├── ── Abordagem Estendida ────────────────────────────────────────────────
├── 13_preprocessamento_features.ipynb # recodificação numérica completa + NU_ANO
├── 14_preprocessamento_ml.ipynb       # join IDHM + splits fixos test/val
├── 15_classificacao.ipynb             # 10 modelos × M1/M2/M3 (1M amostras)
├── 16_classificacao_5m.ipynb          # 10 modelos × M1/M2/M3 (5M amostras)
├── 17_classificacao_full.ipynb        # dataset completo (~13M treino)
└── 17_classificacao_full.py           # versão script (execução via nohup)
```

### Configuração do ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=enem-venv --display-name="Python (enem)"
jupyter lab --notebook-dir=.
```

Para o notebook 17 (dataset completo), executar como script para evitar estouro de memória do kernel VS Code:

```bash
cd src
nohup python 17_classificacao_full.py > ../logs/17_full.log 2>&1 &
tail -f ../logs/17_full.log
```

---

## Notebooks — Abordagem Inicial (NB 01–12)

### 01 — Lookup de Municípios

Cria bridge código IBGE → nome do município + sigla UF.

- Input: `datasets/códigos municipais/RELATORIO_DTB_BRASIL_2024_MUNICIPIOS.ods`
- Output: `data/processed/lookup/lookup_municipios.parquet` (5.571 municípios)

### 02 — Limpeza ENEM

Lê CSVs brutos via DuckDB, aplica filtros, cria variáveis derivadas e exporta parquet por ano.

- Input: `datasets/enem/microdados_enem_AAAA/DADOS/MICRODADOS_ENEM_AAAA.csv`
  - 2016 tem nome em minúsculo; 2024 dividido em PARTICIPANTES + RESULTADOS (join por ROW_NUMBER)
- Filtros: presença nas 4 provas + NU_NOTA_MT não-nula (remove ~32,8% das inscrições)
- Variáveis derivadas: `FAIXA_*`, `Q006_HARM`, `REGIAO`, `Q_*` (14 colunas harmonizadas)
- Output: `data/processed/enem/enem_AAAA.parquet` (13 arquivos)

### 03 — Limpeza Atlas Brasil

Processa XLSX do Atlas Brasil → formato long, extraindo desagregações por raça e gênero.

- Output: `atlas_uf.parquet` (351 linhas: 27 UFs × 13 anos)

### 04 — Join Final + DuckDB

Integra todos os datasets, executa queries SQL obrigatórias Q1–Q4.

- Estratégia: `COALESCE(atlas_municipal, atlas_uf)` com coluna `nivel_atlas`
- Output: `dataset_analitico.parquet` (777 MB, 51,3M linhas, 37 colunas)

### 05 — EDA

Análise exploratória com DuckDB in-memory sobre `dataset_analitico.parquet`.

**Resultados principais:**

| Região | Nota MT média | Região | Nota MT média |
|--------|--------------|--------|--------------|
| Norte | 473,8 | Sul | 526,7 |
| Nordeste | 491,7 | Sudeste | 532,8 |
| Centro-Oeste | 506,6 | **Brecha N–SE** | **59,0 pts** |

Correlação Spearman renda × nota MT: ρ ≈ 0,45 (varia por região — suporta H2)

**Figuras:** `fig_boxplot_regioes.png`, `fig_heatmap_spearman.png`, `fig_faixas_notas.png`, `fig_scatter_idhm_nota.png`

### 06 — KNN

Grid search k ∈ {5,7,9,11,21} × {euclidean, manhattan}. Amostra 500k. Target: 3 classes (500/700).

| Modelo | k | Distância | Accuracy | Macro F1 |
|--------|---|-----------|----------|----------|
| Baseline | — | — | 0,5255 | 0,2296 |
| KNN M1 | 21 | manhattan | 0,6141 | **0,4787** |
| KNN M2 | 11 | manhattan | 0,6014 | 0,4697 |

**Figuras:** `fig_cm_comparativo.png`, `fig_cm_m1.png`, `fig_cm_m2.png`, `fig_kselect_m1.png`, `fig_kselect_m2.png`

### 07 — Algoritmo Genético + LinearSVC

AG: pop=25, gens=20, torneio k=3, elitismo top-2. Genes: log₁₀(C) ∈ [−3,2] e class_weight ∈ {None, balanced}.

| Modelo | C | balanced | Macro F1 |
|--------|---|----------|----------|
| LinearSVC+AG M1 | 0,0016 | Sim | 0,4660 |
| LinearSVC+AG M2 | 43,04 | Sim | 0,4717 |

**Figuras:** `fig_ga_convergencia.png`, `fig_svm_cm_comparativo.png`, `fig_svm_f1_classe.png`

### 08 — LinearSVC Grid Search

Grid search C ∈ {0.001, …, 50} × class_weight para comparação com AG do NB07.

### 09 — AG + LinearSVC (versão consolidada)

**Figuras:** `fig_ag_convergencia.png`, `fig_ag_svm_cm.png`, `fig_ag_svm_f1_classe.png`

### 10 — LinearSVC+AG + CV 5-fold

1M amostras, 80/20, 5-fold StratifiedKFold.

| Modelo | Macro F1 (média) | Desvio padrão |
|--------|-----------------|---------------|
| LinearSVC+AG M1 | 0,4698 | ±0,0030 |
| LinearSVC+AG M2 | 0,4739 | ±0,0023 |

**Figuras:** `fig_10_ag_convergencia.png`, `fig_10_cm.png`, `fig_10_cv_boxplot.png`, `fig_10_f1_classe.png`

### 11 — Ensemble SVM + LightGBM + KNN

Combina 3 modelos com hard e soft voting (1M amostras, M4=31 features).

| Modelo | Accuracy | Macro F1 |
|--------|----------|----------|
| LightGBM (500 est.) | 0,5710 | 0,5149 |
| Ensemble Hard Voting | 0,5866 | 0,5142 |
| **Ensemble Soft Voting** | **0,6031** | **0,5334** |

**Figuras:** `fig_11_cm.png`, `fig_11_f1_classe.png`, `fig_11_ag_convergencia.png`

### 12 — MLP + LightGBM+AG + SMOTE

SMOTE rebalanceia as classes no treino. M4 = 36 features (~277k amostras).

| Modelo | Macro F1 | F1 classe Alto |
|--------|----------|---------------|
| Baseline | 0,2289 | 0,000 |
| MLP 256→128→64 + SMOTE | 0,5081 | 0,339 |
| **LightGBM+AG+SMOTE** | **0,5406** | **0,369** |

Top features LGB: escolaridade dos pais (Q001/Q002), renda (Q006), computador, internet, carro, ocupação dos pais.

**Figuras:** `fig_12_cm.png`, `fig_12_f1_classe.png`, `fig_12_feature_importance.png`

---

## Notebooks — Abordagem Estendida (NB 13–17)

A abordagem estendida redefine completamente o pipeline de features, usa divisão temporal-estratificada por ano, conjuntos de teste/validação fixos, e testa até o dataset completo (~13M linhas de treino).

### 13 — Preprocessamento Features (`13_preprocessamento_features.ipynb`)

Reescreve a codificação de features do zero. Todas as variáveis categóricas recebem sufixo `_NUM`. Inclui `NU_ANO` como feature (captura tendência temporal).

- Input: CSVs brutos ENEM 2012–2024
- Output: `data/processed/features_socioeconomicas.parquet` (51.321.197 linhas)

### 14 — Preprocessamento ML (`14_preprocessamento_ml.ipynb`)

Join `features_socioeconomicas.parquet × atlas_uf.parquet` via DuckDB. Cria `_row_id`, `idhm_cand_raca` e `idhm_cand_sexo`. Gera conjuntos de teste e validação **fixos e selados** (mesmos para todos os classificadores).

**Divisão estratificada por ano (2019–2024):**
- 10% → `test_set.parquet` (1.654.191 linhas) — **SELADO**
- 10% → `val_set.parquet` (1.488.771 linhas)
- 80% → pool de treino (amostrado conforme o notebook)

- Output: `ml_features.parquet` (51.321.197 linhas, 55 colunas, ZSTD)

### 15 — Classificação 1M amostras (`15_classificacao.ipynb`)

10 modelos × 3 variantes (M1/M2/M3). AG otimiza LR, SVM, LGB e HGB. Limiares 500/700.

**Algoritmos:** LogisticRegression, DecisionTree, RandomForest, HistGradientBoosting, SVM (LinearSVC), LightGBM, Ensemble SVM+LGB Hard, Ensemble SVM+LGB Soft, Ensemble Top-2 Soft, Ensemble Full Soft.

**Melhores resultados (validação, 1M amostras):**

| Modelo | M1 | M2 | M3 |
|--------|----|----|-----|
| **SVM (LinearSVC)+AG** | 0,4872 | 0,4892 | **0,4893** |
| HistGradientBoosting | 0,4691 | 0,4717 | 0,4737 |
| LightGBM | 0,4682 | 0,4728 | 0,4733 |
| Ensemble SVM+LGB Hard | 0,4626 | 0,4665 | 0,4671 |
| DecisionTree | 0,4523 | 0,4493 | 0,4436 |

Melhor: **SVM M3 — Macro F1 = 0,4893**

**Figuras:** `ag_convergencia.png`, `classificacao_resultados.png`, `confusion_matrices_todos.png`, `confusion_matrix_m1m2m3.png`, `feature_importances.png`

### 16 — Classificação 5M amostras (`16_classificacao_5m.ipynb`)

Mesma estrutura do NB15 com `SAMPLE_TRAIN = 5.000.000` e `GA_SUB = 50.000`.

**Figuras:** `ag_convergencia_5m.png`, `classificacao_resultados_5m.png`, `confusion_matrices_5m_todos.png`, `confusion_matrix_5m_best.png`, `confusion_matrix_5m_m1m2m3.png`, `feature_importances_5m.png`

### 17 — Classificação Dataset Completo (`17_classificacao_full.ipynb` / `17_classificacao_full.py`)

Dataset completo: **~13,4M linhas de treino** (anos 2019–2024, excluindo test/val fixos). Limiares revisados: **500 e 730 pontos**.

**Otimizações de memória implementadas:**
- `SELECT` apenas colunas necessárias (não `SELECT *`)
- Arrays float32 — reduz uso de RAM em ~50% (~4–5 GB pico)
- M1 e M2 como slices de colunas de M3 (sem cópia de dados)
- Checkpoints joblib após cada variante (permite reiniciar sem re-treinar)
- `SGDClassifier(loss='modified_huber')` como proxy do SVM em ensembles soft (LinearSVC não emite `predict_proba`)
- `RandomForest(max_samples=0.10)` — limita 1,34M amostras por árvore em 13M linhas

**Resultados completos (validação, limiares 500/730):**

| Modelo | M1 | M2 | M3 | Tempo |
|--------|----|----|-----|-------|
| **Ensemble SVM+LGB Hard** | 0,4847 | **0,4860** | 0,4859 | ~900s |
| SVM (LinearSVC) | 0,4810 | 0,4824 | 0,4824 | ~600s |
| Ensemble SVM+LGB Soft | 0,4521 | 0,4559 | 0,4559 | ~390s |
| LightGBM | 0,4432 | 0,4491 | 0,4487 | ~325s |
| HistGradientBoosting | 0,4424 | 0,4480 | 0,4474 | ~440s |
| Ensemble Full Soft | 0,4379 | 0,4421 | 0,4421 | ~980s |
| DecisionTree | 0,4305 | 0,4327 | 0,4325 | ~150s |
| RandomForest | 0,4260 | 0,4272 | 0,4276 | ~160s |
| LogisticRegression | 0,4177 | 0,4192 | 0,4196 | ~4600s |

Melhor: **Ensemble SVM+LGB Hard (M2) — Macro F1 = 0,4860 | Accuracy = 0,5916**

**Relatório do melhor modelo (val set, 1.488.771 linhas):**

| Classe | Precisão | Recall | F1 | Suporte |
|--------|----------|--------|----|---------|
| Baixo | 0,61 | 0,75 | 0,67 | 696.506 |
| Médio | 0,62 | 0,48 | 0,54 | 707.105 |
| Alto | 0,23 | 0,26 | 0,24 | 85.160 |

**Acurácia por macrorregião (melhor modelo):**

| Região | Acurácia | Candidatos |
|--------|----------|-----------|
| Norte | 0,651 | 162.381 |
| Nordeste | 0,602 | 537.742 |
| Centro-Oeste | 0,579 | 122.537 |
| Sul | 0,576 | 161.739 |
| Sudeste | 0,570 | 504.372 |

> Norte tem menor média de notas mas maior acurácia de classificação — menor variância intraclasse facilita a separação das classes.

**Figuras:** `ag_convergencia_full.png`, `classificacao_resultados_full.png`, `confusion_matrices_full_todos.png`, `confusion_matrix_full_best.png`, `confusion_matrix_full_m1m2m3.png`, `feature_importances_full.png`

---

## Visão Geral dos Resultados (todos os notebooks)

| NB | Modelo | Features | N treino | Limiares | Macro F1 |
|----|--------|----------|----------|----------|----------|
| 06 | KNN (k=21, manhattan) | M1 (8) | 500k | 500/700 | 0,4787 |
| 07 | LinearSVC+AG | M2 (17) | 500k | 500/700 | 0,4717 |
| 10 | LinearSVC+AG 5-CV | M2 (17) | 1M | 500/700 | 0,4739 |
| 11 | Ensemble Soft Voting | M4 (31) | 1M | 500/700 | 0,5334 |
| 12 | LightGBM+AG+SMOTE | M4 (36) | 277k | 500/700 | 0,5406 |
| 15 | SVM+AG (abordagem estendida) | M3 (39) | 1M | 500/700 | 0,4893 |
| 17 | Ensemble SVM+LGB Hard | M2 (37) | ~13M | 500/730 | 0,4860 |

Baseline (majority classifier): 0,2289

> O melhor Macro F1 nominal é do NB12 (LGB+SMOTE = 0,5406), mas foi treinado em subconjunto de 277k com SMOTE e conjunto de teste não-fixo. A avaliação mais rigorosa (conjuntos fixos, sem SMOTE, dataset completo) é o NB17 com Macro F1 ≈ 0,49.

---

## Figuras Geradas (`data/processed/`)

**EDA (NB05):** `fig_boxplot_regioes.png`, `fig_heatmap_spearman.png`, `fig_faixas_notas.png`, `fig_scatter_idhm_nota.png`, `fig_faixas_mt.png`

**KNN (NB06):** `fig_cm_comparativo.png`, `fig_cm_m1.png`, `fig_cm_m2.png`, `fig_kselect_m1.png`, `fig_kselect_m2.png`, `fig_f1_por_classe.png`

**AG+SVM (NB07/09):** `fig_ga_convergencia.png`, `fig_ag_convergencia.png`, `fig_svm_cm.png`, `fig_svm_cm_comparativo.png`, `fig_svm_f1_classe.png`, `fig_ag_svm_cm.png`, `fig_ag_svm_f1_classe.png`

**SVM+CV (NB10):** `fig_10_ag_convergencia.png`, `fig_10_cm.png`, `fig_10_cv_boxplot.png`, `fig_10_f1_classe.png`

**Ensemble (NB11):** `fig_11_cm.png`, `fig_11_f1_classe.png`, `fig_11_ag_convergencia.png`

**MLP+LGB+SMOTE (NB12):** `fig_12_cm.png`, `fig_12_f1_classe.png`, `fig_12_feature_importance.png`

**Classificação 1M (NB15):** `ag_convergencia.png`, `classificacao_resultados.png`, `confusion_matrices_todos.png`, `confusion_matrix_m1m2m3.png`, `feature_importances.png`, `confusion_matrix_val.png`

**Classificação 5M (NB16):** `ag_convergencia_5m.png`, `classificacao_resultados_5m.png`, `confusion_matrices_5m_todos.png`, `confusion_matrix_5m_best.png`, `confusion_matrix_5m_m1m2m3.png`, `feature_importances_5m.png`

**Classificação Full (NB17):** `ag_convergencia_full.png`, `classificacao_resultados_full.png`, `confusion_matrices_full_todos.png`, `confusion_matrix_full_best.png`, `confusion_matrix_full_m1m2m3.png`, `feature_importances_full.png`

---

## Estrutura do Repositório

```
Desempenho-Socioeconomico-Enem/
├── artigo.tex                         # artigo científico (LaTeX, SBC template)
├── REGISTRO_COMPLETO.txt              # documentação técnica completa
├── README.md
├── requirements.txt
│
├── src/
│   ├── 01_lookup_municipios.ipynb
│   ├── 02_limpeza_enem.ipynb
│   ├── 03_limpeza_atlas.ipynb
│   ├── 04_join_final.ipynb
│   ├── 05_eda.ipynb
│   ├── 06_modelagem.ipynb
│   ├── 07_algoritmo_genetico.ipynb
│   ├── 08_svm.ipynb
│   ├── 09_ag_svm.ipynb
│   ├── 10_svm_ag_cv.ipynb
│   ├── 11_ensemble.ipynb
│   ├── 12_mlp_lgb_smote.ipynb
│   ├── 13_preprocessamento_features.ipynb
│   ├── 14_preprocessamento_ml.ipynb
│   ├── 15_classificacao.ipynb
│   ├── 16_classificacao_5m.ipynb
│   ├── 17_classificacao_full.ipynb
│   └── 17_classificacao_full.py       # script Python (nohup para dataset completo)
│
├── datasets/                          # dados brutos (não versionados)
│   ├── enem/                          # microdados ENEM 2012–2024 (csv)
│   ├── atlas brasil/                  # indicadores socioeconômicos (xlsx)
│   └── códigos municipais/            # lookup IBGE 2024 (ods)
│
├── data/                              # dados processados (não versionados)
│   ├── enem.duckdb                    # banco DuckDB (~2,7 GB)
│   └── processed/
│       ├── dataset_analitico.parquet  # 51,3M linhas, 777 MB
│       ├── features_socioeconomicas.parquet  # (NB13)
│       ├── ml_features.parquet        # (NB14) — 51,3M linhas, 55 colunas
│       ├── test_set.parquet           # 1,65M linhas — SELADO
│       ├── val_set.parquet            # 1,49M linhas
│       ├── resultados_full.csv        # resultados NB17 (30 linhas)
│       ├── ckpt_full/                 # checkpoints NB17 (pkl por variante)
│       ├── atlas/
│       ├── enem/                      # parquets por ano (2012–2024)
│       └── lookup/
│
└── logs/
    └── 17_full.log                    # log de execução do script NB17
```

## Dependências Principais

| Pacote | Versão | Uso |
|--------|--------|-----|
| duckdb | 1.5.3 | SQL in-process, reservoir sampling, integração dos datasets |
| pandas | 3.0.3 | transformações e limpeza |
| pyarrow | 24.0.0 | persistência em Parquet |
| scikit-learn | 1.9.0 | KNN, LinearSVC, MLP, pipelines, métricas |
| lightgbm | 4.6.0 | gradient boosting |
| imbalanced-learn | 0.13.0 | SMOTE (balanceamento de classes) |
| joblib | — | checkpoints entre variantes de features |
| scipy | 1.17.1 | softmax para soft voting, correlações |
| jupyterlab | 4.5.7 | notebooks interativos |
| matplotlib | 3.10.9 | visualizações e figuras |
| rapidfuzz | 3.14.5 | fuzzy matching de municípios |
| openpyxl | 3.1.5 | leitura dos xlsx do Atlas Brasil |
