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

**Metodologia — Ablation Study de Features:**
- **M1:** variáveis individuais do ENEM (renda, escolaridade dos pais, tipo de escola, cor/raça, região, ano) — 8 features
- **M2:** M1 + IDHM estadual (educação, renda, longevidade, renda per capita, analfabetismo, etc.) — 17 features
- **M3:** M1 + Questões socioeconômicas estendidas Q_* (computador, internet, carro, ocupação dos pais, bolsa família...) — 22 features
- **M4:** M1 + M2 + M3 + IDHM desagregado por raça/gênero — 31–36 features

**Algoritmos testados:** KNN, LinearSVC+AG, LightGBM, MLP, Ensemble (Soft/Hard Voting)

**Variável alvo:** faixa de nota em Matemática — 3 classes (Baixo < 500 / Médio 500–700 / Alto > 700)

**Métrica principal:** Macro F1 (mais justa para classes desbalanceadas: Alto = ~7% dos candidatos)

**Ferramenta de integração SQL:** DuckDB (requisito da disciplina) — queries Q1–Q4 documentadas no notebook 04.

---

## Dados

### Dataset Principal — Microdados ENEM (2012–2024)

Base de dados constituída pelos microdados do Exame Nacional do Ensino Médio (ENEM), disponibilizados pelo INEP/Governo Federal via Lei de Acesso à Informação (LAI).

- **Período:** 2012–2024 (13 anos)
- **Volume bruto:** ~76,3 milhões de inscrições
- **Volume após filtro de presença:** ~51,3 milhões de candidatos
- **Filtro aplicado:** presença nos quatro dias de prova e nota de Matemática válida

Variáveis base: ano, município/UF da escola, sexo, cor/raça, tipo de escola, escolaridade dos pais (Q001/Q002), renda familiar (Q006), notas por área.

**Variáveis estendidas (2019–2024):** Q003–Q023 — ocupação dos pais, número de pessoas no domicílio, bens domésticos (carro, computador, internet, TV, geladeira, máquina de lavar), empregada doméstica, tipo de escola no EM, Bolsa Família. Disponíveis para ~46% do dataset (questões reestruturadas pelo INEP em 2019).

> Acesso aos microdados: [dados.gov.br — Microdados ENEM](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem)

### Dataset Socioeconômico — Atlas Brasil / PNAD Contínua (2012–2024)

Indicadores socioeconômicos anuais por estado, extraídos do [Atlas Brasil](https://www.atlasbrasil.org.br/) (PNUD/IPEA/FJP) e complementados com dados da PNAD Contínua (IBGE).

Indicadores incluídos:
- **IDHM Geral** e subíndices: Educação, Renda, Longevidade
- **Renda per capita** mensal (R$)
- **Taxa de analfabetismo** — população 15 anos ou mais
- **Taxa de envelhecimento** — proporção da população com 65+
- **Esperança de vida ao nascer** (anos)
- **Mortalidade infantil** (por mil nascidos vivos)
- **Desagregações do IDHM:** por raça (branco/negro) e por gênero (homem/mulher)

> **Limitação conhecida:** os dados do Atlas Brasil disponíveis neste repositório estão agregados no **nível estadual** (27 UFs × 13 anos). O IDHM municipal do PNUD é calculado apenas para anos censitários. O pipeline já está preparado para receber dados municipais quando disponíveis (ver `atlas_municipal.parquet`).

### Tabela de Códigos Municipais — IBGE (2024)

Tabela de lookup com os 5.571 municípios brasileiros, utilizada para bridge entre o código IBGE dos microdados do ENEM e os nomes de municípios do Atlas Brasil.

> Fonte: [IBGE — Divisão Territorial Brasileira 2024](https://www.ibge.gov.br/geociencias/organizacao-do-territorio/estrutura-territorial/23701-divisao-territorial-brasileira.html)

---

## Pipeline de Dados

O processamento dos dados é realizado em cinco notebooks Jupyter localizados em `src/`, executados sequencialmente. O ambiente requer o kernel **Python (enem)** (venv configurado em `.venv/`).

```
src/
├── 01_lookup_municipios.ipynb      # lookup IBGE de municípios
├── 02_limpeza_enem.ipynb           # limpeza + Q_* harmonizadas (2019–2024)
├── 03_limpeza_atlas.ipynb          # limpeza Atlas Brasil
├── 04_join_final.ipynb             # join ENEM × Atlas, queries Q1–Q4
├── 05_eda.ipynb                    # análise exploratória + figuras H1/H2/H3
├── 06_modelagem.ipynb              # KNN M1/M2/M3/M4, ablation study
├── 07_algoritmo_genetico.ipynb     # AG + LinearSVC
├── 08_svm.ipynb                    # LinearSVC grid search
├── 09_ag_svm.ipynb                 # AG + LinearSVC (versão consolidada)
├── 10_svm_ag_cv.ipynb              # LinearSVC+AG + 5-fold CV (1M amostras)
├── 11_ensemble.ipynb               # Ensemble: SVM + LightGBM + KNN
└── 12_mlp_lgb_smote.ipynb          # MLP + LightGBM+AG + SMOTE (M4)
```

### Configuração do ambiente

```bash
# Criar e ativar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# Instalar dependências
pip install -r requirements.txt

# Registrar o kernel no Jupyter
python -m ipykernel install --user --name=enem-venv --display-name="Python (enem)"

# Abrir o JupyterLab
jupyter lab --notebook-dir=.
```

### 01 — Lookup de Municípios (`01_lookup_municipios.ipynb`)

Cria a tabela de bridge entre o código IBGE de 7 dígitos presente nos microdados do ENEM e os nomes de municípios utilizados pelo Atlas Brasil.

- **Input:** `datasets/códigos municipais/RELATORIO_DTB_BRASIL_2024_MUNICIPIOS.ods`
- **Processamento:** normalização de nomes (remoção de acentos, lowercase), mapeamento de código numérico de UF para sigla
- **Output:** `data/processed/lookup/lookup_municipios.parquet` (5.571 municípios, 6 colunas)

### 02 — Limpeza ENEM (`02_limpeza_enem.ipynb`)

Lê os microdados brutos de cada ano via **DuckDB SQL**, aplica filtros, cria variáveis derivadas e exporta um parquet por ano.

- **Input:** `datasets/enem/microdados_enem_AAAA/DADOS/MICRODADOS_ENEM_AAAA.csv`
  - Atenção: 2016 tem nome em minúsculo; 2024 está dividido em `PARTICIPANTES_2024.csv` + `RESULTADOS_2024.csv` (join posicional por ROW_NUMBER)
- **Filtros:** presença nas 4 provas e nota de Matemática não-nula (remove ~32,8% das inscrições)
- **Variáveis derivadas:**
  - `FAIXA_*`: faixas de nota para cada área (`<400`, `400-500`, ..., `>800`)
  - `Q006_HARM`: renda familiar harmonizada entre os questionários de 2012–2014 (A–J) e 2015–2024 (A–P) para escala comparável A–E
  - `REGIAO`: macrorregião a partir da UF da escola
  - `Q_*` (14 colunas): questões socioeconômicas harmonizadas — `Q_OCUP_PAI`, `Q_OCUP_MAE`, `Q_N_PESSOAS`, `Q_EMPREGADA`, `Q_CARRO`, `Q_MOTO`, `Q_GELADEIRA`, `Q_LAVADORA`, `Q_TV`, `Q_COMPUTADOR`, `Q_INTERNET`, `Q_CELULAR`, `Q_TIPO_ESCOLA_EM`, `Q_BOLSA_FAM`
  - **Harmonização 2024:** questões renumeradas em 2024 são remapeadas no SQL (ex: Q008→Q007, Q011→Q010)
  - **Anos 2012–2018:** Q_* definidas como NULL (questionário incompatível com 2019+)
- **Output:** `data/processed/enem/enem_AAAA.parquet` — 13 arquivos

### 03 — Limpeza Atlas Brasil (`03_limpeza_atlas.ipynb`)

Processa os arquivos XLSX do Atlas Brasil, convertendo o formato wide para long e extraindo as desagregações por raça e gênero.

- **Input:** 9 arquivos XLSX em `datasets/atlas brasil/`
- **Processamento:** filtro apenas das linhas estaduais com dados; `pd.melt` para formato long; extração de desagregações por posição de coluna
- **Output:**
  - `data/processed/atlas/atlas_uf.parquet` — 351 linhas (27 UFs × 13 anos), 100% cobertura
  - `data/processed/atlas/atlas_municipal.parquet` — schema pronto, vazio (placeholder para dados municipais futuros)

### 04 — Join Final + DuckDB (`04_join_final.ipynb`)

Integra todos os datasets no banco DuckDB, executa as queries SQL obrigatórias da disciplina e persiste o dataset analítico final.

- **Estratégia de join:** `COALESCE(atlas_municipal, atlas_uf)` — prioriza dado municipal quando disponível, com fallback para estadual
- **Coluna `nivel_atlas`:** indica o nível usado em cada linha (`'municipal'` ou `'estadual'`)
- **Queries SQL obrigatórias:**
  - **Q1:** join principal ENEM × Atlas por município/UF e ano
  - **Q2:** nota média por município e ano
  - **Q3:** distribuição de faixas de nota por macrorregião
  - **Q4:** contagem de candidatos por ano após filtro de presença
- **Output:**
  - `data/enem.duckdb` — banco DuckDB com 5 tabelas: `lookup`, `enem`, `atlas_uf`, `atlas_municipal`, `dataset_analitico`
  - `data/processed/dataset_analitico.parquet` — 777 MB, 51,3 milhões de linhas, 37 colunas

### 06 — Modelagem Preditiva (`06_modelagem.ipynb`)

Treina e avalia modelos KNN para prever a faixa de nota em Matemática e Redação.
Cobre a **Etapa 3 (Resultados Parciais)** do artigo.

**Configuração da primeira execução (resultados parciais):**
- Amostra: **500 mil registros estratificados por classe**, dos 14,2M com IDHM disponível
- Split: 60% treino / 20% validação / 20% teste (teste selado para entrega final)
- Target (6 faixas originais): `<400`, `400-500`, `500-600`, `600-700`, `700-800`, `>800`
- Grid de hiperparâmetros: k ∈ {5, 7, 9, 11, 21} × distância ∈ {euclidean, manhattan}
- Seleção via **Macro F1 no validation set** (métrica mais justa para classes desbalanceadas)
- Baseline: `DummyClassifier(strategy='most_frequent')` — sempre prediz `400-500` (~36%)

**Modelos treinados:**

| Modelo | Features |
|---|---|
| M1 — ENEM only | `TP_SEXO`, `TP_COR_RACA`, `TP_ESCOLA`, `Q001`, `Q002`, `Q006_HARM`, `REGIAO`, `NU_ANO` |
| M2 — ENEM + IDHM | M1 + `idhm`, `idhm_educacao`, `idhm_renda`, `idhm_longevidade`, `renda_percapita`, `tx_analfabetismo`, `tx_envelhecimento`, `esperanca_vida`, `mortalidade_infantil` |

**Resultados — FAIXA_MT (validation set, 500k amostras, 6 faixas originais):**

| Modelo | k ótimo | Distância | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| Baseline (majority) | — | — | 0.3626 | 0.0887 | 0.1929 |
| KNN — ENEM only | 21 | manhattan | 0.3784 | **0.2578** | 0.3487 |
| KNN — ENEM + IDHM | 11 | manhattan | 0.3671 | **0.2586** | 0.3469 |

Δ Macro F1 (M2 − M1) = +0.0008 · Δ Accuracy = −0.0114

**Resultados — FAIXA_REDACAO (validation set, 500k amostras, 6 faixas originais):**

| Modelo | k ótimo | Distância | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| Baseline (majority) | — | — | 0.2565 | 0.0680 | 0.1047 |
| KNN — ENEM only | 21 | euclidean | 0.2922 | **0.2706** | 0.2799 |
| KNN — ENEM + IDHM | 21 | manhattan | 0.2978 | **0.2774** | 0.2859 |

Δ Macro F1 (M2 − M1) = +0.0068 · Δ Accuracy = +0.0056

**Observações sobre os resultados parciais:**
- O IDHM estadual agrega ganho marginal em Macro F1 (+0.0008 em MT, +0.0068 em Redação), evidência preliminar para H3.
- O baixo Macro F1 geral reflete o desbalanceamento severo de 6 classes — a faixa `>800` tem ~1,3% dos candidatos e F1 ≈ 0,05.
- Modelos e test sets selados salvos em `data/models/` para avaliação final com SVM.

**Resultados — FAIXA_MT (3 classes, validation set):**

| Modelo | k | Distância | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| Baseline (majority) | — | — | 0.5255 | 0.2296 | 0.3620 |
| KNN — M1 ENEM only | 21 | manhattan | 0.6141 | **0.4787** | 0.5976 |
| KNN — M2 ENEM+IDHM | 11 | manhattan | 0.6014 | **0.4697** | 0.5863 |

Classe Alto (7% do dataset): recall ≈ 0.13 — severamente sub-prevista sem técnicas de balanceamento.

**Arquivos gerados:**
- `data/models/knn_enem_only_faixa_mt.pkl`, `knn_enem_idhm_faixa_mt.pkl`
- `data/models/test_sets_selados.pkl` — test sets selados para entrega final
- `data/processed/fig_cm_comparativo.png` — matrizes de confusão M1 vs M2

### 07 — Algoritmo Genético + LinearSVC (`07_algoritmo_genetico.ipynb`)

LinearSVC escala para 500k+ amostras em segundos vs KNN que tem custo O(n) por predição.
O AG encontra hiperparâmetros de C e class_weight sem grid search exaustivo.

**Configuração do AG:** pop=25, gens=20, genes: log₁₀(C)∈[−3,2] e class_weight∈{None,balanced}

**Resultados (validation set, M1 e M2):**

| Modelo | C ótimo | balanced | Macro F1 | Accuracy |
|---|---|---|---|---|
| LinearSVC+AG — M1 | 0.0016 | Sim | 0.4660 | 0.5403 |
| LinearSVC+AG — M2 | 43.04 | Sim | 0.4717 | 0.5453 |

**Arquivos gerados:**
- `data/models/ag_svm_enem_only_faixa_mt.pkl`, `ag_svm_enem_idhm_faixa_mt.pkl`
- `data/processed/fig_ga_convergencia.png`, `fig_svm_cm_comparativo.png`, `fig_svm_f1_classe.png`

### 08 — LinearSVC Grid Search (`08_svm.ipynb`)

Grid search explícito sobre C ∈ {0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50} × class_weight ∈ {None, balanced} para comparação direta com o AG do notebook 07.

### 09 — AG + LinearSVC independente (`09_ag_svm.ipynb`)

Versão limpa do AG+SVM com resultados consolidados para M1 e M2.

### 10 — LinearSVC+AG + Validação Cruzada 5-fold (`10_svm_ag_cv.ipynb`)

1M de amostras, 80/20 split, 5-fold StratifiedKFold no treino para avaliar estabilidade.

**Resultados (5-fold CV, Macro F1):**

| Modelo | Média | Desvio Padrão | Accuracy (média) |
|---|---|---|---|
| LinearSVC+AG — M1 | 0.4698 | ±0.0030 | 0.5421 |
| LinearSVC+AG — M2 | 0.4739 | ±0.0023 | 0.5463 |

Baixo desvio padrão confirma estabilidade do modelo.

**Arquivos gerados:**
- `data/models/svm_ag_cv_enem_only.pkl`, `svm_ag_cv_enem_idhm.pkl`
- `data/processed/fig_10_ag_convergencia.png`, `fig_10_cm.png`, `fig_10_cv_boxplot.png`

### 11 — Ensemble: SVM + LightGBM + KNN (`11_ensemble.ipynb`)

Combina 3 modelos distintos com hard voting e soft voting (probabilidades via softmax da decision_function do SVM).

**Resultados (validation set, 1M amostras, M4 features=31):**

| Modelo | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Baseline (majority) | 0.5229 | 0.2289 | 0.3591 |
| KNN (k=11, manhattan, 100k) | 0.5995 | 0.4768 | 0.5857 |
| LinearSVC+AG (C=0.041) | 0.5495 | 0.4738 | 0.5404 |
| LightGBM (500 est.) | 0.5710 | **0.5149** | 0.5780 |
| Ensemble Hard Voting | 0.5866 | 0.5142 | 0.5782 |
| Ensemble Soft Voting | 0.6031 | **0.5334** | 0.5995 |

**Arquivos gerados:**
- `data/models/ens_svm_ag.pkl`, `ens_lgb.pkl`, `ens_knn.pkl`
- `data/processed/fig_11_cm.png` (grade 2×2: SVM/LGB/Hard/Soft), `fig_11_f1_classe.png`

### 12 — MLP + LightGBM+AG + SMOTE (`12_mlp_lgb_smote.ipynb`)

Aborda os dois principais problemas identificados: (1) desbalanceamento via SMOTE e (2) não-linearidades via MLP e LightGBM com AG. Usa features M4 expandidas (36 features incluindo Q_* e IDHM desagregado).

**Hiperparâmetros do LightGBM (AG):** n_est=440, num_leaves=115, lr=0.0289, min_child=79

**Resultados (validation set, ~277k amostras, M4=36 features):**

| Modelo | Accuracy | Macro F1 | Weighted F1 | F1 classe Alto |
|---|---|---|---|---|
| Baseline (majority) | 0.5229 | 0.2289 | 0.3590 | 0.000 |
| MLP 256→128→64 + SMOTE | 0.5816 | 0.5081 | 0.5853 | 0.339 |
| **LightGBM+AG+SMOTE** | **0.6239** | **0.5406** | **0.6188** | **0.369** |

SMOTE foi o principal responsável pelo salto no F1 da classe Alto (de ~0.00 para 0.37).

**Top features (LightGBM):** escolaridade dos pais (q001/q002), renda (q006), computador, internet, carro, ocupação dos pais, tipo de escola.

**Arquivos gerados:**
- `data/models/mlp_smote_m4.pkl`, `lgb_ag_smote_m4.pkl`
- `data/processed/fig_12_cm.png`, `fig_12_f1_classe.png`, `fig_12_feature_importance.png`

---

### Visão Geral dos Resultados (todos os notebooks)

| NB | Modelo | Features | Macro F1 |
|---|---|---|---|
| 06 | KNN (k=21, manhattan) | M1 (8) | 0.4787 |
| 06 | KNN (k=11, manhattan) | M2 (17) | 0.4697 |
| 07 | LinearSVC+AG | M1 | 0.4660 |
| 07 | LinearSVC+AG | M2 | 0.4717 |
| 10 | LinearSVC+AG 5-CV (média) | M2 | 0.4739 |
| 11 | LightGBM (500 est.) | M4 (31) | 0.5149 |
| 11 | Ensemble Soft Voting | M4 (31) | 0.5334 |
| 12 | MLP+SMOTE | M4 (36) | 0.5081 |
| **12** | **LightGBM+AG+SMOTE** | **M4 (36)** | **0.5406** |

Baseline: 0.2289 | **Melhor modelo: LightGBM+AG+SMOTE — Macro F1 = 0.5406** (+0.31 sobre baseline)

---

### 05 — Análise Exploratória (`05_eda.ipynb`)

Gera as visualizações para o artigo, usando DuckDB in-memory sobre o parquet final (sem travar o banco).

- Boxplot de notas por macrorregião (Matemática e Redação) — evidência visual de H1
- Heatmap de correlação de Spearman (renda familiar × nota por área e região) — evidência de H2
- Distribuição de faixas de nota (Matemática, Redação e Média Geral) — verifica desbalanceamento para os modelos ML
- Scatter IDHM estadual × nota média municipal, colorido por macrorregião (Média Geral, Matemática e Redação) — evidência de H3

**Figuras geradas em `data/processed/`:**
- `fig_boxplot_regioes.png`
- `fig_heatmap_spearman.png`
- `fig_faixas_notas.png`
- `fig_scatter_idhm_nota.png`

---

## Estrutura do Repositório

```
Desempenho-Socioeconomico-Enem/
├── src/                         # notebooks de processamento e análise
├── datasets/                    # dados brutos (não versionados)
│   ├── enem/                    # microdados ENEM 2012–2024
│   ├── atlas brasil/            # indicadores socioeconômicos (xlsx)
│   └── códigos municipais/      # lookup IBGE 2024 (ods)
├── data/                        # dados processados (não versionados)
│   ├── enem.duckdb              # banco DuckDB principal (~2,7 GB)
│   └── processed/               # parquets e figuras
├── .venv/                       # ambiente virtual Python
└── requirements.txt             # dependências do projeto
```

## Dependências Principais

| Pacote | Versão | Uso |
|---|---|---|
| duckdb | 1.5.3 | SQL in-process, reservoir sampling, integração dos datasets |
| pandas | 3.0.3 | transformações e limpeza |
| pyarrow | 24.0.0 | persistência em Parquet |
| scikit-learn | 1.9.0 | KNN, LinearSVC, MLP, pipelines, métricas |
| lightgbm | 4.6.0 | gradient boosting (ensemble e notebook 12) |
| imbalanced-learn | 0.13.0 | SMOTE (balanceamento de classes) |
| scipy | 1.17.1 | softmax para soft voting, correlações |
| jupyterlab | 4.5.7 | notebooks interativos |
| matplotlib | 3.10.9 | visualizações e figuras |
| rapidfuzz | 3.14.5 | fuzzy matching de municípios |
| openpyxl | 3.1.5 | leitura dos xlsx do Atlas Brasil |
