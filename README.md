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

**Metodologia:**
- Três modelos preditivos em ablation study:
  - **M1:** apenas variáveis individuais (renda, escolaridade dos pais, tipo de escola, cor/raça, região)
  - **M2:** M1 + IDHM estadual geral
  - **M3:** M2 + desagregações do IDHM (por raça e gênero)
- Algoritmos: Regressão Logística (baseline) e Random Forest (principal)
- Variável alvo: faixa de nota em Matemática (6 classes: <400 até >800)
- Métrica: F1-macro com validação cruzada estratificada 5-fold
- Análise de outliers positivos: candidatos que performam 2+ faixas acima do previsto pelo M1

**Ferramenta de integração SQL:** DuckDB (requisito da disciplina) — queries Q1–Q4 documentadas no notebook 04.

---

## Dados

### Dataset Principal — Microdados ENEM (2012–2024)

Base de dados constituída pelos microdados do Exame Nacional do Ensino Médio (ENEM), disponibilizados pelo INEP/Governo Federal via Lei de Acesso à Informação (LAI).

- **Período:** 2012–2024 (13 anos)
- **Volume bruto:** ~76,3 milhões de inscrições
- **Volume após filtro de presença:** ~51,3 milhões de candidatos
- **Filtro aplicado:** presença nos quatro dias de prova e nota de Matemática válida

Variáveis utilizadas: ano, município e UF da escola, sexo, cor/raça, tipo de escola, escolaridade dos pais (Q001/Q002), renda familiar (Q006), notas por área (CN, CH, LC, MT, Redação).

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
├── 01_lookup_municipios.ipynb
├── 02_limpeza_enem.ipynb
├── 03_limpeza_atlas.ipynb
├── 04_join_final.ipynb
└── 05_eda.ipynb
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
- **Output:** `data/processed/enem/enem_AAAA.parquet` — 13 arquivos, ~635 MB total

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
| duckdb | 1.5.3 | SQL in-process, integração dos datasets |
| pandas | 3.0.3 | transformações e limpeza |
| pyarrow | 24.0.0 | persistência em Parquet |
| jupyterlab | 4.5.7 | notebooks interativos |
| matplotlib | 3.10.9 | visualizações |
| scipy | 1.17.1 | correlações e estatísticas |
| rapidfuzz | 3.14.5 | fuzzy matching de municípios |
| openpyxl | 3.1.5 | leitura dos xlsx do Atlas Brasil |
