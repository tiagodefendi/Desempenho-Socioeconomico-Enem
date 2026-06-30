"""
Classificação ENEM — Dataset Completo | Limiares 500 / 730
==========================================================
Executar a partir da raiz do projeto:
    mkdir -p logs
    nohup python src/17_classificacao_full.py > logs/17_full.log 2>&1 &
    tail -f logs/17_full.log

Otimizações de memória:
  - SELECT apenas as colunas necessárias (não SELECT *)
  - Um único array float32 para M3 (superset); M1 e M2 são slices de colunas
    → peak ~4-5 GB em vez de ~10 GB
  - df_train removido da memória assim que os arrays numpy são extraídos
"""

import time, gc, warnings
warnings.filterwarnings('ignore')

import joblib
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn.linear_model  import LogisticRegression, SGDClassifier
from sklearn.svm           import LinearSVC
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import (RandomForestClassifier,
                                   HistGradientBoostingClassifier,
                                   VotingClassifier)
from sklearn.pipeline      import Pipeline
from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (classification_report, accuracy_score,
                                   f1_score, ConfusionMatrixDisplay)
from sklearn.base          import clone
from pathlib import Path

# ── log com timestamp e flush imediato ───────────────────────────────────────
def log(msg=''):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

# ── caminhos ─────────────────────────────────────────────────────────────────
BASE           = Path('data/processed')
ML_PARQUET     = BASE / 'ml_features.parquet'
TEST_SET       = BASE / 'test_set.parquet'
VAL_SET        = BASE / 'val_set.parquet'
CHECKPOINT_DIR = BASE / 'ckpt_full'
CHECKPOINT_DIR.mkdir(exist_ok=True)
Path('logs').mkdir(exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────
SEED        = 42
CLASS_3     = ['Baixo', 'Medio', 'Alto']
THRESH_LOW  = 500
THRESH_HIGH = 730

# IMPORTANTE: M3 é o superset. M1 = primeiras 28 colunas, M2 = primeiras 37.
# O array numpy de treino é extraído uma única vez (M3) e fatiado para M1/M2.
FEATS_M1 = [
    'REGIAO_NUM', 'TP_SEXO_NUM', 'TP_COR_RACA_NUM', 'TP_ESCOLA_NUM',
    'Q001_NUM', 'Q002_NUM', 'Q006_NUM',
    'Q_N_PESSOAS_NUM', 'Q_OCUP_PAI_NUM', 'Q_OCUP_MAE_NUM',
    'Q_EMPREGADA_NUM', 'Q_BANHEIRO_NUM', 'Q_QUARTO_NUM',
    'Q_CARRO_NUM', 'Q_MOTO_NUM', 'Q_GELADEIRA_NUM',
    'Q_FREEZER_NUM', 'Q_LAVADORA_NUM', 'Q_MICROONDAS_NUM', 'Q_IMPRESSORA_NUM',
    'Q_TV_NUM', 'Q_COMPUTADOR_NUM', 'Q_INTERNET_NUM', 'Q_CELULAR_NUM', 'Q_TEL_FIXO_NUM',
    'Q_TIPO_ESCOLA_EM_NUM', 'Q_BOLSA_FAM_NUM', 'Q_TRABALHA_NUM',
]
FEATS_IDHM = [
    'idhm', 'idhm_educacao', 'idhm_renda', 'idhm_longevidade',
    'renda_percapita', 'tx_analfabetismo', 'tx_envelhecimento',
    'esperanca_vida', 'mortalidade_infantil',
]
FEATS_PERS = ['idhm_cand_raca', 'idhm_cand_sexo']
FEATS_M2   = FEATS_M1 + FEATS_IDHM
FEATS_M3   = FEATS_M2 + FEATS_PERS          # superset — ordem importa
N_M1, N_M2 = len(FEATS_M1), len(FEATS_M2)   # índices de corte para slices

log('=' * 65)
log('Classificação ENEM — Dataset Completo | Limiares 500/730')
log('=' * 65)
log(f'M1={N_M1}  M2={N_M2}  M3={len(FEATS_M3)} features')

# ── 1. Carregamento — apenas colunas necessárias ──────────────────────────────
def nota_to_class(s: pd.Series) -> pd.Series:
    out = pd.Series('Medio', index=s.index, dtype=object)
    out[s < THRESH_LOW]   = 'Baixo'
    out[s >= THRESH_HIGH] = 'Alto'
    return out

COLS_SELECT = ', '.join(['_row_id', 'NU_NOTA_MT'] + FEATS_M3)

log('Carregando val e test (só colunas necessárias)...')
df_val  = pd.read_parquet(VAL_SET,  columns=['_row_id', 'NU_NOTA_MT'] + FEATS_M3)
df_test = pd.read_parquet(TEST_SET, columns=['_row_id', 'NU_NOTA_MT'] + FEATS_M3)
df_val  = df_val.dropna(subset=['NU_NOTA_MT'])
df_test = df_test.dropna(subset=['NU_NOTA_MT'])
df_val['FAIXA_3C']  = nota_to_class(df_val['NU_NOTA_MT'])
df_test['FAIXA_3C'] = nota_to_class(df_test['NU_NOTA_MT'])

excl_ids = pd.concat([df_val[['_row_id']], df_test[['_row_id']]])

log('Carregando treino completo (sem amostragem, só colunas necessárias)...')
t0 = time.time()
con = duckdb.connect()
con.execute("SET memory_limit = '14GB'")
con.execute("SET threads = 8")
con.register('excl_ids', excl_ids)
df_train = con.execute(f"""
    SELECT {COLS_SELECT}
    FROM read_parquet('{ML_PARQUET}')
    WHERE NU_NOTA_MT IS NOT NULL
      AND NU_ANO BETWEEN 2019 AND 2024
      AND _row_id NOT IN (SELECT _row_id FROM excl_ids)
""").df()
con.close()
log(f'Treino: {len(df_train):,} linhas  ({time.time()-t0:.0f}s)')

df_train['FAIXA_3C'] = nota_to_class(df_train['NU_NOTA_MT'])
log(f'Val: {len(df_val):,}  |  Teste: {len(df_test):,} (SELADO)')
log('Distribuição treino:')
for cls, pct in df_train['FAIXA_3C'].value_counts(normalize=True).reindex(CLASS_3).items():
    log(f'  {cls}: {pct:.1%}')

# ── arrays float32 — M3 único, M1/M2 são slices ───────────────────────────────
log('Extraindo arrays float32...')
y_train = df_train['FAIXA_3C'].values
y_val   = df_val['FAIXA_3C'].values
y_test  = df_test['FAIXA_3C'].values

# Um único array contíguo (M3); slices não duplicam memória
X_tr_M3 = df_train[FEATS_M3].astype('float32').values
X_v_M3  = df_val[FEATS_M3].astype('float32').values
X_te_M3 = df_test[FEATS_M3].astype('float32').values

del df_train
gc.collect()
log(f'df_train removido. RAM X_tr_M3: {X_tr_M3.nbytes/1e9:.2f} GB')

# Slices por variante — sem cópia
arrays = {
    'M1': (X_tr_M3[:, :N_M1],  X_v_M3[:, :N_M1],  X_te_M3[:, :N_M1]),
    'M2': (X_tr_M3[:, :N_M2],  X_v_M3[:, :N_M2],  X_te_M3[:, :N_M2]),
    'M3': (X_tr_M3,             X_v_M3,             X_te_M3),
}
for vname, (Xtr, Xv, _) in arrays.items():
    log(f'  {vname}: treino={Xtr.shape}  val={Xv.shape}  NaN={np.isnan(Xtr).mean()*100:.1f}%')

# ── 2. Algoritmo Genético ─────────────────────────────────────────────────────
GA_SUB   = 50_000
rng_sub  = np.random.default_rng(SEED)
idx_sub  = rng_sub.choice(len(X_tr_M3), size=min(GA_SUB, len(X_tr_M3)), replace=False)
idx_subv = rng_sub.choice(len(X_v_M3),  size=min(15_000, len(X_v_M3)),  replace=False)
X_sub,    y_sub    = arrays['M1'][0][idx_sub],  y_train[idx_sub]
X_subval, y_subval = arrays['M1'][1][idx_subv], y_val[idx_subv]
log(f'Subset AG: treino={len(X_sub):,}  val={len(X_subval):,}')

def _tournament(pop, fits, k, rng):
    idx = rng.integers(0, len(pop), k)
    return pop[idx[np.argmax(fits[idx])]].copy()

def _crossover(p1, p2, rng):
    mask = rng.random(len(p1)) < 0.5
    return np.where(mask, p1, p2), np.where(mask, p2, p1)

def _mutate(ind, gene_min, gene_max, binary_mask, rng):
    ind = ind.copy(); ranges = gene_max - gene_min
    for i in range(len(ind)):
        if binary_mask[i]:
            if rng.random() < 0.30: ind[i] = 1.0 - ind[i]
        else:
            if rng.random() < 0.40:
                ind[i] = np.clip(ind[i] + rng.normal(0, 0.15*ranges[i]),
                                 gene_min[i], gene_max[i])
    return ind

def evolve_ag(gene_min, gene_max, fitness_fn, binary_mask,
              pop_size=20, n_gens=15, seed=SEED, label=''):
    rng = np.random.default_rng(seed)
    pop = np.column_stack([rng.uniform(gene_min[i], gene_max[i], pop_size)
                           for i in range(len(gene_min))])
    for i, b in enumerate(binary_mask):
        if b: pop[:, i] = np.round(pop[:, i])
    fits = np.array([fitness_fn(ind) for ind in pop])
    log(f'  [{label}] Gen  0  best={fits.max():.4f}  mean={fits.mean():.4f}')
    hist = [fits.max()]
    for gen in range(1, n_gens + 1):
        elite = list(pop[np.argsort(fits)[-2:]])
        new_pop = elite[:]
        while len(new_pop) < pop_size:
            p1 = _tournament(pop, fits, 3, rng)
            p2 = _tournament(pop, fits, 3, rng)
            c1, c2 = (_crossover(p1, p2, rng) if rng.random() < 0.70
                      else (p1.copy(), p2.copy()))
            if rng.random() < 0.25: c1 = _mutate(c1, gene_min, gene_max, binary_mask, rng)
            if rng.random() < 0.25: c2 = _mutate(c2, gene_min, gene_max, binary_mask, rng)
            new_pop.append(c1)
            if len(new_pop) < pop_size: new_pop.append(c2)
        pop = np.array(new_pop)
        fits = np.array([fitness_fn(ind) for ind in pop])
        hist.append(fits.max())
        log(f'  [{label}] Gen {gen:2d}  best={fits.max():.4f}  mean={fits.mean():.4f}')
    return pop[fits.argmax()], fits.max(), hist

_imp = lambda: SimpleImputer(strategy='median')
_scl = lambda: StandardScaler()

def decode_lr(ind):  return float(10**np.clip(ind[0],-3,2)), bool(round(np.clip(ind[1],0,1)))
def decode_svm(ind): return float(10**np.clip(ind[0],-3,2)), bool(round(np.clip(ind[1],0,1)))
def decode_lgb(ind): return float(np.clip(ind[0],0.01,0.30)), int(round(np.clip(ind[1],15,127)))
def decode_hgb(ind): return float(np.clip(ind[0],0.01,0.30)), int(round(np.clip(ind[1],3,12)))

def fitness_lr(ind):
    C, bal = decode_lr(ind)
    try:
        m = Pipeline([('imp',_imp()),('scl',_scl()),
                      ('clf',LogisticRegression(C=C, class_weight='balanced' if bal else None,
                                                max_iter=300, solver='saga', n_jobs=-1, random_state=SEED))])
        m.fit(X_sub, y_sub)
        return f1_score(y_subval, m.predict(X_subval), average='macro', labels=CLASS_3, zero_division=0)
    except: return 0.0

def fitness_svm(ind):
    C, bal = decode_svm(ind)
    try:
        m = Pipeline([('imp',_imp()),('scl',_scl()),
                      ('clf',LinearSVC(C=C, class_weight='balanced' if bal else None,
                                       max_iter=2000, dual='auto', random_state=SEED))])
        m.fit(X_sub, y_sub)
        return f1_score(y_subval, m.predict(X_subval), average='macro', labels=CLASS_3, zero_division=0)
    except: return 0.0

def fitness_lgb(ind):
    lr, nl = decode_lgb(ind)
    try:
        m = lgb.LGBMClassifier(learning_rate=lr, num_leaves=nl, n_estimators=200,
                                max_depth=8, n_jobs=-1, random_state=SEED, verbose=-1)
        m.fit(X_sub, y_sub)
        return f1_score(y_subval, m.predict(X_subval), average='macro', labels=CLASS_3, zero_division=0)
    except: return 0.0

def fitness_hgb(ind):
    lr, md = decode_hgb(ind)
    try:
        m = HistGradientBoostingClassifier(learning_rate=lr, max_depth=md,
                                           max_iter=200, min_samples_leaf=50, random_state=SEED)
        m.fit(X_sub, y_sub)
        return f1_score(y_subval, m.predict(X_subval), average='macro', labels=CLASS_3, zero_division=0)
    except: return 0.0

log('=== AG — LR ===')
t0 = time.time()
best_lr_ind, _, ag_hist_lr = evolve_ag(np.array([-3.,0.]), np.array([2.,1.]),
    fitness_lr, [False,True], 20, 15, label='LR')
best_C_lr, best_bal_lr = decode_lr(best_lr_ind)
log(f'LR: C={best_C_lr:.4f}  balanced={best_bal_lr}  ({(time.time()-t0)/60:.1f} min)')

log('=== AG — SVM ===')
t0 = time.time()
best_svm_ind, _, ag_hist_svm = evolve_ag(np.array([-3.,0.]), np.array([2.,1.]),
    fitness_svm, [False,True], 20, 15, label='SVM')
best_C_svm, best_bal_svm = decode_svm(best_svm_ind)
log(f'SVM: C={best_C_svm:.4f}  balanced={best_bal_svm}  ({(time.time()-t0)/60:.1f} min)')

log('=== AG — LGB ===')
t0 = time.time()
best_lgb_ind, _, ag_hist_lgb = evolve_ag(np.array([0.01,15.]), np.array([0.30,127.]),
    fitness_lgb, [False,False], 15, 10, label='LGB')
best_lr_lgb, best_nl_lgb = decode_lgb(best_lgb_ind)
log(f'LGB: lr={best_lr_lgb:.4f}  num_leaves={best_nl_lgb}  ({(time.time()-t0)/60:.1f} min)')

log('=== AG — HGB ===')
t0 = time.time()
best_hgb_ind, _, ag_hist_hgb = evolve_ag(np.array([0.01,3.]), np.array([0.30,12.]),
    fitness_hgb, [False,False], 15, 10, label='HGB')
best_lr_hgb, best_md_hgb = decode_hgb(best_hgb_ind)
log(f'HGB: lr={best_lr_hgb:.4f}  max_depth={best_md_hgb}  ({(time.time()-t0)/60:.1f} min)')

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (nome, hist) in zip(axes, [('LR',ag_hist_lr),('SVM',ag_hist_svm),
                                    ('LGB',ag_hist_lgb),('HGB',ag_hist_hgb)]):
    ax.plot(range(len(hist)), hist, 'o-', color='steelblue', markersize=4)
    ax.set_title(f'AG — {nome}'); ax.set_xlabel('Geração')
    ax.set_ylabel('Macro F1'); ax.grid(True, alpha=0.3)
plt.suptitle('Convergência AG — Dataset Completo | 500/730', fontsize=13)
plt.tight_layout()
plt.savefig(BASE / 'ag_convergencia_full.png', dpi=120); plt.close()
log('Salvo: ag_convergencia_full.png')

# ── 3. Definição dos modelos ──────────────────────────────────────────────────
modelos_base = {
    'LogisticRegression': Pipeline([('imp',_imp()),('scl',_scl()),
        ('clf', LogisticRegression(C=best_C_lr,
            class_weight='balanced' if best_bal_lr else None,
            max_iter=500, solver='saga', n_jobs=-1, random_state=SEED))]),
    'DecisionTree': Pipeline([('imp',_imp()),
        ('clf', DecisionTreeClassifier(max_depth=12, min_samples_leaf=100, random_state=SEED))]),
    'RandomForest': Pipeline([('imp',_imp()),
        ('clf', RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=100,
                                       max_samples=0.10, n_jobs=-1, random_state=SEED))]),
    'HistGradientBoosting': HistGradientBoostingClassifier(
        learning_rate=best_lr_hgb, max_depth=best_md_hgb,
        max_iter=200, min_samples_leaf=100, random_state=SEED),
    'SVM (LinearSVC)': Pipeline([('imp',_imp()),('scl',_scl()),
        ('clf', LinearSVC(C=best_C_svm,
            class_weight='balanced' if best_bal_svm else None,
            max_iter=3000, dual='auto', random_state=SEED))]),
    'LightGBM': lgb.LGBMClassifier(learning_rate=best_lr_lgb, num_leaves=best_nl_lgb,
        n_estimators=300, max_depth=8, n_jobs=-1, random_state=SEED, verbose=-1),
}

def make_sgd_svm():
    return Pipeline([('imp',_imp()),('scl',_scl()),
                     ('clf', SGDClassifier(loss='modified_huber',
                         class_weight='balanced' if best_bal_svm else None,
                         alpha=1e-4, max_iter=200, n_jobs=-1, random_state=SEED))])
def make_lgb():
    return lgb.LGBMClassifier(learning_rate=best_lr_lgb, num_leaves=best_nl_lgb,
                               n_estimators=300, max_depth=8, n_jobs=-1,
                               random_state=SEED, verbose=-1)
def make_rf():
    return Pipeline([('imp',_imp()),
                     ('clf', RandomForestClassifier(n_estimators=80, max_depth=14,
                         min_samples_leaf=100, max_samples=0.10,
                         n_jobs=-1, random_state=SEED))])
def make_hgb():
    return HistGradientBoostingClassifier(learning_rate=best_lr_hgb, max_depth=best_md_hgb,
                                          max_iter=200, min_samples_leaf=100, random_state=SEED)

ensembles_fixos = {
    'Ensemble SVM+LGB Hard': VotingClassifier(
        estimators=[('svm', clone(modelos_base['SVM (LinearSVC)'])), ('lgb', make_lgb())],
        voting='hard', n_jobs=1),
    'Ensemble SVM+LGB Soft': VotingClassifier(
        estimators=[('sgd', make_sgd_svm()), ('lgb', make_lgb())],
        voting='soft', n_jobs=1),
    'Ensemble Full Soft': VotingClassifier(
        estimators=[('sgd', make_sgd_svm()), ('lgb', make_lgb()),
                    ('rf', make_rf()), ('hgb', make_hgb())],
        voting='soft', n_jobs=1),
}

# ── 4. Loop de treinamento com checkpoint por variante ────────────────────────
# Checkpoint salva: métricas, predições no val, importâncias de features.
# NÃO salva modelos (muito grandes). Se reiniciar, pula variantes concluídas.
all_results      = []
all_val_preds    = {}   # (nome, vname) → y_pred array (para matrizes de confusão)
all_importances  = {}   # (nome, vname) → feature_importances array
top2_ensemble    = None
top2_names       = None

def _save_checkpoint(vname, results_v, preds_v, imp_v, t2_names):
    ckpt = {'results': results_v, 'preds': preds_v,
            'importances': imp_v, 'top2_names': t2_names}
    joblib.dump(ckpt, CHECKPOINT_DIR / f'ckpt_{vname}.pkl')
    log(f'  Checkpoint salvo: ckpt_{vname}.pkl')

def _load_checkpoints():
    global top2_names, top2_ensemble
    done = set()
    for vname in ['M1', 'M2', 'M3']:
        p = CHECKPOINT_DIR / f'ckpt_{vname}.pkl'
        if not p.exists(): break
        ckpt = joblib.load(p)
        all_results.extend(ckpt['results'])
        for k, v in ckpt['preds'].items():        all_val_preds[(k, vname)]   = v
        for k, v in ckpt['importances'].items():  all_importances[(k, vname)] = v
        if ckpt.get('top2_names'):
            top2_names = ckpt['top2_names']
            def _wrap(name):
                return make_sgd_svm() if 'SVM' in name else clone(modelos_base[name])
            top2_ensemble = VotingClassifier(
                estimators=[(n.replace(' ','_').replace('(','').replace(')',''), _wrap(n))
                            for n in top2_names],
                voting='soft', n_jobs=1)
        done.add(vname)
        log(f'Checkpoint carregado: {vname} ({len(ckpt["results"])} modelos)')
    return done

done_variants = _load_checkpoints()
if done_variants:
    log(f'Variantes já concluídas: {sorted(done_variants)} — pulando.')

for vname, (X_tr, X_v, X_te) in arrays.items():
    if vname in done_variants:
        continue

    log('=' * 65)
    log(f'Variante {vname} | treino: {len(X_tr):,} | {X_tr.shape[1]} features')
    log('=' * 65)

    modelos_v = {n: clone(m) for n, m in modelos_base.items()}
    modelos_v.update({n: clone(m) for n, m in ensembles_fixos.items()})
    if top2_ensemble is not None:
        modelos_v['Ensemble Top-2 Soft'] = clone(top2_ensemble)

    results_v = []; preds_v = {}; imp_v = {}; f1_scores_v = {}

    for nome, modelo in modelos_v.items():
        log(f'  {nome}...')
        t0 = time.time()
        modelo.fit(X_tr, y_train)
        t_tr = time.time() - t0

        y_pred = modelo.predict(X_v)
        acc  = accuracy_score(y_val, y_pred)
        f1_w = f1_score(y_val, y_pred, average='weighted', labels=CLASS_3, zero_division=0)
        f1_m = f1_score(y_val, y_pred, average='macro',    labels=CLASS_3, zero_division=0)
        log(f'    acc={acc:.4f}  F1m={f1_m:.4f}  F1w={f1_w:.4f}  {t_tr:.0f}s')

        row = {'Variante': vname, 'Modelo': nome,
               'Acurácia': acc, 'F1 Weighted': f1_w,
               'F1 Macro': f1_m, 'Tempo (s)': round(t_tr, 1)}
        results_v.append(row)
        all_results.append(row)
        preds_v[nome] = y_pred
        all_val_preds[(nome, vname)] = y_pred
        f1_scores_v[nome] = f1_m

        # Feature importances para modelos de árvore
        clf = modelo.named_steps['clf'] if hasattr(modelo, 'named_steps') else modelo
        if hasattr(clf, 'feature_importances_'):
            imp_v[nome] = clf.feature_importances_
            all_importances[(nome, vname)] = clf.feature_importances_

    if vname == 'M1' and top2_ensemble is None:
        base_f1    = {n: f1_scores_v[n] for n in modelos_base}
        top2_names = sorted(base_f1, key=base_f1.get, reverse=True)[:2]
        log(f'Top-2: {top2_names}')
        def _wrap(name):
            return make_sgd_svm() if 'SVM' in name else clone(modelos_base[name])
        top2_ensemble = VotingClassifier(
            estimators=[(n.replace(' ','_').replace('(','').replace(')',''), _wrap(n))
                        for n in top2_names],
            voting='soft', n_jobs=1)

    _save_checkpoint(vname, results_v, preds_v, imp_v, top2_names)

df_res = pd.DataFrame(all_results)
df_res.to_csv(BASE / 'resultados_full.csv', index=False)
log('Salvo: resultados_full.csv')

pivot = df_res.pivot_table(index='Modelo', columns='Variante', values='F1 Macro', aggfunc='first')
log('\n=== PIVOT F1 Macro (Validação) ===\n' + pivot.round(4).to_string())

# ── 5. Gráficos de resultados ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
pivot_sorted = pivot.loc[pivot.mean(axis=1).sort_values(ascending=True).index]
pivot_sorted.plot(kind='barh', ax=axes[0], width=0.7)
axes[0].set_xlabel('F1 Macro')
axes[0].set_title('F1 Macro — Dataset Completo | 500/730')
axes[0].legend(loc='lower right'); axes[0].set_xlim(0, 1)
if all(v in pivot.columns for v in ['M1','M2','M3']):
    delta = pd.DataFrame({'ΔF1: M2−M1': pivot['M2']-pivot['M1'],
                          'ΔF1: M3−M2': pivot['M3']-pivot['M2']}
                         ).sort_values('ΔF1: M2−M1')
    delta.plot(kind='barh', ax=axes[1], width=0.7, color=['steelblue','darkorange'])
    axes[1].axvline(0, color='k', lw=0.8)
    axes[1].set_xlabel('ΔF1 Macro'); axes[1].set_title('Impacto IDHM')
    axes[1].legend(loc='lower right')
plt.tight_layout()
plt.savefig(BASE / 'classificacao_resultados_full.png', dpi=120, bbox_inches='tight')
plt.close(); log('Salvo: classificacao_resultados_full.png')

# ── 6. Melhor modelo + matrizes de confusão ───────────────────────────────────
# Usa predições salvas nos checkpoints (não precisa dos modelos em memória)
best_row  = df_res.loc[df_res['F1 Macro'].idxmax()]
best_nome = best_row['Modelo']
best_var  = best_row['Variante']
y_pred_best = all_val_preds[(best_nome, best_var)]

log(f'\nMelhor: {best_nome} ({best_var})  F1m={best_row["F1 Macro"]:.4f}')
log('\n' + classification_report(y_val, y_pred_best, labels=CLASS_3, zero_division=0))

# Matriz do melhor
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_val, y_pred_best, labels=CLASS_3,
    normalize='true', cmap='Blues', ax=ax, colorbar=False, values_format='.2f')
ax.set_title(f'{best_nome} ({best_var})\nDataset Completo | 500/730 | Validação')
plt.tight_layout()
plt.savefig(BASE / 'confusion_matrix_full_best.png', dpi=120)
plt.close(); log('Salvo: confusion_matrix_full_best.png')

# Grid todos os modelos
modelos_ord = (df_res[df_res['Variante'] == best_var]
               .sort_values('F1 Macro', ascending=False)['Modelo'].tolist())
n_cols = 5; n_rows = (len(modelos_ord) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
axes = axes.ravel()
for idx, nome in enumerate(modelos_ord):
    ax = axes[idx]; yp = all_val_preds.get((nome, best_var))
    if yp is None: ax.set_visible(False); continue
    f1m = f1_score(y_val, yp, average='macro', labels=CLASS_3, zero_division=0)
    ConfusionMatrixDisplay.from_predictions(y_val, yp, labels=CLASS_3,
        normalize='true', cmap='Blues', ax=ax, colorbar=False, values_format='.2f')
    ax.set_title(f'{nome}\nF1m={f1m:.3f}', fontsize=9)
    ax.tick_params(labelsize=7); ax.set_xlabel('Previsto',fontsize=7); ax.set_ylabel('Real',fontsize=7)
for ax in axes[len(modelos_ord):]: ax.set_visible(False)
plt.suptitle(f'Matrizes — {best_var} | Dataset Completo | 500/730', fontsize=12)
plt.tight_layout()
plt.savefig(BASE / 'confusion_matrices_full_todos.png', dpi=120, bbox_inches='tight')
plt.close(); log('Salvo: confusion_matrices_full_todos.png')

# M1/M2/M3 do melhor
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, vname in zip(axes, ['M1','M2','M3']):
    yp = all_val_preds.get((best_nome, vname))
    if yp is None: ax.set_visible(False); continue
    f1m = f1_score(y_val, yp, average='macro', labels=CLASS_3, zero_division=0)
    acc = accuracy_score(y_val, yp)
    ConfusionMatrixDisplay.from_predictions(y_val, yp, labels=CLASS_3,
        normalize='true', cmap='Blues', ax=ax, colorbar=False, values_format='.2f')
    ax.set_title(f'{best_nome} — {vname}\nF1m={f1m:.3f}  acc={acc:.3f}', fontsize=10)
    ax.tick_params(labelsize=8)
plt.suptitle(f'Impacto IDHM — {best_nome} | Dataset Completo | 500/730', fontsize=12)
plt.tight_layout()
plt.savefig(BASE / 'confusion_matrix_full_m1m2m3.png', dpi=120, bbox_inches='tight')
plt.close(); log('Salvo: confusion_matrix_full_m1m2m3.png')

# ── 7. Importância das features ───────────────────────────────────────────────
feats = FEATS_M3 if best_var == 'M3' else (FEATS_M2 if best_var == 'M2' else FEATS_M1)
importances = {nome: arr for (nome, v), arr in all_importances.items() if v == best_var}

if importances:
    n_plt = min(3, len(importances))
    fig, axes = plt.subplots(1, n_plt, figsize=(6*n_plt, 6))
    if n_plt == 1: axes = [axes]
    for ax, (nome, imp) in zip(axes, list(importances.items())[:n_plt]):
        top = np.argsort(imp)[::-1][:15]
        ax.barh([feats[i] for i in top[::-1]], imp[top[::-1]], color='steelblue')
        ax.set_title(f'{nome} ({best_var})'); ax.set_xlabel('Importância')
        ax.tick_params(axis='y', labelsize=8)
    plt.suptitle(f'Top-15 Features | {best_var} | Dataset Completo', fontsize=13)
    plt.tight_layout()
    plt.savefig(BASE / 'feature_importances_full.png', dpi=120, bbox_inches='tight')
    plt.close(); log('Salvo: feature_importances_full.png')

# ── 8. Acurácia por região ────────────────────────────────────────────────────
REGIAO_MAP = {0:'Norte', 1:'Nordeste', 2:'Centro-Oeste', 3:'Sudeste', 4:'Sul'}
df_ev = df_val[['REGIAO_NUM','FAIXA_3C']].copy()
df_ev['y_pred'] = y_pred_best
df_ev['acerto'] = (df_ev['y_pred'] == df_ev['FAIXA_3C']).astype(int)
acc_reg = (df_ev.dropna(subset=['REGIAO_NUM'])
           .groupby('REGIAO_NUM')['acerto']
           .agg(['mean','count'])
           .rename(columns={'mean':'Acurácia','count':'N'}))
acc_reg.index = acc_reg.index.map(REGIAO_MAP)
log('\nAcurácia por região:\n' + acc_reg.round(4).to_string())

log('\n' + '='*65)
log('CONCLUÍDO.')
log('='*65)
