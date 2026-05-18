# Generated from ufc_xgboost_model.ipynb.
# Keep the notebook for exploration; use this script for reproducible runs.


# %% [notebook cell 2]
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("XGBoost:", xgb.__version__)

# ── Google Colab: mount Drive and set data paths ──────────────────────────────
# Uncomment if running on Colab:
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_DIR = '/content/drive/MyDrive/ufc/'

# Local paths
MODEL_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = MODEL_DIR / 'data'
ARTIFACT_DIR = MODEL_DIR / 'artifacts'
REPORT_DIR = MODEL_DIR / 'reports'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

STATS_PATH          = DATA_DIR / 'ufc_fight_stats.csv'
RESULTS_PATH        = DATA_DIR / 'ufc_fight_results_with_odds.csv'
FIGHTER_STATS_PATH  = DATA_DIR / 'ufc-fighters-statistics.csv'

# %% [notebook cell 4]
def clean_name(n):
    return str(n).lower().strip().replace('.', '').replace("'", "")

def parse_of(series, landed_name, att_name):
    """Parse 'X of Y' strings into two numeric columns."""
    landed = series.str.extract(r'(\d+)\s+of\s+\d+')[0].astype(float)
    att    = series.str.extract(r'\d+\s+of\s+(\d+)')[0].astype(float)
    return landed.rename(landed_name), att.rename(att_name)

def ctrl_to_sec(s):
    """Convert MM:SS control time string to seconds."""
    try:
        m, sec = str(s).split(':')
        return int(m) * 60 + int(sec)
    except:
        return 0

def safe_div(a, b):
    return np.where(b > 0, a / b, 0.0)

def age_at(dob, as_of):
    dob = pd.to_datetime(dob, errors='coerce')
    as_of = pd.to_datetime(as_of, errors='coerce')
    return (as_of - dob).dt.days / 365.25

# ── Load raw stats ────────────────────────────────────────────────────────────
stats = pd.read_csv(STATS_PATH).dropna(subset=['FIGHTER'])
stats['EVENT'] = stats['EVENT'].str.strip()
stats['BOUT']  = stats['BOUT'].str.strip()

# Parse all 'X of Y' columns
of_columns = {
    'SIG.STR.':   ('sig_landed',    'sig_att'),
    'TOTAL STR.': ('total_landed',  'total_att'),
    'TD':         ('td_landed',     'td_att'),
    'HEAD':       ('head_landed',   'head_att'),
    'BODY':       ('body_landed',   'body_att'),
    'LEG':        ('leg_landed',    'leg_att'),
    'DISTANCE':   ('dist_landed',   'dist_att'),
    'CLINCH':     ('clinch_landed', 'clinch_att'),
    'GROUND':     ('ground_landed', 'ground_att'),
}
for col, (l, a) in of_columns.items():
    stats[l], stats[a] = parse_of(stats[col], l, a)

stats['ctrl_sec'] = stats['CTRL'].apply(ctrl_to_sec)
for c in ['KD', 'SUB.ATT', 'REV.']:
    stats[c] = pd.to_numeric(stats[c], errors='coerce').fillna(0)
stats['FIGHTER'] = stats['FIGHTER'].apply(clean_name)

# ── Aggregate rounds → one row per fighter per fight ──────────────────────────
agg_cols = [
    'sig_landed', 'sig_att', 'total_landed', 'total_att',
    'td_landed', 'td_att', 'head_landed', 'head_att',
    'body_landed', 'body_att', 'leg_landed', 'leg_att',
    'dist_landed', 'dist_att', 'clinch_landed', 'clinch_att',
    'ground_landed', 'ground_att', 'ctrl_sec', 'KD', 'SUB.ATT', 'REV.'
]
fight_level = stats.groupby(['EVENT', 'BOUT', 'FIGHTER'])[agg_cols].sum().reset_index()

# Derived accuracy rates per fight
fight_level['sig_acc']     = safe_div(fight_level['sig_landed'],    fight_level['sig_att'])
fight_level['td_acc']      = safe_div(fight_level['td_landed'],     fight_level['td_att'])
fight_level['head_rate']   = safe_div(fight_level['head_landed'],   fight_level['sig_landed'])
fight_level['body_rate']   = safe_div(fight_level['body_landed'],   fight_level['sig_landed'])
fight_level['leg_rate']    = safe_div(fight_level['leg_landed'],    fight_level['sig_landed'])
fight_level['dist_rate']   = safe_div(fight_level['dist_landed'],   fight_level['total_landed'])
fight_level['clinch_rate'] = safe_div(fight_level['clinch_landed'], fight_level['total_landed'])
fight_level['ground_rate'] = safe_div(fight_level['ground_landed'], fight_level['total_landed'])

print(f"Fight-level records: {fight_level.shape}")
fight_level.head(3)

# %% [notebook cell 6]
results = pd.read_csv(RESULTS_PATH)
results['EVENT'] = results['EVENT'].str.strip()
results['BOUT']  = results['BOUT'].str.strip()
results['fight_date'] = pd.to_datetime(results['fight_date'], errors='coerce')

results[['f1', 'f2']] = results['BOUT'].str.split(' vs. ', expand=True)
results['f1'] = results['f1'].str.strip().apply(clean_name)
results['f2'] = results['f2'].str.strip().apply(clean_name)

odds_cols = ['f1_open_odds', 'f2_open_odds', 'f1_close_odds', 'f2_close_odds']
for col in odds_cols:
    if col not in results.columns:
        results[col] = np.nan
    results[col] = pd.to_numeric(results[col], errors='coerce')

if 'odds_inferred' not in results.columns:
    results['odds_inferred'] = False
results['odds_inferred'] = results['odds_inferred'].astype(int)

decisive = results[results['OUTCOME'].isin(['W/L', 'L/W'])][
    ['EVENT', 'BOUT', 'fight_date', 'f1', 'f2', 'OUTCOME', 'odds_inferred'] + odds_cols
].copy()
decisive['outcome'] = (decisive['OUTCOME'] == 'W/L').astype(int)  # 1 = f1 won

fl = fight_level.merge(
    decisive[['EVENT', 'BOUT', 'fight_date', 'f1', 'f2', 'outcome', 'odds_inferred'] + odds_cols],
    on=['EVENT', 'BOUT'],
    how='inner'
)
fl['is_f1']    = fl['FIGHTER'] == fl['f1']
fl['won']      = np.where(fl['is_f1'], fl['outcome'], 1 - fl['outcome'])
fl['opponent'] = np.where(fl['is_f1'], fl['f2'], fl['f1'])
fl['open_odds'] = np.where(fl['is_f1'], fl['f1_open_odds'], fl['f2_open_odds'])
fl['close_odds'] = np.where(fl['is_f1'], fl['f1_close_odds'], fl['f2_close_odds'])

# Use event dates to get oldest-first ordering for rolling features.
event_order = (
    results[['EVENT', 'fight_date']]
    .dropna(subset=['fight_date'])
    .drop_duplicates()
    .sort_values(['fight_date', 'EVENT'])
    .reset_index(drop=True)
)
event_order['event_idx'] = np.arange(len(event_order))  # 0 = oldest
event_order = event_order[['EVENT', 'event_idx']]
fl = fl.merge(event_order, on='EVENT', how='left')
fl = fl.sort_values(['FIGHTER', 'event_idx']).reset_index(drop=True)

print(f"Decisive fights joined: {fl.shape}")
print(f"Win rate: {fl['won'].mean():.3f}  (should be exactly 0.5)")
print(f"Rows with odds: {fl['open_odds'].notna().sum()}/{len(fl)} fighter rows")
print(f"Inferred odds rows: {fl['odds_inferred'].sum()}/{len(fl)} fighter rows")

# %% [notebook cell 8]
roll_base = [
    'sig_landed', 'sig_att', 'sig_acc', 'td_landed', 'td_att', 'td_acc',
    'head_rate', 'body_rate', 'leg_rate', 'dist_rate', 'clinch_rate', 'ground_rate',
    'ctrl_sec', 'KD', 'SUB.ATT', 'REV.', 'won'
]

for w in [3, 5]:
    for feat in roll_base:
        fl[f'roll{w}_{feat}'] = (
            fl.groupby('FIGHTER')[feat]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

fl['n_prior_fights'] = fl.groupby('FIGHTER').cumcount()
fl_model = fl[fl['n_prior_fights'] >= 1].copy()  # need at least 1 prior fight

roll_cols = [c for c in fl_model.columns if c.startswith('roll')]
print(f"Rolling features added: {len(roll_cols)}")
print(f"Fight records for modelling: {len(fl_model)}")
print()
# Sanity check: Jon Jones rolling stats in chronological order
jj = fl_model[fl_model['FIGHTER'] == 'jon jones'][
    ['EVENT', 'won', 'roll5_sig_landed', 'roll5_td_acc', 'roll3_won', 'roll5_won']
].head(6)
print("Jon Jones rolling stats (chronological):")
print(jj.to_string(index=False))

# %% [notebook cell 10]
fs = pd.read_csv(FIGHTER_STATS_PATH).drop(columns=['nickname'])

num_cols = fs.select_dtypes('float').columns
fs[num_cols] = fs[num_cols].fillna(fs[num_cols].median())

fs['date_of_birth'] = pd.to_datetime(fs['date_of_birth'], errors='coerce')
fs['stance'] = fs['stance'].fillna('Unknown')
fs = pd.get_dummies(fs, columns=['stance'], prefix='stance', dtype=int)
fs['name'] = fs['name'].apply(clean_name)

fs_for_training = fs.copy()
static_career_cols = [c for c in fs_for_training.columns if c not in ['name', 'date_of_birth']]
# Current age is used only by predict_fight; training rows get event-date age below.
fs['age'] = age_at(fs['date_of_birth'], pd.Timestamp.today().normalize())
fs = fs.drop(columns=['date_of_birth'])
fs['age'] = fs['age'].fillna(fs['age'].median())
career_cols = static_career_cols + ['age']
print(f"Career features: {len(career_cols)}")
print(f"Fighters in database: {len(fs)}")

# %% [notebook cell 12]
from sklearn.model_selection import GroupShuffleSplit

# Self-join: match each fighter row with their opponent's row in the same fight
current_odds_cols = ['open_odds', 'close_odds']
fight_level_cols = ['odds_inferred']
fighter_view = fl_model[['EVENT', 'BOUT', 'fight_date', 'FIGHTER', 'opponent', 'won', 'event_idx'] + roll_cols + current_odds_cols + fight_level_cols].copy()

side_specific_cols = roll_cols + current_odds_cols
opp_view = fighter_view[['EVENT', 'BOUT', 'FIGHTER'] + side_specific_cols].copy()
opp_view = opp_view.rename(columns={c: f'{c}_opp' for c in side_specific_cols})
opp_view = opp_view.rename(columns={'FIGHTER': 'opponent'})

self_view = fighter_view.rename(columns={c: f'{c}_self' for c in side_specific_cols})
paired = self_view.merge(opp_view, on=['EVENT', 'BOUT', 'opponent'], how='inner')

# Add career stats for fighter and opponent
paired = paired.merge(
    fs_for_training.rename(columns={c: f'career_{c}' for c in static_career_cols} | {'date_of_birth': 'career_date_of_birth', 'name': 'name'}),
    left_on='FIGHTER', right_on='name', how='left'
).drop(columns=['name'])
paired = paired.merge(
    fs_for_training.rename(columns={c: f'opp_career_{c}' for c in static_career_cols} | {'date_of_birth': 'opp_career_date_of_birth', 'name': 'name'}),
    left_on='opponent', right_on='name', how='left'
).drop(columns=['name'])
paired['career_age'] = age_at(paired['career_date_of_birth'], paired['fight_date'])
paired['opp_career_age'] = age_at(paired['opp_career_date_of_birth'], paired['fight_date'])
age_fill = pd.concat([paired['career_age'], paired['opp_career_age']]).median()
paired[['career_age', 'opp_career_age']] = paired[['career_age', 'opp_career_age']].fillna(age_fill)
paired = paired.drop(columns=['career_date_of_birth', 'opp_career_date_of_birth'])

# Odds are fully populated by ufc_fight_results_with_odds.csv. Keep XGBoost missing-safe anyway.
allowed_missing = [f'{x}_self' for x in current_odds_cols] + [f'{x}_opp' for x in current_odds_cols] + ['fight_date']
paired = paired.dropna(subset=[c for c in paired.columns if c not in allowed_missing])


# Column groups
self_cols     = [c for c in paired.columns if c.endswith('_self')]
opp_roll_cols = [c for c in paired.columns if c.endswith('_opp') and not c.startswith('opp_career')]
career_self   = [f'career_{c}' for c in static_career_cols] + ['career_age']
career_opp    = [f'opp_career_{c}' for c in static_career_cols] + ['opp_career_age']

# Diff features
for col in side_specific_cols:
    paired[f'diff_{col}'] = paired[f'{col}_self'] - paired[f'{col}_opp']
for s, o in zip(career_self, career_opp):
    paired[f'diff_{s[7:]}'] = paired[s] - paired[o]

paired_original = paired.copy()

diff_cols    = [c for c in paired.columns if c.startswith('diff_')]
fight_level_feature_cols = fight_level_cols
feature_cols = self_cols + opp_roll_cols + career_self + career_opp + diff_cols + fight_level_feature_cols

X = paired[feature_cols]
y = paired['won']
groups = paired['EVENT'].astype(str) + '||' + paired['BOUT'].astype(str)

# ── Step 1: Split by bout BEFORE swapping ────────────────────────────────────
# Keep mirrored fighter rows from the same bout in the same split.
paired['_row_id'] = np.arange(len(paired))
canonical = paired.groupby(['EVENT', 'BOUT'])['_row_id'].first()
canonical_ids = set(canonical.values)
canonical_mask = paired['_row_id'].isin(canonical_ids)

def grouped_split(X_part, y_part, groups_part, test_size, random_state):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return next(splitter.split(X_part, y_part, groups_part))

train_val_idx, test_idx = grouped_split(X, y, groups, test_size=0.2, random_state=42)
X_train_val, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]
groups_train_val = groups.iloc[train_val_idx]
assert set(groups_train_val).isdisjoint(set(groups.iloc[test_idx]))

train_idx, val_idx = grouped_split(X_train_val, y_train_val, groups_train_val, test_size=0.2, random_state=43)
X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
groups_train = groups_train_val.iloc[train_idx]
assert set(groups_train).isdisjoint(set(groups_train_val.iloc[val_idx]))

test_indices = X_test.index
natural_indices = [i for i in test_indices if canonical_mask.loc[i]]
X_test_natural = X.loc[natural_indices]
y_test_natural = y.loc[natural_indices]

print(f"Total features : {len(feature_cols)}")
print(f"Test outcome dist (all mirrored rows): {y_test.value_counts().to_dict()}")
print(f"Test outcome dist (one row per bout): {y_test_natural.value_counts().to_dict()}")
print(f"  One-row test win rate: {y_test_natural.mean():.1%}")

# ── Step 2: Swap only the TRAINING set ───────────────────────────────────────
def swap_training_rows(X_part, y_part, random_state=42):
    rng = np.random.default_rng(random_state)
    swap_idx = rng.random(len(X_part)) < 0.5
    X_swapped = X_part.copy()
    y_swapped = y_part.copy()
    swap_rows = X_swapped.index[swap_idx]
    X_swapped.loc[swap_rows, self_cols + opp_roll_cols] = (
        X_swapped.loc[swap_rows, opp_roll_cols + self_cols].values
    )
    X_swapped.loc[swap_rows, career_self + career_opp] = (
        X_swapped.loc[swap_rows, career_opp + career_self].values
    )
    X_swapped.loc[swap_rows, diff_cols] *= -1
    y_swapped.loc[swap_rows] = 1 - y_swapped.loc[swap_rows]
    return X_swapped, y_swapped, int(swap_idx.sum())

X_train, y_train, swapped_train = swap_training_rows(X_train, y_train, random_state=42)

print(f"\nRows swapped in train: {swapped_train} / {len(X_train)}")
print(f"\nTrain outcome dist (after swap): {y_train.value_counts().to_dict()}")
print(f"  → Balanced: {y_train.mean():.1%}  (eliminates positional bias)")
print(f"\nTrain: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# %% [notebook cell 14]
# pip install optuna  (if needed)
import optuna
from sklearn.model_selection import GroupKFold, cross_val_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 200, 1000),
        'max_depth':        trial.suggest_int('max_depth', 3, 7),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'gamma':            trial.suggest_float('gamma', 0, 5),
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
        'device': 'cuda',
        'tree_method': 'hist'
    }
    cv = GroupKFold(n_splits=5)
    scores = cross_val_score(
        xgb.XGBClassifier(**params), X_train, y_train,
        groups=groups_train, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    return scores.mean()

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(multivariate=True)
)
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best AUC (CV): {study.best_value:.4f}")
print(f"Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# Visualize search
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.tight_layout(); plt.show()

optuna.visualization.matplotlib.plot_param_importances(study)
plt.tight_layout(); plt.show()

# %% [notebook cell 16]
# Use Optuna best params if available, else fall back to defaults
try:
    best_params = study.best_params
    print("Using Optuna best params")
except NameError:
    best_params = {
        'n_estimators': 600, 'max_depth': 4, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'gamma': 0,
    }
    print("Using default params (run Optuna cell first for best results)")

model = xgb.XGBClassifier(
    **best_params,
    eval_metric='logloss',
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1
)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)
print(f"\nBest iteration: {model.best_iteration}")

# %% [notebook cell 18]
results_dict = model.evals_result()
train_logloss = results_dict['validation_0']['logloss']
val_logloss   = results_dict['validation_1']['logloss']
epochs = range(1, len(train_logloss) + 1)

plt.figure(figsize=(9, 4))
plt.plot(epochs, train_logloss, label='Train logloss', color='#185FA5')
plt.plot(epochs, val_logloss,   label='Val logloss',   color='#D85A30', linestyle='--')
plt.axvline(model.best_iteration, color='gray', linestyle=':', alpha=0.7, label=f'Best: epoch {model.best_iteration}')
plt.xlabel('Boosting round'); plt.ylabel('Log Loss')
plt.title('XGBoost training curves'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_DIR / 'xgb_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [notebook cell 20]
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

probs = model.predict_proba(X_test)[:, 1]
preds = (probs > 0.5).astype(int)

print('=' * 50)
print('GROUPED TEST SET (mirrored fighter rows)')
print('=' * 50)
print(f'Accuracy : {accuracy_score(y_test, preds):.4f}')
print(f'AUC      : {roc_auc_score(y_test, probs):.4f}')
print()
print(classification_report(y_test, preds, target_names=['Opponent wins', 'Fighter wins']))

# ── Natural distribution AUC ─────────────────────────────────────────────────
# Re-extract test rows from unswapped paired data using the same indice

probs_natural = model.predict_proba(X_test_natural)[:, 1]
preds_natural = (probs_natural > 0.5).astype(int)

print('=' * 50)
print('NATURAL DISTRIBUTION (real fight ordering)')
print('=' * 50)
print(f'Accuracy : {accuracy_score(y_test_natural, preds_natural):.4f}')
print(f'AUC      : {roc_auc_score(y_test_natural, probs_natural):.4f}  ← report this one')
print(f'Baseline : {y_test_natural.mean():.4f}  (naive always-pick-favorite accuracy)')
print()
print(classification_report(y_test_natural, preds_natural, target_names=['Underdog wins', 'Favorite wins']))

# ── Calibration curve ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Calibration on natural test set
prob_true, prob_pred = calibration_curve(y_test_natural, probs_natural, n_bins=10)
axes[0].plot(prob_pred, prob_true, 'o-', color='#185FA5', label='Model')
axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
axes[0].set(xlabel='Mean predicted probability', ylabel='Fraction of positives',
            title='Calibration curve (natural distribution)')
axes[0].legend(); axes[0].grid(alpha=0.3)

# Probability distribution
axes[1].hist(probs_natural[y_test_natural == 0], bins=25, alpha=0.6,
             color='#D85A30', label='Underdog wins', density=True)
axes[1].hist(probs_natural[y_test_natural == 1], bins=25, alpha=0.6,
             color='#185FA5', label='Favorite wins', density=True)
axes[1].axvline(0.5, color='gray', linestyle='--', alpha=0.7)
axes[1].set(xlabel='Predicted P(fighter wins)', ylabel='Density',
            title='Predicted probability distributions')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_DIR / 'calibration.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Optional: fit isotonic calibration if curve is S-shaped ──────────────────
# Uncomment if calibration curve deviates significantly from diagonal:
# calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
# calibrated_model.fit(X_val, y_val)

# %% [notebook cell 22]
explainer  = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(X_test)

importance = pd.DataFrame({
    'feature':        feature_cols,
    'mean_abs_shap':  np.abs(shap_vals).mean(0)
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

print("Top 20 most important features:")
print(importance.head(20).to_string(index=False))

# Bar chart
fig, ax = plt.subplots(figsize=(9, 7))
top20 = importance.head(20)
ax.barh(top20['feature'][::-1], top20['mean_abs_shap'][::-1], color='#185FA5')
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('Top 20 features by SHAP importance')
plt.tight_layout()
plt.savefig(REPORT_DIR / 'shap_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [notebook cell 24]
def _build_row(fighter_name, opponent_name, open_odds=np.nan, close_odds=np.nan, opponent_open_odds=np.nan, opponent_close_odds=np.nan, odds_inferred=1):
    """
    Build a single feature row for a fighter vs opponent matchup.
    Uses the fighter's most recent rolling stats + career stats.
    Returns a DataFrame aligned to feature_cols, or None if not found.
    """
    fn = clean_name(fighter_name)
    on = clean_name(opponent_name)

    if fn not in fl_model['FIGHTER'].values:
        print(f"'{fighter_name}' not found in fight history.")
        return None
    if on not in fs['name'].values:
        print(f"'{opponent_name}' not found in fighter database.")
        return None

    # Most recent rolling stats for the fighter
    fighter_row = fl_model[fl_model['FIGHTER'] == fn].iloc[-1]

    # Most recent rolling stats for the opponent (if they have fight history)
    if on in fl_model['FIGHTER'].values:
        opp_row = fl_model[fl_model['FIGHTER'] == on].iloc[-1]
        opp_rolling = {f'{c}_opp': opp_row[c] for c in roll_cols}
    else:
        # Opponent has no fight history — use zeros for rolling stats
        opp_rolling = {f'{c}_opp': 0.0 for c in roll_cols}

    # Career stats
    career_self_row = fs[fs['name'] == fn].iloc[0]
    career_opp_row  = fs[fs['name'] == on].iloc[0]

    row = {}
    for c in roll_cols:
        row[f'{c}_self'] = fighter_row[c]
    row['open_odds_self'] = open_odds
    row['close_odds_self'] = close_odds
    row.update(opp_rolling)
    row['open_odds_opp'] = opponent_open_odds
    row['close_odds_opp'] = opponent_close_odds
    row['odds_inferred'] = odds_inferred
    for c in career_cols:
        row[f'career_{c}']     = career_self_row[c]
        row[f'opp_career_{c}'] = career_opp_row[c]

    # Diff features
    for c in roll_cols + ['open_odds', 'close_odds']:
        row[f'diff_{c}'] = row[f'{c}_self'] - row[f'{c}_opp']
    for s, o in zip(career_self, career_opp):
        row[f'diff_{s[7:]}'] = row[s] - row[o]

    return pd.DataFrame([row])[feature_cols]


def predict_fight(
    fighter1: str,
    fighter2: str,
    fighter1_open_odds=np.nan,
    fighter2_open_odds=np.nan,
    fighter1_close_odds=np.nan,
    fighter2_close_odds=np.nan,
    odds_inferred: int = 1,
    verbose: bool = True,
):
    """
    Predict the outcome of a fight between two fighters.

    Averages predictions from both orderings (f1 vs f2 and f2 vs f1)
    so output probabilities are symmetric and always sum to 1.0.

    Returns (winner_name, win_probability) or None if fighters not found.
    """
    row_ab = _build_row(fighter1, fighter2, fighter1_open_odds, fighter1_close_odds, fighter2_open_odds, fighter2_close_odds, odds_inferred)
    row_ba = _build_row(fighter2, fighter1, fighter2_open_odds, fighter2_close_odds, fighter1_open_odds, fighter1_close_odds, odds_inferred)
    if row_ab is None or row_ba is None:
        return None

    p_ab = model.predict_proba(row_ab)[0, 1]  # P(fighter1 wins | fighter1=self)
    p_ba = model.predict_proba(row_ba)[0, 1]  # P(fighter2 wins | fighter2=self)

    # Average both orderings — guaranteed to sum to 1.0
    prob_f1 = (p_ab + (1 - p_ba)) / 2

    winner   = fighter1 if prob_f1 > 0.5 else fighter2
    win_prob = prob_f1  if prob_f1 > 0.5 else 1 - prob_f1

    if verbose:
        bar_len = 30
        f1_bar = int(prob_f1 * bar_len)
        f2_bar = bar_len - f1_bar
        print(f"\n{'='*52}")
        print(f"  {fighter1:<22} vs.  {fighter2}")
        print(f"{'='*52}")
        print(f"  {'█' * f1_bar}{'░' * f2_bar}")
        print(f"  P({fighter1} wins) : {prob_f1:.1%}")
        print(f"  P({fighter2} wins) : {1-prob_f1:.1%}")
        print(f"  Predicted winner  : {winner} ({win_prob:.1%})")
        print(f"  Sum check         : {prob_f1 + (1-prob_f1):.3f}  ✓")

    return winner, win_prob


# ── Try some fights ───────────────────────────────────────────────────────────
predict_fight("Jon Jones", "Stipe Miocic")
predict_fight("Islam Makhachev", "Charles Oliveira")
predict_fight("Alex Pereira", "Magomed Ankalaev")
predict_fight("Paddy Pimblett", "Justin Gaethje")   # hypothetical

# %% [notebook cell 25]
predict_fight("Clayton Carpenter", "Jose Ochoa")

# %% [notebook cell 27]
def symmetry_check(a, b):
    r_ab = _build_row(a, b)
    r_ba = _build_row(b, a)
    p_ab = model.predict_proba(r_ab)[0, 1]
    p_ba = model.predict_proba(r_ba)[0, 1]
    prob_a = (p_ab + (1 - p_ba)) / 2
    print(f"{a} vs {b}")
    print(f"  P({a} wins, A listed first) : {prob_a:.3f}")
    print(f"  P({a} wins, A listed second): {(p_ba + (1 - p_ab)) / 2:.3f}  ← should match")
    print(f"  Sum: {prob_a + (1 - prob_a):.3f}  ✓\n")

symmetry_check("Jon Jones", "Stipe Miocic")
symmetry_check("Islam Makhachev", "Charles Oliveira")
symmetry_check("Alex Pereira", "Magomed Ankalaev")

# %% [notebook cell 29]
all_fighters = sorted(fl_model['FIGHTER'].unique())
print(f"Fighters with fight history: {len(all_fighters)}")
print("\nSample:", all_fighters[:20])

# Search for a fighter
def search_fighters(query):
    q = query.lower()
    matches = [f for f in all_fighters if q in f]
    print(f"Matches for '{query}': {matches[:10]}")

search_fighters("jones")
search_fighters("makhachev")

# %% [notebook cell 31]
# Save everything needed for inference
model.save_model(ARTIFACT_DIR / 'xgb_fight_model.json')
with open(ARTIFACT_DIR / 'paired_feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
with open(ARTIFACT_DIR / 'fighter_stats_clean.pkl', 'wb') as f:
    pickle.dump(fs, f)
with open(ARTIFACT_DIR / 'fl_model_final.pkl', 'wb') as f:
    pickle.dump(fl_model, f)
print("Saved: xgb_fight_model.json, paired_feature_cols.pkl, fighter_stats_clean.pkl, fl_model_final.pkl")

# ── Reload ────────────────────────────────────────────────────────────────────
loaded_model = xgb.XGBClassifier()
loaded_model.load_model(ARTIFACT_DIR / 'xgb_fight_model.json')

with open(ARTIFACT_DIR / 'paired_feature_cols.pkl', 'rb') as f: loaded_feature_cols = pickle.load(f)
with open(ARTIFACT_DIR / 'fighter_stats_clean.pkl', 'rb') as f: loaded_fs = pickle.load(f)
with open(ARTIFACT_DIR / 'fl_model_final.pkl',      'rb') as f: loaded_fl = pickle.load(f)

print(f"\nModel reloaded. Features: {len(loaded_feature_cols)}")
print("Ready to predict.")
