from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("../models/notebooks/ufc_xgboost_model.ipynb")


def set_cell_source(nb: dict, index: int, source: str) -> None:
    nb["cells"][index]["source"] = source.splitlines(keepends=True)


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text())

    cell2 = "".join(nb["cells"][2]["source"])
    cell2 = cell2.replace(
        "RESULTS_PATH        = DATA_DIR + 'ufc_fight_results.csv'",
        "RESULTS_PATH        = DATA_DIR + 'ufc_fight_results_with_odds.csv'",
    )
    set_cell_source(nb, 2, cell2)

    set_cell_source(
        nb,
        6,
        """results = pd.read_csv(RESULTS_PATH)
results['EVENT'] = results['EVENT'].str.strip()
results['BOUT']  = results['BOUT'].str.strip()

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
    ['EVENT', 'BOUT', 'f1', 'f2', 'OUTCOME', 'odds_inferred'] + odds_cols
].copy()
decisive['outcome'] = (decisive['OUTCOME'] == 'W/L').astype(int)  # 1 = f1 won

fl = fight_level.merge(
    decisive[['EVENT', 'BOUT', 'f1', 'f2', 'outcome', 'odds_inferred'] + odds_cols],
    on=['EVENT', 'BOUT'],
    how='inner'
)
fl['is_f1']    = fl['FIGHTER'] == fl['f1']
fl['won']      = np.where(fl['is_f1'], fl['outcome'], 1 - fl['outcome'])
fl['opponent'] = np.where(fl['is_f1'], fl['f2'], fl['f1'])
fl['open_odds'] = np.where(fl['is_f1'], fl['f1_open_odds'], fl['f2_open_odds'])
fl['close_odds'] = np.where(fl['is_f1'], fl['f1_close_odds'], fl['f2_close_odds'])

# Raw data is newest-first; reverse to get oldest-first for rolling
event_order = results[['EVENT']].drop_duplicates().reset_index(drop=True)
event_order['event_idx'] = event_order.index.max() - event_order.index  # 0 = oldest
fl = fl.merge(event_order, on='EVENT', how='left')
fl = fl.sort_values(['FIGHTER', 'event_idx']).reset_index(drop=True)

print(f\"Decisive fights joined: {fl.shape}\")
print(f\"Win rate: {fl['won'].mean():.3f}  (should be exactly 0.5)\")
print(f\"Rows with odds: {fl['open_odds'].notna().sum()}/{len(fl)} fighter rows\")
print(f\"Inferred odds rows: {fl['odds_inferred'].sum()}/{len(fl)} fighter rows\")
""",
    )

    cell12 = "".join(nb["cells"][12]["source"])
    cell12 = cell12.replace(
        "# Self-join: match each fighter row with their opponent's row in the same fight\n"
        "fighter_view = fl_model[['EVENT', 'BOUT', 'FIGHTER', 'opponent', 'won', 'event_idx'] + roll_cols].copy()\n\n"
        "opp_view = fighter_view[['EVENT', 'BOUT', 'FIGHTER'] + roll_cols].copy()\n"
        "opp_view = opp_view.rename(columns={c: f'{c}_opp' for c in roll_cols})\n"
        "opp_view = opp_view.rename(columns={'FIGHTER': 'opponent'})\n\n"
        "self_view = fighter_view.rename(columns={c: f'{c}_self' for c in roll_cols})",
        "# Self-join: match each fighter row with their opponent's row in the same fight\n"
        "current_odds_cols = ['open_odds', 'close_odds']\n"
        "fight_level_cols = ['odds_inferred']\n"
        "fighter_view = fl_model[['EVENT', 'BOUT', 'FIGHTER', 'opponent', 'won', 'event_idx'] + roll_cols + current_odds_cols + fight_level_cols].copy()\n\n"
        "side_specific_cols = roll_cols + current_odds_cols\n"
        "opp_view = fighter_view[['EVENT', 'BOUT', 'FIGHTER'] + side_specific_cols].copy()\n"
        "opp_view = opp_view.rename(columns={c: f'{c}_opp' for c in side_specific_cols})\n"
        "opp_view = opp_view.rename(columns={'FIGHTER': 'opponent'})\n\n"
        "self_view = fighter_view.rename(columns={c: f'{c}_self' for c in side_specific_cols})",
    )
    cell12 = cell12.replace(
        ").drop(columns=['name'])\npaired = paired.dropna()\n\n\n# Column groups",
        ").drop(columns=['name'])\n\n"
        "# Odds are fully populated by ufc_fight_results_with_odds.csv. Keep XGBoost missing-safe anyway.\n"
        "paired = paired.dropna(subset=[c for c in paired.columns if c not in [f'{x}_self' for x in current_odds_cols] + [f'{x}_opp' for x in current_odds_cols]])\n\n\n"
        "# Column groups",
    )
    cell12 = cell12.replace(
        "# Diff features\nfor col in roll_cols:\n    paired[f'diff_{col}'] = paired[f'{col}_self'] - paired[f'{col}_opp']",
        "# Diff features\nfor col in side_specific_cols:\n    paired[f'diff_{col}'] = paired[f'{col}_self'] - paired[f'{col}_opp']",
    )
    cell12 = cell12.replace(
        "diff_cols    = [c for c in paired.columns if c.startswith('diff_')]\n"
        "feature_cols = self_cols + opp_roll_cols + career_self + career_opp + diff_cols",
        "diff_cols    = [c for c in paired.columns if c.startswith('diff_')]\n"
        "fight_level_feature_cols = fight_level_cols\n"
        "feature_cols = self_cols + opp_roll_cols + career_self + career_opp + diff_cols + fight_level_feature_cols",
    )
    set_cell_source(nb, 12, cell12)

    cell24 = "".join(nb["cells"][24]["source"])
    cell24 = cell24.replace(
        "def _build_row(fighter_name, opponent_name):",
        "def _build_row(fighter_name, opponent_name, open_odds=np.nan, close_odds=np.nan, opponent_open_odds=np.nan, opponent_close_odds=np.nan, odds_inferred=1):",
    )
    cell24 = cell24.replace(
        "    for c in roll_cols:\n        row[f'{c}_self'] = fighter_row[c]\n    row.update(opp_rolling)",
        "    for c in roll_cols:\n        row[f'{c}_self'] = fighter_row[c]\n    row['open_odds_self'] = open_odds\n    row['close_odds_self'] = close_odds\n    row.update(opp_rolling)\n    row['open_odds_opp'] = opponent_open_odds\n    row['close_odds_opp'] = opponent_close_odds\n    row['odds_inferred'] = odds_inferred",
    )
    cell24 = cell24.replace(
        "    for c in roll_cols:\n        row[f'diff_{c}'] = row[f'{c}_self'] - row[f'{c}_opp']",
        "    for c in roll_cols + ['open_odds', 'close_odds']:\n        row[f'diff_{c}'] = row[f'{c}_self'] - row[f'{c}_opp']",
    )
    cell24 = cell24.replace(
        "def predict_fight(fighter1: str, fighter2: str, verbose: bool = True):",
        "def predict_fight(\n    fighter1: str,\n    fighter2: str,\n    fighter1_open_odds=np.nan,\n    fighter2_open_odds=np.nan,\n    fighter1_close_odds=np.nan,\n    fighter2_close_odds=np.nan,\n    odds_inferred: int = 1,\n    verbose: bool = True,\n):",
    )
    cell24 = cell24.replace(
        "    row_ab = _build_row(fighter1, fighter2)\n    row_ba = _build_row(fighter2, fighter1)",
        "    row_ab = _build_row(fighter1, fighter2, fighter1_open_odds, fighter1_close_odds, fighter2_open_odds, fighter2_close_odds, odds_inferred)\n    row_ba = _build_row(fighter2, fighter1, fighter2_open_odds, fighter2_close_odds, fighter1_open_odds, fighter1_close_odds, odds_inferred)",
    )
    set_cell_source(nb, 24, cell24)

    NOTEBOOK.write_text(json.dumps(nb, indent=1))
    print(f"Updated {NOTEBOOK}")


if __name__ == "__main__":
    main()
