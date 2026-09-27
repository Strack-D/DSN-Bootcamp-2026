# DSN Mart Sales Prediction — DSN Bootcamp ML Track Qualifier

## Problem Statement
Predict `total_sales` for a given product at a given DSN Mart store, using historical product
and store characteristics. This is the qualifying hackathon for the DSN AI Bootcamp
([competition link](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track)).

## Dataset Overview
- **Training data:** 6,818 records × 13 features
- **Test data:** 1,705 records × 12 features (target removed)
- **Target variable:** `total_sales` (complete, no missing values)
- Only **10 unique stores** and **1,555 unique products** — a sparse product × store grid,
  not a full cross join (only ~55% of possible product-store combinations actually occur)

## Data Quality
| Feature | Missing (%) | Handling |
|---|---|---|
| `store_size` | 28.15% | Imputed from the same store's own known value where available; explicit `"Unknown"` category for the 3 stores that never report it at all |
| `product_weight_kg` | 17.97% | Imputed from the same product's known weight where available, falling back to the category median for the 4 products missing everywhere |
| All others | 0% | Clean, no duplicate rows |

## Feature Breakdown
**Numerical (5 raw):** `product_weight_kg`, `shelf_visibility`, `product_price`, `store_age_years`, `total_sales`

**Categorical (5 raw, before engineering):**
- `fat_content` — 2 categories (Low Fat, Regular), already clean
- `product_category` — 16 real categories, but 48 raw string values due to inconsistent casing (Title/lower/UPPER variants of the same labels) — needed standardization
- `store_size` — 3 categories (Small, Medium, Large) + missing
- `store_location_tier` — 3 tiers
- `store_format` — 4 types: Standard Supermarket (4,462), Corner Shop (866), Flagship Hypermarket (748), Superstore (742)

**Identifiers:** `id`, `product_code` (1,555 unique), `store_code` (10 unique)

## Key Insights from EDA
- **Only 10 stores drive every store-level attribute.** `store_size`, `store_format`,
  `store_location_tier`, and `store_age_years` don't vary within a store — they're fully
  determined by `store_code`. This shaped the encoding strategy: `store_code` is used
  directly as a feature rather than encoded, since at only 10 levels there's no cardinality
  problem to solve.
- **`product_category` needed real cleanup** — 48 raw strings collapse to 16 categories once
  case-normalized.
- **`shelf_visibility` contains disguised missing data.** Exact-zero values (~6% of rows)
  aren't real measurements — a listed product can't have 0% shelf space — and were treated
  as missing, imputed by category mean.
- **Standard Supermarket dominates** (65.4% of training rows) and **Tier_3 locations are
  most common** (39.3%) — noted for later error analysis, not severe enough to need
  resampling.
- **`total_sales` is right-skewed** (mean ~2,175, max ~12,997) — a log transform was tested
  but the raw scale performed at least as well and was used for the final models.

## Approach
1. Data Cleaning
2. Feature Engineering
3. Multicollinearity Check (VIF)
4. Model Selection
5. Hyperparameter Tuning
6. Ensemble
7. Submission

### 1. Data Cleaning
- Normalized `product_category` casing.
- Treated `shelf_visibility == 0` as missing, imputed by category mean.
- Two-step imputation for `product_weight_kg` (same-product mean, then category median) and
  `store_size` (same-store mode, then `"Unknown"`).
- **Result:** 0 missing values remaining.

### 2. Feature Engineering
- `price_per_kg` — normalizes price against physical size.
- `weight_visibility`, `price_visibility` — interaction terms combining physical/price
  attributes with shelf prominence.
- `visibility_ratio` — relative shelf-space prominence vs. the category norm (`price_rank_in_category`
  was tried and dropped in favor of this).
- Interaction categoricals: `category_fat` (surfaces that `hard drinks` / `health and
  hygiene` / `household` / `others` are 100% "Low Fat" — a meaningless default label for
  non-food categories, not real variation), `category_store_type`, `format_tier`,
  `product_format`, `category_store`.
- K-fold target encoding applied to `product_code`, `product_category`, `category_store`,
  and `product_format` — the last of these is flagged as risky (~4,314 distinct
  combinations from ~8,500 rows, most occurring once or twice), so its contribution is
  worth an ablation check.
- **Result:** 20 engineered features from 9 raw predictor columns.

### 3. Multicollinearity Check (VIF)
- VIF computed on the continuous numerics (including the target-encoded columns) — all
  within a normal range (highest is `shelf_visibility` at ~13.5, everything else under 11).
- Separately, a crosstab check confirms `format_tier` (`store_format` × `store_location_tier`)
  is structurally redundant — only 7 of 12 possible combinations exist, because it's really
  just re-describing which of the 10 stores a row belongs to. Kept in the final feature set
  to let feature importance make the final call rather than assuming.

### 4. Model Selection
Four genuinely different models, not just tree variants:

| Model | CV RMSE (baseline) |
|---|---|
| LightGBM | 1088.97 |
| XGBoost | 1083.22 |
| CatBoost | 1071.45 |
| Structural model (store throughput × product price, shrinkage-adjusted) | 1069.49 |

The structural model is a hand-derived hypothesis — `total_sales ≈ store units sold ×
product price` — rather than a generic learner; it never sees any of the engineered
features, which makes it a genuinely diverse ensemble member rather than a weaker version
of the trees.

### 5. Hyperparameter Tuning
Optuna (TPE sampler + median pruning), 50 trials per model, 5-fold CV:

| Model | Tuned CV RMSE |
|---|---|
| LightGBM | 1081.48 |
| XGBoost | 1076.75 |
| CatBoost | 1071.18 |

### 6. Ensemble
LightGBM was dropped from the final blend — it was the weakest performer both before and
after tuning, and excluding it improved the result:

- Weighted blend of XGBoost + CatBoost + structural model (weights: 0.13 / 0.29 / 0.58):
  **1067.45**
- Ridge stacking meta-model on the same three models' OOF predictions: **1067.42** ← used
  for final submission

This resolves an earlier concern: with all four models blended, the ensemble (1070.21) was
actually slightly *worse* than the standalone structural model (1069.49). Dropping the
correlated, underperforming LightGBM let the blend genuinely add value — the final ensemble
(1067.42) now clearly beats every individual model, including the structural model alone.
The structural model earning the largest blend weight (0.58) tracks with it being the
strongest individual performer.

## Final Model Performance
- **Final OOF RMSE:** 1067.42 (Ridge-stacked ensemble of XGBoost, CatBoost, and the
  structural model)
- Predictions clipped at 0 as a safeguard (sales can't be negative) — not actually
  triggered on this run; the lowest prediction came in at ~69

**Submission-ready ✓**

## How to Submit
1. Run the notebook end to end to generate `submission.csv`.
2. Go to the [DSN Bootcamp Kaggle competition](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track).
3. Upload `submission.csv` and check the leaderboard.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/dsn_mart_sales_pipeline.ipynb
# or open directly in Google Colab
```

## Author
Stephen | September 2026
