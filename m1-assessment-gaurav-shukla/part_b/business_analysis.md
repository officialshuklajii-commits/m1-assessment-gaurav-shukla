# Part B: Business Case Analysis
## Scenario: Promotion Effectiveness at a Fashion Retail Chain

**Author:** Gaurav Anand Shukla
**Student ID:** BITSoM_BA_25111017
**Course:** Business Analytics with Gen & Agentic AI (BITSoM × Masai)
**Assessment:** ML Fundamentals — Graded

---

## Context

A fashion retailer operates **50 stores** across urban, semi-urban, and rural locations. Each month, the marketing team deploys one of five promotions: *Flat Discount, BOGO, Free Gift with Purchase, Category-Specific Offer,* and *Loyalty Points Bonus*. Stores vary in size, monthly footfall, local competition density, and customer demographics. The company wants to determine which promotion to deploy in each store each month to **maximise items sold**.

---

## B1. Problem Formulation — 8 marks

### (a) Formulating as an ML Problem — 3 marks

**Target Variable:** `items_sold` — the total number of items sold per store per month.

**Type of ML Problem:** This is a **supervised regression** problem.

**Candidate Input Features:**

| Feature | Type | Justification |
|---------|------|---------------|
| `store_size` | Categorical | Physical capacity determines maximum throughput |
| `location_type` | Categorical | Urban/semi-urban/rural drives footfall and demographics |
| `promotion_type` | Categorical | The core intervention variable — what we are optimising |
| `is_weekend` | Binary | Weekend footfall differs significantly from weekdays |
| `is_festival` | Binary | Festival periods create demand spikes across all store types |
| `competition_density` | Numerical | Nearby competitor count reduces market capture |
| `month` | Ordinal | Seasonal effects — e.g., Oct–Dec peak retail season |
| `day_of_week` | Ordinal | Within-week sales rhythm |
| `is_month_end` | Binary | Month-end salary effect boosts consumer spending (day ≥ 25) |
| Store-level history | Numerical | Rolling 3-month average items sold as autoregressive signal |

**Justification for Regression over Classification:**
The target `items_sold` is a **continuous numerical count**. The business goal is to maximise the *quantity* of items sold — not to classify stores into high/low buckets. Regression directly predicts the expected volume for each promotion type, enabling the marketing team to compare predicted volumes and choose the best promotion per store per month (e.g., "Store 12 in December: Flat Discount → 412 items vs BOGO → 387 items → choose Flat Discount"). A classification model would lose this crucial quantitative signal.

---

### (b) Items Sold vs Total Sales Revenue — 3 marks

**Why `items_sold` is a more reliable target than total sales revenue:**

1. **Promotions alter price directly.** Flat Discount reduces unit price; BOGO effectively halves per-unit revenue. A model trained to maximise revenue would learn to *avoid* discounts (since they reduce price), even when those discounts generate far more volume — directly contradicting the company's goal of selling more items.

2. **Revenue conflates volume with product mix.** A store selling 50 premium jackets at ₹5,000 each produces the same revenue as one selling 500 basic accessories at ₹500 each — yet the business outcome is entirely different. `items_sold` strips away this confounding product-mix effect and measures the promotion's true volume impact.

3. **Direct goal alignment.** The problem statement says "maximise the number of items sold." `items_sold` is a **direct, unambiguous measurement of that goal** with no intermediate confounders.

**Broader Principle — Target Variable Selection in ML:**
This illustrates the principle of **metric-goal alignment**: the target variable must directly measure what the business actually wants to optimise, not a downstream financial proxy. Using a proxy target (revenue) introduces confounders that cause the model to optimise the proxy at the expense of the real goal — a phenomenon known as *Goodhart's Law* in ML systems. Choosing the right target is often more impactful on business outcomes than choosing the right algorithm.

---

### (c) Alternative to One Global Model — 2 marks

**Problem with a single global model across all 50 stores:**
Stores in urban, semi-urban, and rural locations respond *very differently* to the same promotion. A BOGO offer in a high-footfall urban store may drive 300+ extra items; the same offer in a rural store with 20 daily visitors may generate almost no uplift. Forcing one model to fit all contexts produces mediocre predictions everywhere by averaging across incompatible response patterns.

**Proposed Strategy: Location-Stratified Models**

Train **three separate regression models** — one per `location_type` (Urban, Semi-Urban, Rural):

1. **Segment** all 50 stores by location type (e.g., 20 Urban, 18 Semi-Urban, 12 Rural stores).
2. **Train a separate Random Forest or Linear Regression pipeline** for each segment using that segment's historical store-month records.
3. Each model learns the promotion-response function specific to its customer demographic and competitive context.
4. At prediction time, route each store to its segment model.

**Justification:** This is a **hierarchical modelling** strategy. Each sub-model has sufficient data (~20 stores × 36 months = 720 store-month records) to train reliably, while capturing context-specific patterns a global model would suppress. An equivalent approach is to include `location_type × promotion_type` interaction features inside a single model — which achieves the same goal without splitting the data, and is preferable when some location types have fewer stores.

---

## B2. Data and EDA Strategy — 10 marks

### (a) Joining the Four Tables — 4 marks

**Raw tables:**
- **`transactions`** — one row per transaction: `(transaction_id, store_id, transaction_date, items_sold)`
- **`store_attributes`** — one row per store: `(store_id, store_size, location_type)`
- **`promotion_details`** — one row per store-month: `(store_id, month, promotion_type)`
- **`calendar`** — one row per date: `(date, is_weekend, is_festival)`

**Join Strategy:**

```sql
-- Step 1: Aggregate transactions to store-month grain
monthly_sales AS (
    SELECT
        store_id,
        DATE_TRUNC('month', transaction_date)  AS month,
        SUM(items_sold)                         AS items_sold,
        COUNT(DISTINCT transaction_date)        AS active_days
    FROM transactions
    GROUP BY store_id, month
),

-- Step 2: Join store attributes (many-to-one on store_id)
with_store AS (
    SELECT m.*, s.store_size, s.location_type
    FROM monthly_sales m
    LEFT JOIN store_attributes s ON m.store_id = s.store_id
),

-- Step 3: Join promotion details (one-to-one on store_id + month)
with_promo AS (
    SELECT w.*, p.promotion_type
    FROM with_store w
    LEFT JOIN promotion_details p ON w.store_id = p.store_id AND w.month = p.month
),

-- Step 4: Aggregate calendar flags per month, then join
calendar_monthly AS (
    SELECT
        DATE_TRUNC('month', date)   AS month,
        MAX(is_festival)            AS has_festival,
        SUM(is_weekend)             AS num_weekend_days
    FROM calendar GROUP BY month
)
SELECT w.*, c.has_festival, c.num_weekend_days
FROM with_promo w
LEFT JOIN calendar_monthly c ON w.month = c.month
```

**Grain of Final Modelling Dataset:** **One row = one store × one calendar month**
(e.g., "Store 12, October 2024" with items_sold=412, store_size=large, promotion_type=flat_discount, has_festival=1)

**Pre-modelling aggregations to compute:**
- `lag_1_items_sold` — previous month's items sold per store (autoregressive feature)
- `rolling_3m_avg` — 3-month rolling mean of items sold per store (captures trend momentum)
- `num_weekend_days` — count of weekend days in the month (more precise than binary flag)
- `promotion_change` — binary flag if promotion type changed from previous month (novelty effect)

---

### (b) EDA Strategy — 4 marks

**Four essential analyses before modelling:**

**Analysis 1 — Promotion Type vs Mean Items Sold (Grouped Bar Chart with Error Bars)**
- **Chart**: Bar chart of mean `items_sold` per `promotion_type` with ±1 standard deviation error bars.
- **What to look for**: Which promotion type has the highest mean? How much variance exists within each type? Do confidence intervals overlap significantly?
- **Decision impact**: If Flat Discount consistently outperforms others across all stores, it should be the baseline. High within-promotion variance signals **context dependence** — the same promotion performs very differently across store types — confirming that interaction features (promotion × location) or stratified models are necessary.

**Analysis 2 — Monthly Sales Trend (Time-Series Line Plot)**
- **Chart**: Line plot of total `items_sold` aggregated per month (Jan 2022–Dec 2024) across all 50 stores.
- **What to look for**: Seasonal peaks (festival months, year-end), any upward/downward trend in baseline demand, outlier months.
- **Decision impact**: Strong seasonality confirms that `month` and `is_festival` must be included as features, and validates why a **temporal** (not random) train-test split is mandatory. If a clear upward trend exists, adding `year` as a feature is justified.

**Analysis 3 — Location Type × Promotion Heatmap (Interaction Analysis)**
- **Chart**: Heatmap with `location_type` on the y-axis, `promotion_type` on the x-axis, and cell values = mean `items_sold`.
- **What to look for**: Do certain promotions dominate only in specific locations? Strong off-diagonal patterns (e.g., BOGO works in urban but not rural) indicate **interaction effects**.
- **Decision impact**: If interactions are strong, we must either (a) create explicit `promotion × location` interaction features, or (b) use location-stratified models. Without this analysis, a global model would produce systematic prediction errors for minority-location stores.

**Analysis 4 — Competition Density vs Items Sold (Scatter Plot + Regression Line)**
- **Chart**: Scatter plot of `competition_density` (x-axis) vs `items_sold` (y-axis), points coloured by `store_size`, with a LOWESS smoothing line.
- **What to look for**: Is the relationship linear or does it curve (e.g., items sold drops steeply until 3 competitors, then plateaus)? Do large stores absorb competition better than small stores?
- **Decision impact**: A non-linear relationship validates the use of Random Forest over Linear Regression. An interaction with `store_size` (competition affects small stores more) would justify creating `competition_density × store_size` as an engineered feature — improving model accuracy meaningfully.

---

### (c) Handling 80% No-Promotion Transactions — 2 marks

**How this imbalance harms the model:**
If 80% of store-month records have no promotion, a naive model learns that "no promotion" is the default state and may **underweight the 20% promoted observations** that contain the most actionable business signal. The model may achieve apparently good average RMSE by predicting near-baseline values for all stores — while completely failing to capture promotion uplift.

**Steps to address this:**

1. **Explicit `has_promotion` feature**: Create a binary column (1 = any promotion active, 0 = no promotion). This allows the model to explicitly learn the baseline-vs-promoted performance differential before distinguishing between promotion types.

2. **Stratified sampling in cross-validation**: Ensure each CV fold contains proportional representation of promoted vs non-promoted store-months. Prevents folds where promoted records are absent, producing misleading CV scores.

3. **Uplift modelling (Two-Model Approach)**: Train **Model A** on the 80% no-promotion records to learn baseline demand per store-month. Train **Model B** on the 20% promoted records to predict the *incremental uplift* above baseline. The final recommendation is: choose the promotion with the highest predicted uplift for each store-month. This focuses Model B entirely on the signal most relevant to the business question.

4. **Weighted training**: Apply `sample_weight` to up-weight promoted store-month records during model fitting, forcing the model to prioritise accuracy on the commercially relevant minority.

5. **Collect more data deliberately**: Design a structured monthly A/B experiment where a rotating ~40% of stores are assigned different promotion types, increasing the fraction of promoted observations over time from 20% to 40%+ — improving model precision and coverage across promotion types.

---

## B3. Model Evaluation and Deployment — 12 marks

### (a) Train-Test Split and Evaluation Metrics — 4 marks

**Train-Test Split Setup:**

Given 3 years of monthly store-level data (36 months × 50 stores = 1,800 store-month records):

- **Sort all records chronologically** by month.
- **Training set**: Months 1–30 (first 30 months — Years 1, 2, and first half of Year 3).
- **Test set**: Months 31–36 (the final 6 months of Year 3).

**Why random split is inappropriate:**
Monthly retail data has strong temporal autocorrelation — sales in month T are correlated with month T–1 (momentum) and the same month in previous years (seasonality). A random split places future months in the training set, allowing the model to "learn from the future" — a form of data leakage that produces test scores that are impossibly optimistic and will not generalise to real deployment. The test set must always represent a **strictly future period** the model has never seen.

**Evaluation Metrics and Business Interpretation:**

| Metric | Definition | Business Interpretation |
|--------|-----------|------------------------|
| **RMSE** (Root Mean Squared Error) | √( mean((actual − predicted)²) ) | Average prediction error in items. **Penalises large errors heavily** — a 200-item error counts 4× more than a 100-item error. Critical when large mispredictions cause costly overstock or severe stockouts. *Lower is better.* |
| **MAE** (Mean Absolute Error) | mean( |actual − predicted| ) | Average absolute error per store-month. **More interpretable for operations**: "On average, our forecast is off by X items." Less sensitive to outlier predictions. *Lower is better.* |
| **MAPE** (Mean Absolute Percentage Error) | mean( |actual − predicted| / actual ) | Percentage error — **essential for comparing stores of different sizes**. A 30-item error is minor for an urban store selling 500 items/month, but catastrophic for a rural store selling 50 items/month. Targets <15% for reliable promotional planning. |

**Practical interpretation**: If RMSE = 27 items, the marketing team treats each store recommendation with ±27 item uncertainty and sets safety stock accordingly. If MAPE = 12%, the model is reliable enough for monthly promotion decisions across all store sizes.

---

### (b) Explaining Different Recommendations via Feature Importance — 4 marks

**Scenario**: The model recommends Loyalty Points Bonus for Store 12 in December but Flat Discount for Store 12 in March.

**Step 1 — Global Feature Importance:**
Extract the Random Forest's global feature importance scores (mean decrease in impurity across all trees). From our Q3 analysis, the top drivers are: `is_festival`, `store_size_small`, `location_type_urban`, `day_of_week`, `is_weekend`. This tells us that *festival timing* and *store context* are the primary forces shaping recommendations — not just which promotion is run.

**Step 2 — SHAP Analysis for Local Explanations:**
For the two specific store-month predictions, compute individual SHAP (SHapley Additive exPlanations) values:

```python
import shap
explainer = shap.TreeExplainer(rf_model)

# Store 12, December (Loyalty Points recommended)
shap_dec = explainer.shap_values(X_store12_december)
shap.waterfall_plot(shap_dec[0])
# → Shows: is_festival=1 contributes +52 items, month=12 contributes +38 items
# → Loyalty Points amplifies these context effects more than Flat Discount

# Store 12, March (Flat Discount recommended)
shap_mar = explainer.shap_values(X_store12_march)
shap.waterfall_plot(shap_mar[0])
# → Shows: is_festival=0 contributes -18 items, is_weekend rate lower
# → Flat Discount provides immediate price incentive to drive non-organic traffic
```

**Communicating to the Marketing Team (plain-language narrative):**

> *"In December, Store 12 falls during the festival season (`is_festival=1`) and the peak retail month. Our model finds that customers visiting during festivals are already motivated to buy — they respond more to perceived long-term value (loyalty points accumulate across purchases) than to immediate discounts. Loyalty Points Bonus amplifies the festival demand effect by an estimated +52 items.*
>
> *In March, there is no festival and footfall is below average. Here, customers need an immediate incentive to visit the store at all — Flat Discount provides a clear, tangible price reason to choose this store over competitors. The model estimates Flat Discount generates approximately +38 items more than Loyalty Points in this low-season context."*

This translates the model's mathematical SHAP values into a **causal business narrative** the marketing team can validate with domain knowledge, challenge with market intelligence, and act on confidently.

---

### (c) End-to-End Deployment Process — 4 marks

**Goal**: Generate monthly promotion recommendations for all 50 stores at the start of each month, without retraining every month.

---

**Step 1 — Model Serialisation**

```python
import joblib

# Save entire pipeline (preprocessor + model — one object, no leakage risk)
joblib.dump(lr_pipeline, 'models/promotion_recommender_v1.pkl')

# Save version metadata alongside
metadata = {
    'version'       : '1.0',
    'trained_on'    : '2022-01-01 to 2024-06-11',
    'test_period'   : '2024-06-12 to 2024-12-31',
    'test_rmse'     : 27.13,
    'test_mae'      : 21.07,
    'features'      : cat_features + num_features,
    'n_stores'      : 50
}
import json
with open('models/model_metadata_v1.json','w') as f:
    json.dump(metadata, f)
```

Saving the **complete pipeline** (not just the model weights) ensures the same OHE categories and StandardScaler statistics applied during training are automatically reused at inference — preventing transformation mismatch bugs in production.

---

**Step 2 — Monthly Data Preparation**

At the start of each month (e.g., November 1st), the data pipeline:

1. Pulls all 50 stores' static attributes: `store_size`, `location_type`, `competition_density`
2. Constructs date features for the target month: `month=11`, `year=2024`, `day_of_week`, `is_month_end` patterns
3. Fetches the upcoming month's calendar: `is_festival` flags (from company calendar)
4. **Generates all 5 promotion scenarios for all 50 stores** = 250 prediction rows:

```python
import itertools

stores    = store_attributes_df  # 50 rows
promos    = ['flat_discount','bogo','free_gift','loyalty_points','category_offer']
scenarios = []

for _, store in stores.iterrows():
    for promo in promos:
        scenarios.append({
            'store_id'           : store['store_id'],
            'store_size'         : store['store_size'],
            'location_type'      : store['location_type'],
            'promotion_type'     : promo,
            'competition_density': store['competition_density'],
            'month'              : 11,   # November
            'year'               : 2024,
            'day_of_week'        : 5,    # November 2024 starts Saturday
            'is_weekend'         : 1,
            'is_festival'        : 0,
            'is_month_end'       : 0
        })

scenarios_df = pd.DataFrame(scenarios)
print(f'Total prediction rows: {len(scenarios_df)}')   # 250
```

---

**Step 3 — Generating Recommendations**

```python
# Load saved pipeline
model = joblib.load('models/promotion_recommender_v1.pkl')

# Predict items_sold for all 250 scenarios
scenarios_df['predicted_items_sold'] = model.predict(scenarios_df.drop('store_id', axis=1))

# For each store, select the promotion with the highest predicted items_sold
recommendations = (
    scenarios_df
    .groupby('store_id')
    .apply(lambda g: g.loc[g['predicted_items_sold'].idxmax()])
    .reset_index(drop=True)
)[['store_id','promotion_type','predicted_items_sold']]

print('November 2024 — Promotion Recommendations:')
print(recommendations.head(10).to_string())
# Output: store_id → best promotion → predicted volume
```

This runs in seconds and requires no retraining.

---

**Step 4 — Monitoring and Retraining**

| Monitoring Type | Metric | Alert Threshold | Action |
|-----------------|--------|-----------------|--------|
| **Prediction accuracy** | Monthly RMSE on actuals vs predictions | RMSE rises >20% above baseline (27.13) | Investigate data quality; consider retraining |
| **Data drift** | KS-test on key input feature distributions (competition_density, is_festival rate) | p-value < 0.05 for any key feature | Flag for re-evaluation; retrain if sustained |
| **Concept drift** | Promotion effectiveness shift (new competitor opens, consumer trend changes) | Any single promotion's mean uplift drops >30% | Retrain immediately |
| **Business performance** | Month-over-month total items sold vs model-recommended period baseline | >15% sustained decline | Escalate to business review + model audit |

**Retraining cadence**: Retrain every **6 months** incorporating all newly collected data. Use the most recent 6 months as the holdout test set each time. This keeps the model current with evolving consumer behaviour, new store openings, and shifting competitive landscape.

**Deployment architecture summary:**

```
Monthly Data Pipeline
        ↓
Feature Engineering (date extraction, scenario generation × 5 promotions × 50 stores)
        ↓
Saved Pipeline (joblib) → 250 Predictions
        ↓
argmax per store_id → 50 Recommendations
        ↓
Monthly Recommendation Report → Marketing Team
        ↓
Actuals collected next month → Monitoring Dashboard → Alert System → Retraining Trigger
```

---

*Submitted by: Gaurav Anand Shukla | ID: BITSoM_BA_25111017 | Course: Business Analytics with Gen & Agentic AI*
