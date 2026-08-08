# ══════════════════════════════════════════════════════════════════════════════
#  CREDIT RISK MODELLING — PROBABILITY OF DEFAULT
#  Dataset : German Credit Data (modeldata package, 4,454 loan applicants)
#  Models  : Logistic Regression (industry standard) + XGBoost (benchmark)
#  Metrics : Gini Coefficient, KS Statistic, ROC-AUC, PR-AUC
# ══════════════════════════════════════════════════════════════════════════════
#
#  WHAT IS CREDIT RISK?
#  ─────────────────────
#  Credit risk is the risk that a borrower fails to repay a loan as agreed.
#  Banks and lenders must quantify this risk to:
#    (a) Decide whether to approve or reject a loan application
#    (b) Set appropriate interest rates (riskier borrowers pay more)
#    (c) Calculate regulatory capital reserves (Basel II / III framework)
#    (d) Manage their overall portfolio risk
#
#  THE CREDIT RISK FRAMEWORK — THE THREE COMPONENTS
#  ──────────────────────────────────────────────────
#  Credit risk is decomposed into three key quantities:
#
#    PD  (Probability of Default)
#         The probability that a borrower will fail to repay within a given
#         time horizon (typically 1 year). This is what our model predicts.
#         Example: PD = 0.08 means 8% chance of default.
#
#    LGD (Loss Given Default)
#         If the borrower does default, what fraction of the outstanding
#         balance will the bank lose? This depends on collateral, recovery
#         processes, and loan type.
#         Example: LGD = 0.45 means the bank loses 45% of the loan.
#
#    EAD (Exposure at Default)
#         The total amount owed at the time of default (can differ from the
#         original loan amount due to partial repayments or credit lines).
#         Example: EAD = €10,000 remaining balance.
#
#    Expected Loss (EL) = PD × LGD × EAD
#         The average loss the bank expects to incur. This drives loan pricing
#         and provisioning decisions.
#         Example: 0.08 × 0.45 × €10,000 = €360 expected loss per loan.
#
#  THIS SCRIPT FOCUSES ON PD MODELLING.
#  We build a statistical model that takes loan application characteristics
#  as input and outputs the probability that the applicant will default.
#
#  THE REGULATORY CONTEXT — BASEL II / III
#  ─────────────────────────────────────────
#  Under the Basel Capital Accords, banks are required to hold capital
#  proportional to their credit risk. Banks using the "Internal Ratings
#  Based" (IRB) approach can use their own PD models to calculate capital
#  requirements — subject to regulatory approval.
#
#  This places strict requirements on PD models:
#    - Interpretability: regulators need to understand and audit the model
#    - Stability: model performance must be monitored over time (PSI)
#    - Validation: independent validation required (backtesting, benchmarking)
#    - Documentation: every decision must be documented
#
#  This is why LOGISTIC REGRESSION remains the dominant model in credit risk,
#  even though machine learning methods often perform better statistically.
#  Logistic regression coefficients have a clear interpretation (log-odds),
#  can be converted into a points-based scorecard, and are easy to audit.
#
#  THE DATASET
#  ────────────
#  We use the German Credit dataset (Statlog version), originally collected
#  from a German bank. Each row is one loan applicant with:
#    - 13 predictor variables (demographics, financial history, loan details)
#    - 1 outcome: Status = "bad" (defaulted) or "good" (repaid)
#
#  Variables:
#    Seniority : years of employment seniority
#    Home      : home ownership type (owner, rent, parents, etc.)
#    Time      : duration of the loan in months
#    Age       : applicant's age in years
#    Marital   : marital status
#    Records   : whether the applicant has court records (yes/no)
#    Job       : employment type (fixed, parttime, freelance, etc.)
#    Expenses  : monthly household expenses
#    Income    : annual income
#    Assets    : total asset value
#    Debt      : existing debt
#    Amount    : loan amount requested
#    Price     : price of the item being purchased
#    Status    : OUTCOME — "bad" = defaulted, "good" = repaid
#
# ══════════════════════════════════════════════════════════════════════════════


# ── 0. Packages ───────────────────────────────────────────────────────────────
#
# tidymodels : unified modelling framework (recipes, parsnip, yardstick, tune)
# modeldata  : contains the credit_data dataset
# patchwork  : combine ggplot2 figures side by side or stacked

library(tidymodels)
library(modeldata)
library(ggplot2)
library(patchwork)
library(dplyr)

# install.packages(c("tidymodels", "modeldata", "xgboost", "patchwork", "bundle"))


# ── 1. Load Data ──────────────────────────────────────────────────────────────

data(credit_data)
cat("Rows:", nrow(credit_data), " | Columns:", ncol(credit_data), "\n")
glimpse(credit_data)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION A — EXPLORATORY DATA ANALYSIS
#
#  Before building any model, we need to understand:
#    1. The default rate (how imbalanced is the outcome?)
#    2. The distribution of predictors across good and bad borrowers
#    3. Missing data patterns
#    4. Relationships between variables
#
#  EDA informs every modelling decision: metric choice, preprocessing steps,
#  which variables may be most predictive, and potential data quality issues.
# ══════════════════════════════════════════════════════════════════════════════

# ── 2a. Default Rate ──────────────────────────────────────────────────────────
#
#  "Bad" in this dataset means the borrower defaulted on their loan.
#  The default rate is the most fundamental statistic in credit risk —
#  it tells us the base rate a naive model must beat, and it drives
#  the imbalance corrections needed in the model.
#
#  A default rate of ~30% is moderately imbalanced. Unlike fraud detection
#  (0.17%), this is manageable without extreme corrections, but we still
#  need to use stratified sampling and track PR-AUC alongside ROC-AUC.

default_counts <- credit_data |>
  count(Status) |>
  mutate(pct = scales::percent(n / sum(n), accuracy = 0.1))

print(default_counts)

p_default <- ggplot(default_counts, aes(x = Status, y = n, fill = Status)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = paste0(n, "\n(", pct, ")")), vjust = -0.4, size = 4.5) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.15))) +
  scale_fill_manual(values = c(bad = "#E05C5C", good = "#4E9AF1")) +
  labs(
    title    = "Loan Default Rate",
    subtitle = "Status = 'bad' means the borrower defaulted — our target event",
    x = NULL, y = "Number of Applicants"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

print(p_default)


# ── 2b. Missing Values ────────────────────────────────────────────────────────
#
#  In credit risk modelling, missing data is common and carries meaning.
#  A missing income field might indicate a self-employed applicant who
#  doesn't have a formal salary slip. Missing asset information may mean
#  no assets. We must handle NAs before modelling.
#
#  We will use median imputation for numeric variables and mode (most
#  frequent value) imputation for categorical ones. These are applied
#  within the recipe to prevent data leakage — imputation statistics are
#  computed on the training fold only, then applied to both folds.

missing_summary <- credit_data |>
  summarise(across(everything(), ~sum(is.na(.)))) |>
  tidyr::pivot_longer(everything(), names_to = "variable", values_to = "n_missing") |>
  mutate(pct_missing = scales::percent(n_missing / nrow(credit_data), accuracy = 0.1)) |>
  filter(n_missing > 0) |>
  arrange(desc(n_missing))

cat("\n── Variables with Missing Values ──\n")
print(missing_summary)

# Visualise missing data
if (nrow(missing_summary) > 0) {
  ggplot(missing_summary, aes(x = n_missing, y = reorder(variable, n_missing))) +
    geom_col(fill = "#E05C5C") +
    geom_text(aes(label = pct_missing), hjust = -0.1, size = 4) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(
      title    = "Missing Values by Variable",
      subtitle = "Will be handled via median/mode imputation in the recipe",
      x = "Number of Missing Values", y = NULL
    ) +
    theme_minimal(base_size = 13)
}


# ── 2c. Default Rate by Categorical Variables ─────────────────────────────────
#
#  For each categorical predictor, we compute the default rate within each
#  category. This is the foundation of Weight of Evidence (WoE) analysis —
#  a traditional credit risk technique where categories are ranked by their
#  default rate and encoded as log-odds transformations.
#
#  Even without formal WoE, this plot tells us:
#    - Which categories are high-risk vs. low-risk
#    - Whether a variable has discriminatory power (large variation = useful)
#    - Potential data quality issues (unexpected patterns)

cat_vars <- c("Home", "Marital", "Records", "Job")

credit_data |>
  tidyr::pivot_longer(all_of(cat_vars), names_to = "variable", values_to = "category") |>
  group_by(variable, category) |>
  summarise(
    n           = n(),
    default_rate = mean(Status == "bad", na.rm = TRUE),
    .groups = "drop"
  ) |>
  filter(!is.na(category)) |>
  ggplot(aes(x = default_rate, y = reorder(category, default_rate), size = n)) +
  geom_point(color = "#E05C5C", alpha = 0.8) +
  scale_x_continuous(labels = scales::percent, limits = c(0, NA)) +
  scale_size_continuous(range = c(2, 8), name = "n applicants") +
  facet_wrap(~variable, scales = "free_y", ncol = 2) +
  labs(
    title    = "Default Rate by Categorical Variable",
    subtitle = "Larger dots = more applicants | High variation = strong predictor",
    x = "Default Rate", y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")


# ── 2d. Distribution of Numeric Variables by Status ───────────────────────────
#
#  Overlapping density plots reveal how numeric predictors separate defaulters
#  from non-defaulters. A wide gap between the two distributions indicates
#  strong predictive power.
#
#  Key things to look for:
#    - Income: defaulters likely have lower income
#    - Age: younger borrowers may have higher default rates
#    - Amount: larger loans may carry more risk (or larger borrowers are better screened)
#    - Seniority: long-tenured employees may be more stable
#
#  We also look for skewness (long right tails) which might benefit from
#  log-transformation during preprocessing.

num_vars <- c("Seniority", "Time", "Age", "Expenses", "Income", "Assets", "Debt", "Amount")

credit_data |>
  tidyr::pivot_longer(all_of(num_vars), names_to = "variable", values_to = "value") |>
  filter(!is.na(value)) |>
  ggplot(aes(x = value, fill = Status, color = Status)) +
  geom_density(alpha = 0.35) +
  scale_fill_manual(values  = c(bad = "#E05C5C", good = "#4E9AF1")) +
  scale_color_manual(values = c(bad = "#E05C5C", good = "#4E9AF1")) +
  facet_wrap(~variable, scales = "free", ncol = 4) +
  labs(
    title    = "Distribution of Numeric Features by Loan Status",
    subtitle = "Red = defaulted (bad) | Blue = repaid (good) | Separation = predictive power",
    x = NULL, y = "Density", fill = "Status", color = "Status"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")


# ── 2e. Default Rate by Loan Duration and Amount ──────────────────────────────
#
#  Two of the most important credit risk drivers: how long is the loan,
#  and how much is being borrowed? Longer duration and higher amounts
#  typically increase risk. We bin these into quartiles to show the trend.

credit_data |>
  filter(!is.na(Amount), !is.na(Time)) |>
  mutate(
    Amount_q = cut(Amount, breaks = quantile(Amount, probs = c(0,.25,.5,.75,1), na.rm = TRUE),
                   include.lowest = TRUE, labels = c("Q1 (low)", "Q2", "Q3", "Q4 (high)")),
    Time_q   = cut(Time, breaks = quantile(Time, probs = c(0,.25,.5,.75,1), na.rm = TRUE),
                   include.lowest = TRUE, labels = c("Q1 (short)", "Q2", "Q3", "Q4 (long)"))
  ) |>
  group_by(Amount_q, Time_q) |>
  summarise(default_rate = mean(Status == "bad", na.rm = TRUE), n = n(), .groups = "drop") |>
  ggplot(aes(x = Amount_q, y = Time_q, fill = default_rate)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = scales::percent(default_rate, accuracy = 1)), size = 4) +
  scale_fill_gradient2(
    low = "#4E9AF1", mid = "#FFFFCC", high = "#E05C5C",
    midpoint = 0.3, labels = scales::percent
  ) +
  labs(
    title    = "Default Rate by Loan Amount and Duration",
    subtitle = "Darker red = higher default risk | Cells show % defaulted",
    x = "Loan Amount Quartile", y = "Loan Duration Quartile", fill = "Default Rate"
  ) +
  theme_minimal(base_size = 13)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION B — DATA PREPARATION & MODELLING
# ══════════════════════════════════════════════════════════════════════════════

# ── 3. Prepare Outcome Variable ───────────────────────────────────────────────
#
#  We need "bad" (default) to be the positive class — the event we are trying
#  to predict and detect. In tidymodels (yardstick), the SECOND factor level
#  is treated as the positive class by default.
#
#  We relevel so: levels = c("good", "bad") → "bad" is second → positive class.
#  This ensures metrics like ROC-AUC, PR-AUC, precision, and recall all
#  measure our ability to detect defaults, not repayments.

credit_df <- credit_data |>
  mutate(Status = factor(Status, levels = c("good", "bad")))

cat("Factor levels:", levels(credit_df$Status), "\n")
cat("Positive class (event):", levels(credit_df$Status)[2], "\n")


# ── 4. Train / Test Split ─────────────────────────────────────────────────────
#
#  We split 80% training / 20% test, stratified on Status to preserve the
#  default rate in both sets. The test set is sealed until final evaluation.
#
#  With ~4,454 observations, this gives:
#    Training : ~3,563 applicants
#    Test     : ~891 applicants
#
#  This is smaller than the fraud detection dataset, so we are more cautious
#  about overfitting and rely more heavily on cross-validation for model selection.

set.seed(2947)
split <- initial_split(credit_df, prop = 0.8, strata = Status)
train <- training(split)
test  <- testing(split)

cat("Training:", nrow(train), "| Test:", nrow(test), "\n")
cat("Train default rate:", scales::percent(mean(train$Status == "bad"), accuracy = 0.1), "\n")
cat("Test  default rate:", scales::percent(mean(test$Status  == "bad"), accuracy = 0.1), "\n")


# ── 5. Cross-Validation Folds ─────────────────────────────────────────────────
#
#  With a smaller dataset, 5-fold CV provides stable estimates while keeping
#  each validation fold large enough (~713 observations) to evaluate metrics
#  on. Stratified to preserve the default rate across all folds.

folds <- vfold_cv(train, v = 5, strata = Status)


# ── 6. Metrics ────────────────────────────────────────────────────────────────
#
#  CREDIT RISK STANDARD METRICS
#  ──────────────────────────────
#  ROC-AUC : the standard metric in credit risk, used to report model quality.
#             Also forms the basis of the GINI COEFFICIENT (see Section 13).
#
#  PR-AUC  : more sensitive than ROC-AUC when classes are imbalanced.
#             At ~30% default rate, imbalance is moderate, so PR-AUC adds
#             useful information but is less critical than in fraud detection.
#
#  GINI COEFFICIENT (computed from ROC-AUC in Section 13):
#             Gini = 2 × AUC − 1
#             The dominant performance metric in credit risk / banking.
#             Ranges from 0 (no discrimination) to 1 (perfect discrimination).
#             Industry benchmarks:
#               > 0.30  Acceptable
#               > 0.50  Good
#               > 0.70  Excellent (rare in practice)
#
#  KS STATISTIC (Kolmogorov-Smirnov, computed in Section 13):
#             KS = max(TPR − FPR) across all thresholds
#             The maximum separation between cumulative good and bad score
#             distributions. Also widely used in credit risk.
#               > 0.20  Acceptable
#               > 0.40  Good
#
#  event_level = "second" ensures yardstick treats "bad" (second factor level)
#  as the positive/default class for all metric computations.

metrics <- metric_set(yardstick::roc_auc, yardstick::pr_auc)


# ── 7. Parallel Processing ────────────────────────────────────────────────────

plan(multisession, workers = 6)
cat("Parallel processing: 6 cores\n")


# ── 8. Null Model Baseline ────────────────────────────────────────────────────
#
#  The null model always predicts the majority class ("good"). It establishes
#  the floor that any real model must beat.
#  Expected: ROC-AUC = 0.5, PR-AUC ≈ 0.30 (the default rate base rate).

null_rec  <- recipe(Status ~ ., data = train)
null_spec <- null_model(mode = "classification") |> set_engine("parsnip")
null_wf   <- workflow() |> add_recipe(null_rec) |> add_model(null_spec)

null_res  <- fit_resamples(
  null_wf, resamples = folds, metrics = metrics,
  control = control_resamples(event_level = "second")
)

cat("\n── Null Baseline ──\n")
print(collect_metrics(null_res))


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 1: LOGISTIC REGRESSION
#
#  WHY LOGISTIC REGRESSION IS THE INDUSTRY STANDARD IN CREDIT RISK
#  ────────────────────────────────────────────────────────────────
#  Despite the availability of powerful machine learning methods, logistic
#  regression remains the dominant model in bank credit scoring because:
#
#    1. INTERPRETABILITY: coefficients can be directly interpreted as
#       log-odds ratios, and the model can be converted to a points-based
#       scorecard that loan officers can use and explain to applicants.
#
#    2. REGULATORY COMPLIANCE: Basel II/III requires banks to understand
#       and document their models. Regulators expect to be able to challenge
#       individual coefficients and their economic rationale.
#
#    3. MONOTONICITY: logistic regression enforces monotonic relationships
#       between predictors and the outcome. This is economically sensible
#       (higher income should always reduce default risk, never increase it).
#       Complex models can violate this and produce counterintuitive results.
#
#    4. STABILITY: logistic regression is less prone to overfitting small
#       datasets and degrades gracefully on new data over time.
#
#    5. AUDIT TRAIL: every prediction can be decomposed into the contribution
#       of each variable — essential for explaining rejection decisions to
#       applicants (required by regulation in many jurisdictions, e.g. ECOA).
#
#  HOW LOGISTIC REGRESSION WORKS
#  ──────────────────────────────
#  Logistic regression models the LOG-ODDS of the outcome as a linear function
#  of predictors:
#
#    log(P(default) / P(repay)) = β₀ + β₁×Income + β₂×Age + β₃×Amount + ...
#
#  The log-odds are then converted to a probability via the logistic function:
#
#    P(default) = 1 / (1 + exp(−(β₀ + β₁×X₁ + ...)))
#
#  INTERPRETING COEFFICIENTS
#  ──────────────────────────
#  Each coefficient βⱼ represents the change in log-odds of defaulting
#  associated with a one-unit increase in predictor Xⱼ, holding all else constant.
#
#  To interpret more intuitively: exp(βⱼ) = ODDS RATIO
#    exp(β_income) < 1  → higher income reduces default odds (protective)
#    exp(β_amount) > 1  → larger loans increase default odds (risky)
#    exp(β_age)    < 1  → older applicants have lower default odds (experience)
#
#  RECIPE — PREPROCESSING FOR LOGISTIC REGRESSION
#  ─────────────────────────────────────────────────
#  Unlike tree-based models, logistic regression requires:
#
#    step_impute_median()  : fill numeric NAs with the training median
#    step_impute_mode()    : fill categorical NAs with the most frequent value
#    step_dummy()          : convert categorical variables to 0/1 indicator columns
#                            (logistic regression cannot use factor types directly)
#    step_normalize()      : scale numeric predictors to mean=0, sd=1
#                            This does NOT change predictions, but it makes
#                            coefficients comparable in magnitude (larger |β|
#                            = more influential predictor) and improves the
#                            numerical stability of the optimizer.
#    step_zv()             : remove zero-variance columns (can appear after
#                            dummy encoding if a category has only one level)
# ══════════════════════════════════════════════════════════════════════════════

lr_rec <- recipe(Status ~ ., data = train) |>
  step_impute_median(all_numeric_predictors()) |>          # median impute numerics
  step_impute_mode(all_nominal_predictors()) |>            # mode impute categoricals
  step_dummy(all_nominal_predictors()) |>                  # one-hot encode factors
  step_normalize(all_numeric_predictors()) |>              # standardize to N(0,1)
  step_zv(all_predictors())                                # remove zero-variance columns

lr_spec <- logistic_reg(
  penalty = tune(),    # L2 regularization strength (Ridge regression)
  mixture = 0         # mixture = 0 → Ridge (L2); mixture = 1 → Lasso (L1)
) |>
  set_engine("glmnet")
#
#  WHY REGULARIZATION (penalty)?
#  The penalty parameter adds a constraint that shrinks coefficients toward zero.
#  This reduces overfitting — especially important with small datasets and many
#  dummy-encoded categories. We tune it via cross-validation to find the optimal
#  trade-off between bias and variance.
#
#  mixture = 0 : Ridge (L2) — shrinks all coefficients proportionally,
#                keeping all variables in the model. Standard in credit risk.
#  mixture = 1 : Lasso (L1) — drives some coefficients exactly to zero,
#                performing variable selection. Useful for high-dimensional data.

lr_wf <- workflow() |>
  add_recipe(lr_rec) |>
  add_model(lr_spec)

# Tune the penalty parameter over a grid of 20 values
lr_grid <- grid_regular(penalty(range = c(-4, 0)), levels = 20)

set.seed(1122)
lr_tune <- tune_grid(
  lr_wf,
  resamples = folds,
  grid      = lr_grid,
  metrics   = metrics,
  control   = control_grid(event_level = "second")
)

# Select the penalty value that maximises ROC-AUC
best_lr_penalty <- select_best(lr_tune, metric = "roc_auc")
cat("\nBest logistic regression penalty:", round(best_lr_penalty$penalty, 6), "\n")

# Finalise the workflow with the optimal penalty
lr_wf_final <- finalize_workflow(lr_wf, best_lr_penalty)

# Refit on all 5 folds to get stable CV metrics
set.seed(1122)
lr_res <- fit_resamples(
  lr_wf_final,
  resamples = folds,
  metrics   = metrics,
  control   = control_resamples(event_level = "second", save_pred = TRUE)
)

cat("\n── Logistic Regression CV Results ──\n")
print(collect_metrics(lr_res))


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 2: XGBOOST
#
#  We fit XGBoost as a high-performance benchmark to see how much predictive
#  power is "left on the table" by using logistic regression.
#
#  In practice, the gap between logistic regression and XGBoost narrows
#  after careful feature engineering and WoE transformation. If the gap is
#  large, it may signal that important nonlinear interactions exist in the
#  data — which might be capturable through feature engineering.
#
#  XGBoost does NOT require normalization or dummy encoding (tree-based models
#  handle numeric and categorical splits natively — though in R's parsnip
#  interface, factor variables still need to be encoded).
#
#  We use step_impute_median for NAs (XGBoost can handle NAs natively in
#  Python, but the R parsnip wrapper requires complete data).
# ══════════════════════════════════════════════════════════════════════════════

xgb_rec <- recipe(Status ~ ., data = train) |>
  step_impute_median(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors())

xgb_spec <- boost_tree(
  mode       = "classification",
  trees      = 500,
  learn_rate = 0.05,
  tree_depth = 4,       # shallower than default (6) to reduce overfitting on small data
  min_n      = 10       # higher than default (5) — more conservative on small dataset
) |>
  set_engine("xgboost", eval_metric = "auc")

xgb_wf <- workflow() |>
  add_recipe(xgb_rec) |>
  add_model(xgb_spec)

set.seed(6681)
xgb_res <- fit_resamples(
  xgb_wf,
  resamples = folds,
  metrics   = metrics,
  control   = control_resamples(event_level = "second", save_pred = TRUE)
)

cat("\n── XGBoost CV Results ──\n")
print(collect_metrics(xgb_res))


# ── 9. Model Comparison Table ─────────────────────────────────────────────────
#
#  Compare all three models side by side. The null model provides the floor,
#  logistic regression provides the interpretable baseline, and XGBoost shows
#  the performance ceiling. Gini coefficients (2×AUC−1) are shown below.

model_comparison <- bind_rows(
  collect_metrics(null_res) |> mutate(model = "Null baseline"),
  collect_metrics(lr_res)   |> mutate(model = "Logistic Regression"),
  collect_metrics(xgb_res)  |> mutate(model = "XGBoost")
) |>
  select(model, .metric, mean, std_err) |>
  mutate(across(c(mean, std_err), \(x) round(x, 4)))

cat("\n── Model Comparison (5-Fold CV) ──\n")
print(model_comparison)

# Gini coefficients for AUC-based models
auc_comparison <- model_comparison |>
  filter(.metric == "roc_auc") |>
  mutate(gini = round(2 * mean - 1, 4)) |>
  select(model, `ROC-AUC` = mean, Gini = gini)

cat("\n── Gini Coefficients (= 2×AUC − 1) ──\n")
print(auc_comparison)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION C — EVALUATION PLOTS (CROSS-VALIDATION)
# ══════════════════════════════════════════════════════════════════════════════

lr_preds  <- collect_predictions(lr_res)
xgb_preds <- collect_predictions(xgb_res)


# ── 10. ROC Curves — Both Models ──────────────────────────────────────────────
#
#  Plotting both ROC curves on the same axes allows direct visual comparison.
#  The model with the curve closest to the top-left corner is better.
#
#  In credit risk, the ROC curve also underlies the LORENZ CURVE / GINI
#  coefficient — a standard representation of model discriminatory power.
#  The Gini is essentially twice the area between the ROC curve and the
#  diagonal.

lr_roc  <- lr_preds  |> roc_curve(truth = Status, .pred_bad, event_level = "second")
xgb_roc <- xgb_preds |> roc_curve(truth = Status, .pred_bad, event_level = "second")

# Combine for plotting
roc_combined <- bind_rows(
  lr_roc  |> mutate(model = "Logistic Regression"),
  xgb_roc |> mutate(model = "XGBoost")
)

lr_auc  <- collect_metrics(lr_res)  |> filter(.metric == "roc_auc") |> pull(mean)
xgb_auc <- collect_metrics(xgb_res) |> filter(.metric == "roc_auc") |> pull(mean)

ggplot(roc_combined, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_line(linewidth = 1.1) +
  geom_abline(linetype = "dashed", color = "grey60") +   # random classifier diagonal
  scale_color_manual(
    values = c("Logistic Regression" = "#4E9AF1", "XGBoost" = "#E05C5C"),
    labels = c(
      sprintf("Logistic Regression  (AUC = %.3f | Gini = %.3f)", lr_auc,  2*lr_auc-1),
      sprintf("XGBoost              (AUC = %.3f | Gini = %.3f)", xgb_auc, 2*xgb_auc-1)
    )
  ) +
  labs(
    title    = "ROC Curves — Logistic Regression vs. XGBoost (5-Fold CV)",
    subtitle = "Diagonal = random classifier | Top-left = perfect | AUC = area under each curve",
    x = "False Positive Rate (1 − Specificity)",
    y = "True Positive Rate (Sensitivity / Recall)",
    color = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")


# ── 11. KS Chart ──────────────────────────────────────────────────────────────
#
#  THE KOLMOGOROV-SMIRNOV (KS) STATISTIC
#  ────────────────────────────────────────
#  The KS statistic is the standard performance measure used by banks and
#  credit bureaus alongside the Gini coefficient.
#
#  It is computed by:
#    1. Scoring all applicants with the model (PD scores)
#    2. Sorting by score (lowest PD → "best" to highest PD → "worst")
#    3. Plotting the cumulative % of BAD applicants captured (TPR)
#       and the cumulative % of GOOD applicants captured (FPR)
#       as we move down the ranked list
#    4. KS = maximum vertical gap between the two cumulative curves
#
#  Interpretation:
#    At the KS point, the model achieves its best separation between the
#    good and bad populations. This is often used as the operational
#    threshold — all applicants above this score are approved, below are declined.
#
#  KS > 0.20 : acceptable
#  KS > 0.40 : good
#  KS > 0.60 : excellent (rarely achieved in practice)

# Compute KS from the logistic regression ROC curve
# KS = max(TPR - FPR) across all thresholds
lr_ks_df <- lr_roc |>
  mutate(ks = sensitivity - (1 - specificity))

lr_ks       <- max(lr_ks_df$ks, na.rm = TRUE)
lr_ks_point <- lr_ks_df |> slice_max(ks, n = 1)

xgb_ks_df  <- xgb_roc |>
  mutate(ks = sensitivity - (1 - specificity))
xgb_ks      <- max(xgb_ks_df$ks, na.rm = TRUE)

cat(sprintf("\nKS Statistic — Logistic Regression: %.4f\n", lr_ks))
cat(sprintf("KS Statistic — XGBoost:             %.4f\n", xgb_ks))

# KS Chart: cumulative good vs bad distributions
ggplot(lr_ks_df, aes(x = 1 - specificity)) +
  geom_line(aes(y = sensitivity,         color = "Cumulative Bad (TPR)"),  linewidth = 1.1) +
  geom_line(aes(y = 1 - specificity,     color = "Cumulative Good (FPR)"), linewidth = 1.1) +
  geom_segment(
    data = lr_ks_point,
    aes(x = 1 - specificity, xend = 1 - specificity,
        y = 1 - specificity, yend = sensitivity),
    color = "#27AE60", linewidth = 1.2, linetype = "dashed"
  ) +
  annotate("text",
           x = lr_ks_point$`1 - specificity` + 0.03,
           y = (lr_ks_point$sensitivity + lr_ks_point$`1 - specificity`) / 2,
           label = sprintf("KS = %.3f", lr_ks),
           color = "#27AE60", size = 4.5, hjust = 0) +
  scale_color_manual(values = c("Cumulative Bad (TPR)"  = "#E05C5C",
                                "Cumulative Good (FPR)" = "#4E9AF1")) +
  labs(
    title    = "KS Chart — Logistic Regression (5-Fold CV)",
    subtitle = "KS = maximum vertical gap between cumulative bad and good distributions",
    x = "Cumulative Fraction of Applicants (sorted by PD score)",
    y = "Cumulative Fraction", color = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")


# ── 12. Predicted Probability Distributions ───────────────────────────────────
#
#  This plot shows the distribution of PD scores (predicted default probabilities)
#  separately for good and bad borrowers.
#
#  A well-discriminating model produces:
#    - Good borrowers: scores concentrated near 0 (low PD)
#    - Bad borrowers: scores concentrated near 1 (high PD)
#
#  Overlap between the distributions = the "grey zone" where the model is
#  uncertain. Applicants in this zone are the hardest to classify — they may
#  require manual underwriting review rather than automatic approval/decline.
#
#  The cutpoint between the distributions corresponds to the KS statistic.

p_lr_scores <- ggplot(lr_preds, aes(x = .pred_bad, fill = Status, color = Status)) +
  geom_density(alpha = 0.45, adjust = 1.2) +
  scale_x_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_fill_manual(values  = c(good = "#4E9AF1", bad = "#E05C5C")) +
  scale_color_manual(values = c(good = "#4E9AF1", bad = "#E05C5C")) +
  labs(title    = "Logistic Regression: PD Score Distribution",
       subtitle = "Less overlap = better separation = higher Gini / KS",
       x = "Predicted Probability of Default", y = "Density",
       fill = "Status", color = "Status") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

p_xgb_scores <- ggplot(xgb_preds, aes(x = .pred_bad, fill = Status, color = Status)) +
  geom_density(alpha = 0.45, adjust = 1.2) +
  scale_x_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_fill_manual(values  = c(good = "#4E9AF1", bad = "#E05C5C")) +
  scale_color_manual(values = c(good = "#4E9AF1", bad = "#E05C5C")) +
  labs(title    = "XGBoost: PD Score Distribution",
       subtitle = "Compare separation to logistic regression",
       x = "Predicted Probability of Default", y = "Density",
       fill = "Status", color = "Status") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

print(p_lr_scores | p_xgb_scores)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION D — FINAL EVALUATION ON TEST SET
#
#  We select logistic regression as our primary model for final evaluation.
#  In practice, the model choice balances performance vs. interpretability.
#  Even if XGBoost performs better statistically, the regulatory requirement
#  for explainability often mandates logistic regression (or requires
#  additional work to explain the XGBoost model via SHAP values).
#
#  We evaluate BOTH models on the test set for completeness.
# ══════════════════════════════════════════════════════════════════════════════

# ── 13. Final Fits ────────────────────────────────────────────────────────────

final_lr  <- last_fit(lr_wf_final, split, metrics = metrics,
                       control = control_last_fit(event_level = "second"))

final_xgb <- last_fit(xgb_wf, split, metrics = metrics,
                       control = control_last_fit(event_level = "second"))

cat("\n── Final Test Set: Logistic Regression ──\n")
print(collect_metrics(final_lr))

cat("\n── Final Test Set: XGBoost ──\n")
print(collect_metrics(final_xgb))

# Final Gini comparison
final_lr_auc  <- collect_metrics(final_lr)  |> filter(.metric == "roc_auc") |> pull(.estimate)
final_xgb_auc <- collect_metrics(final_xgb) |> filter(.metric == "roc_auc") |> pull(.estimate)

cat(sprintf(
  "\n── Final Test Gini Coefficients ──\nLogistic Regression: %.4f\nXGBoost:             %.4f\n",
  2 * final_lr_auc - 1, 2 * final_xgb_auc - 1
))


# ── 14. Logistic Regression Coefficient Plot ──────────────────────────────────
#
#  One of the most valuable outputs of a logistic regression credit model is
#  the coefficient table. Each coefficient tells us how a variable influences
#  default risk, holding all other variables constant.
#
#  We extract coefficients from the final fitted logistic regression,
#  compute ODDS RATIOS (exp(β)), and plot them with confidence intervals.
#
#  HOW TO READ AN ODDS RATIO:
#    OR > 1  : associated with HIGHER default risk (a 1-unit increase in this
#              variable raises the odds of default by a factor of OR)
#    OR < 1  : associated with LOWER default risk (protective factor)
#    OR = 1  : no association (coefficient = 0)
#
#  Because we standardized numeric predictors (step_normalize), coefficients
#  for numeric variables are comparable: larger |OR − 1| = stronger effect.
#
#  For dummy-encoded categorical variables, the OR is relative to the
#  reference category (the one that was dropped during dummy encoding).

final_lr_wf <- extract_workflow(final_lr)

lr_coefs <- final_lr_wf |>
  extract_fit_parsnip() |>
  tidy() |>
  filter(term != "(Intercept)") |>
  mutate(
    odds_ratio = exp(estimate),
    direction  = ifelse(estimate > 0, "Increases default risk", "Reduces default risk"),
    term_clean = stringr::str_replace_all(term, "_X.", ": ") |>
                 stringr::str_replace_all("_", " ")
  )
#
#  NOTE: glmnet does not return standard errors for individual coefficients
#  (the L2 penalty changes the sampling distribution of estimates). Confidence
#  intervals are therefore omitted. For standard errors, use unpenalised
#  logistic regression (set_engine("glm")) or bootstrap resampling.

# Plot top 20 most influential variables
lr_coefs |>
  slice_max(abs(estimate), n = 20) |>
  ggplot(aes(x = odds_ratio, y = reorder(term_clean, estimate), color = direction)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_point(size = 3.5) +
  scale_color_manual(values = c("Increases default risk" = "#E05C5C",
                                "Reduces default risk"   = "#4E9AF1")) +
  scale_x_log10() +
  labs(
    title    = "Logistic Regression — Odds Ratios (Top 20 Variables)",
    subtitle = "OR > 1: increases default risk | OR < 1: reduces risk | Log scale",
    x = "Odds Ratio (log scale)", y = NULL, color = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")


# ── 15. XGBoost Feature Importance ────────────────────────────────────────────
#
#  For XGBoost, we report Gain-based feature importance (see fraud detection
#  script for a full explanation of gain vs. cover vs. frequency).
#  Compare the top features here to the significant coefficients in the
#  logistic regression — consistency suggests the signals are robust.

final_xgb_wf  <- extract_workflow(final_xgb)
xgb_engine    <- extract_fit_engine(final_xgb_wf)

xgboost::xgb.importance(model = xgb_engine) |>
  dplyr::as_tibble() |>
  dplyr::slice_max(Gain, n = 15) |>
  ggplot(aes(x = Gain, y = reorder(Feature, Gain))) +
  geom_col(fill = "#E05C5C") +
  labs(
    title    = "XGBoost — Top 15 Feature Importances (Gain)",
    subtitle = "Compare to logistic regression odds ratios — consistent rankings = robust signal",
    x = "Importance (Gain)", y = NULL
  ) +
  theme_minimal(base_size = 13)


# ── 16. Confusion Matrix & Classification Metrics ─────────────────────────────
#
#  In credit risk, the confusion matrix translates to real business outcomes:
#
#    TP (True Positive):  predicted bad, actually bad → loan correctly declined
#    TN (True Negative):  predicted good, actually good → loan correctly approved
#    FP (False Positive): predicted bad, actually good → profitable loan rejected
#                         (a missed revenue opportunity — Type I error in lending)
#    FN (False Negative): predicted good, actually bad → loan approved to defaulter
#                         (a direct financial loss — Type II error in lending)
#
#  THE ASYMMETRIC COST PROBLEM
#  ────────────────────────────
#  FN (missed default) typically costs more than FP (rejected good loan):
#    Cost(FN) = LGD × EAD = loss on the bad loan
#    Cost(FP) = lost profit margin on the rejected loan
#  This asymmetry argues for a threshold below 0.5 — see Section 17.

test_preds_lr <- collect_predictions(final_lr) |>
  mutate(.pred_class = factor(ifelse(.pred_bad >= 0.5, "bad", "good"),
                              levels = c("good", "bad")))

conf_lr <- conf_mat(test_preds_lr, truth = Status, estimate = .pred_class)

p_conf <- autoplot(conf_lr, type = "heatmap") +
  scale_fill_gradient(low = "#DDEEFF", high = "#1A5FA8") +
  labs(
    title    = "Confusion Matrix — Logistic Regression, Test Set (Threshold = 0.50)",
    subtitle = "Rows = Actual | Columns = Predicted"
  ) +
  theme_minimal(base_size = 13)

print(p_conf)

cat("\n── Classification Metrics at Threshold = 0.5 ──\n")
summary(conf_lr, event_level = "second") |>
  filter(.metric %in% c("accuracy", "precision", "recall", "f_meas", "specificity", "npv")) |>
  mutate(.estimate = round(.estimate, 4)) |>
  select(.metric, .estimate) |>
  print()


# ── 17. Threshold Analysis ────────────────────────────────────────────────────
#
#  SETTING THE CREDIT SCORE CUTOFF
#  ─────────────────────────────────
#  In credit scoring, the approval threshold is called the CUTOFF SCORE.
#  All applicants above the cutoff are approved; below are declined.
#
#  Unlike fraud detection where we might want to catch as many positives as
#  possible, credit decisions require balancing:
#    - APPROVAL RATE    : what fraction of applications are approved?
#    - BAD RATE         : of approved loans, what fraction default?
#    - PROFIT           : revenue from interest minus losses from defaults
#
#  Banks set cutoff scores based on:
#    (a) Target bad rate in the approved book (e.g., "no more than 5% of
#        approved loans should default")
#    (b) Target approval rate (e.g., "we want to approve at least 70% of
#        applicants to meet business volume targets")
#    (c) Regulatory capital: higher risk approved loans require more capital
#
#  The threshold analysis plot helps visualise these tradeoffs explicitly.

threshold_df <- purrr::map_dfr(seq(0.05, 0.95, by = 0.01), function(thresh) {
  preds_t <- test_preds_lr |>
    mutate(.pred_t = factor(ifelse(.pred_bad >= thresh, "bad", "good"),
                            levels = c("good", "bad")))
  tp <- sum(preds_t$Status == "bad"  & preds_t$.pred_t == "bad")
  fp <- sum(preds_t$Status == "good" & preds_t$.pred_t == "bad")
  fn <- sum(preds_t$Status == "bad"  & preds_t$.pred_t == "good")
  tn <- sum(preds_t$Status == "good" & preds_t$.pred_t == "good")

  total   <- tp + fp + fn + tn
  prec    <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)   # bad rate in declined
  rec     <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)   # recall of defaults
  f1      <- if (is.na(prec) | is.na(rec) | (prec + rec) == 0) NA_real_ else 2 * prec * rec / (prec + rec)
  app_rate <- (tn + fp) / total   # approval rate = predicted good / total

  tibble::tibble(.threshold = thresh, precision = prec, recall = rec,
                 f_meas = f1, approval_rate = app_rate)
})

# Plot 1: Precision/Recall/F1 vs threshold
p_thresh1 <- threshold_df |>
  tidyr::pivot_longer(c(precision, recall, f_meas), names_to = ".metric", values_to = ".estimate") |>
  ggplot(aes(x = .threshold, y = .estimate, color = .metric)) +
  geom_line(linewidth = 1, na.rm = TRUE) +
  scale_x_continuous(labels = scales::percent) +
  scale_color_manual(
    values = c(precision = "#E05C5C", recall = "#4E9AF1", f_meas = "#27AE60"),
    labels = c(precision = "Precision (bad rate in declined)",
               recall    = "Recall (% of defaults caught)",
               f_meas    = "F1 Score")
  ) +
  labs(title = "Classification Metrics by Threshold", x = "Cutoff Score", y = NULL, color = NULL) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

# Plot 2: Approval rate vs threshold
p_thresh2 <- ggplot(threshold_df, aes(x = .threshold, y = approval_rate)) +
  geom_line(color = "#4E9AF1", linewidth = 1) +
  scale_x_continuous(labels = scales::percent) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Approval Rate by Threshold",
       subtitle = "Higher threshold = more approvals, more accepted risk",
       x = "Cutoff Score", y = "Approval Rate") +
  theme_minimal(base_size = 12)

print(p_thresh1 / p_thresh2)

# Best F1 cutoff
best_t <- threshold_df |> slice_max(f_meas, n = 1, na_rm = TRUE)
cat(sprintf("\nBest F1 cutoff: %.2f | F1: %.4f | Approval rate: %.1f%%\n",
            best_t$.threshold, best_t$f_meas, best_t$approval_rate * 100))


# ── 18. Rating Grade Migration (Score Bands) ──────────────────────────────────
#
#  CREDIT SCORE BANDS (RATING GRADES)
#  ────────────────────────────────────
#  In practice, banks rarely use a single cutoff score. Instead, they create
#  RATING GRADES — bands of scores that correspond to risk tiers:
#    AAA / 1 : very low risk (e.g., PD < 1%)
#    AA  / 2 : low risk
#    A   / 3 : acceptable risk
#    BBB / 4 : moderate risk
#    BB  / 5 : elevated risk
#    B   / 6 : high risk → decline
#
#  We create 5 score bands from the logistic regression PD scores and show
#  the actual default rate within each band. A well-calibrated model shows
#  monotonically increasing default rates as scores worsen.
#
#  CALIBRATION is the property that the model's predicted PD equals the
#  actual observed default rate. For example, if the model predicts PD = 0.15
#  for a group of applicants, approximately 15% of them should default.

test_preds_lr |>
  mutate(
    score_band = cut(
      .pred_bad,
      breaks = c(0, 0.10, 0.20, 0.35, 0.50, 1.01),
      labels = c("Grade 1\n(PD < 10%)", "Grade 2\n(10–20%)", "Grade 3\n(20–35%)",
                 "Grade 4\n(35–50%)", "Grade 5\n(PD > 50%)"),
      include.lowest = TRUE
    )
  ) |>
  group_by(score_band) |>
  summarise(
    n            = n(),
    n_bad        = sum(Status == "bad"),
    actual_dr    = mean(Status == "bad"),
    mean_pred_pd = mean(.pred_bad),
    .groups = "drop"
  ) |>
  ggplot(aes(x = score_band)) +
  geom_col(aes(y = actual_dr),    fill = "#E05C5C",  alpha = 0.7, width = 0.4,
           position = position_nudge(x = -0.22)) +
  geom_col(aes(y = mean_pred_pd), fill = "#4E9AF1",  alpha = 0.7, width = 0.4,
           position = position_nudge(x =  0.22)) +
  geom_text(aes(y = actual_dr + 0.01, label = paste0(n, " apps")),
            position = position_nudge(x = -0.22), size = 3.2, vjust = 0) +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title    = "Rating Grade Calibration — Logistic Regression (Test Set)",
    subtitle = "Red = actual default rate | Blue = model predicted PD | Good calibration: bars are similar height",
    x = "Score Band (Rating Grade)", y = "Default Rate / Predicted PD"
  ) +
  theme_minimal(base_size = 12)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION E — SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════

# ── 19. Save the Final Logistic Regression Workflow ───────────────────────────
#
#  We save the logistic regression as the primary model (interpretable,
#  regulatory-compliant) and optionally the XGBoost as a challenger model.
#
#  The workflow includes the full recipe (imputation, dummy encoding,
#  normalization) — so new applicants are preprocessed automatically.
#
#  WHAT THE SAVED MODEL NEEDS AT PREDICTION TIME:
#  A data frame with columns: Seniority, Home, Time, Age, Marital, Records,
#  Job, Expenses, Income, Assets, Debt, Amount, Price
#  (Time the column in the dataset, not the system time — it's loan duration)

library(bundle)

# Save logistic regression (primary model)
lr_bundle <- extract_workflow(final_lr) |> bundle()
saveRDS(lr_bundle, "credit_risk_lr_model.rds")
cat("Logistic regression model saved: credit_risk_lr_model.rds\n")

# Save XGBoost (challenger model)
xgb_bundle <- extract_workflow(final_xgb) |> bundle()
saveRDS(xgb_bundle, "credit_risk_xgb_model.rds")
cat("XGBoost challenger model saved:  credit_risk_xgb_model.rds\n")

# ── To score a new applicant ──────────────────────────────────────────────────
#
# lr_model <- readRDS("credit_risk_lr_model.rds") |> unbundle()
#
# new_applicant <- tibble::tibble(
#   Seniority = 5,       # years of job seniority
#   Home      = "owner", # home ownership type
#   Time      = 36,      # loan term in months
#   Age       = 35,      # applicant age
#   Marital   = "married",
#   Records   = "no",    # no court records
#   Job       = "fixed", # fixed employment
#   Expenses  = 1200,    # monthly expenses
#   Income    = 4500,    # monthly income
#   Assets    = 25000,   # total assets
#   Debt      = 2000,    # existing debt
#   Amount    = 8000,    # loan amount requested
#   Price     = 10000    # price of item being purchased
# )
#
# pd_score <- predict(lr_model, new_data = new_applicant, type = "prob")
# cat("Predicted probability of default:", scales::percent(pd_score$.pred_bad), "\n")
#
# # Apply cutoff:
# decision <- ifelse(pd_score$.pred_bad < 0.35, "APPROVE", "DECLINE")
# cat("Decision:", decision, "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  KNOWN LIMITATIONS & EXTENSIONS
# ══════════════════════════════════════════════════════════════════════════════
#
#  1. THROUGH-THE-CYCLE vs. POINT-IN-TIME PD
#     PD models can be calibrated to a long-run average ("Through-the-Cycle",
#     TTC) or to current economic conditions ("Point-in-Time", PIT).
#     TTC is used for regulatory capital; PIT for IFRS 9 provisioning.
#     This distinction requires macroeconomic data not present here.
#
#  2. POPULATION STABILITY INDEX (PSI)
#     In production, the score distribution on new applicants is compared
#     to the training population using PSI = Σ (actual% - expected%) × ln(actual%/expected%).
#     PSI > 0.25 indicates significant population shift → model should be retrained.
#
#  3. WEIGHT OF EVIDENCE (WoE) TRANSFORMATION
#     The classical credit scorecard approach transforms each predictor to WoE:
#       WoE_i = ln(Distribution_Good_i / Distribution_Bad_i)
#     This linearises the relationship with log-odds, handles non-linearities
#     in continuous variables, and produces a points-based scorecard that
#     loan officers can compute by hand. Requires the {scorecard} or {woe} package.
#
#  4. VINTAGE ANALYSIS
#     In lending, performance varies by origination cohort (loan vintage).
#     Models should be validated on multiple cohorts rather than a single
#     random holdout to ensure temporal stability.
#
#  5. REJECT INFERENCE
#     The training data only contains decisions on applicants who were
#     actually approved (we don't know what would have happened to rejected
#     applicants). This creates a sample selection bias. Reject inference
#     techniques attempt to correct for this.
#
#  6. IFRS 9 / EXPECTED CREDIT LOSS
#     Modern accounting standards require banks to provision for Expected
#     Credit Loss (ECL) from loan inception. ECL = PD × LGD × EAD.
#     This requires models for all three components, not just PD.
#
# ══════════════════════════════════════════════════════════════════════════════
