# %%
# import packages
import pandas as pd
import joblib
import time
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from glum import GeneralizedLinearRegressor, NormalDistribution
from lightgbm import LGBMRegressor
from yelp_predict.data import create_sample_split
from yelp_predict.visualization import plot_numeric_hist
from yelp_predict.model import feature_engineering_pipeline
from yelp_predict.evaluation import (
    compute_metrics,
    pred_vs_actual,
    permutation_importance_top,
    plot_PDP,
    compare_metrics_by_rating_level,
    compare_metrics_by_cat,
    learning_curve_plot,
)


# %%
# set path and load cleaned dataset
data_path = Path(__file__).parent.parent / "data" / "cleaned_data.parquet"
figure_path = Path(__file__).parent.parent / "figures" / "evaluation"
model_path = Path(__file__).parent.parent / "configs"
dataset = pd.read_parquet(data_path)

# create train/test split (fraction 8:2)
dataset = create_sample_split(dataset, "business_id")
y = dataset["avg_rating"]
df_train = dataset[dataset["sample"] == "train"].reset_index(drop=True)
df_test = dataset[dataset["sample"] == "test"].reset_index(drop=True)
y_train = df_train["avg_rating"]
y_test = df_test["avg_rating"]

# define columns of dummy and categorical variables
dummies = dataset.iloc[
    :, 31:-1
].columns.to_list()  # need to create residual category
categories = dataset.iloc[:, 5:24].columns.to_list()  # need one-hot encoding


# %%
# define the pipeline for GLM
GLM_feature_engineeerning = feature_engineering_pipeline(
    dummies, categories, "GLM"
)
GLM_pipeline = Pipeline(
    [
        ("feature_engineering", GLM_feature_engineeerning),
        (
            "GLM_model",
            GeneralizedLinearRegressor(
                # family=BinomialDistribution(),
                family=NormalDistribution(),
                scale_predictors=True,
                fit_intercept=True,
            ),
        ),
    ]
)

# define the pipeline for LGBM
LGBM_feature_engineeerning = feature_engineering_pipeline(
    dummies, categories, "LGBM"
)
LGBM_pipeline = Pipeline(
    [
        ("feature_engineering", LGBM_feature_engineeerning),
        (
            "LGBM_model",
            LGBMRegressor(
                # objective = "cross_entropy"
                objective="mse",
                min_child_samples=50,
            ),
        ),
    ]
)


# %%
# Hyper-parameter tuning for GLM using grid-search
GLM_param_grid = {
    "GLM_model__alpha": [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    "GLM_model__l1_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
}
GLM_cv = GridSearchCV(GLM_pipeline, GLM_param_grid, cv=5)
GLM_cv.fit(df_train, y_train)

# Hyper-parameter tuning for LGBM using grid-search
LGBM_param_grid = {
    "LGBM_model__n_estimators": [100, 200, 500],
    "LGBM_model__learning_rate": [0.01, 0.02, 0.05, 0.1],
    "LGBM_model__num_leaves": [6, 15, 31],
    "LGBM_model__max_depth": [-1, 3, 5],
}
LGBM_cv = GridSearchCV(LGBM_pipeline, LGBM_param_grid, cv=5)
LGBM_cv.fit(df_train, y_train)


# %%
# save the best model pipelines
GLM_path = model_path / f"GLM_{int(time.time())}.joblib"
joblib.dump(GLM_cv.best_estimator_, GLM_path)
LGBM_path = model_path / f"LGBM_{int(time.time())}.joblib"
joblib.dump(LGBM_cv.best_estimator_, LGBM_path)


# %%
# evaluate prediction of GLM with finte-tuned hyper-parameters
GLM_cv.best_estimator_.fit(df_train, y_train)
df_train["predict_GLM"] = GLM_cv.best_estimator_.predict(df_train)
df_test["predict_GLM"] = GLM_cv.best_estimator_.predict(df_test)
print(
    compute_metrics(df_test["predict_GLM"], df_test["avg_rating"], model="GLM")
)

# evaluate prediction of LGBM with finte-tuned hyper-parameters
LGBM_cv.best_estimator_.fit(df_train, y_train)
df_train["predict_LGBM"] = LGBM_cv.best_estimator_.predict(df_train)
df_test["predict_LGBM"] = LGBM_cv.best_estimator_.predict(df_test)
print(
    compute_metrics(
        df_test["predict_LGBM"], df_test["avg_rating"], model="LGBM"
    )
)

# evaluate prediction of mean imputation (using mean from train set)
df_test["predict_mean"] = df_train["avg_rating"].mean()
print(
    compute_metrics(
        df_test["predict_mean"], df_test["avg_rating"], model="mean_impute"
    )
)


# %%
# plot the distribution of predicted avg_rating and residual by GLM in the train set
df_train["predict_GLM_residual"] = (
    df_train["avg_rating"] - df_train["predict_GLM"]
)
plot_numeric_hist(
    df_train,
    "predict_GLM",
    title="Distribution of Prediction By GLM in Train Set",
)
plot_numeric_hist(
    df_train,
    "predict_GLM_residual",
    title="Distribution of Residual By GLM in Train Set",
)

# plot the distribution of predicted avg_rating and residual by GLM in the test set
df_test["predict_GLM_residual"] = (
    df_test["avg_rating"] - df_test["predict_GLM"]
)
plot_numeric_hist(
    df_test,
    "predict_GLM",
    title="Distribution of Prediction By GLM in Test Set",
    save_path=figure_path,
)
plot_numeric_hist(
    df_test,
    "predict_GLM_residual",
    title="Distribution of Residual By GLM in Test Set",
)

# plot the distribution of predicted avg_rating and residual by LGBM in the train set
df_train["predict_LGBM_residual"] = (
    df_train["avg_rating"] - df_train["predict_LGBM"]
)
plot_numeric_hist(
    df_train,
    "predict_LGBM",
    title="Distribution of Prediction By LGBM in Train Set",
)
plot_numeric_hist(
    df_train,
    "predict_LGBM_residual",
    title="Distribution of Residual By LGBM in Train Set",
)

# plot the distribution of predicted avg_rating and residual by LGBM in the test set
df_test["predict_LGBM_residual"] = (
    df_test["avg_rating"] - df_test["predict_LGBM"]
)
plot_numeric_hist(
    df_test,
    "predict_LGBM",
    title="Distribution of Prediction By LGBM in Test Set",
    save_path=figure_path,
)
plot_numeric_hist(
    df_test,
    "predict_LGBM_residual",
    title="Distribution of Residual By LGBM in Test Set",
)


# %%
# Plot prediction v.s. actual value for GLM and LGBM
pred_vs_actual(
    "predict_GLM", "avg_rating", df_test, "GLM", save_path=figure_path
)
pred_vs_actual(
    "predict_LGBM", "avg_rating", df_test, "LGBM", save_path=figure_path
)


# %%
# compute feature relevance through permutation importance
GLM_importance = permutation_importance_top(GLM_cv, df_test, y_test)
print(GLM_importance.head(10))
LGBM_importance = permutation_importance_top(LGBM_cv, df_test, y_test)
print(LGBM_importance.head(10))


# %%
# plot PDP for top 5 most important features from LGBM for both model
top_5_features = LGBM_importance["col"].head(5).to_list()
plot_PDP(
    top_5_features, GLM_cv.best_estimator_, "GLM", df_test, y_test, figure_path
)
plot_PDP(
    top_5_features,
    LGBM_cv.best_estimator_,
    "LGBM",
    df_test,
    y_test,
    figure_path,
)


# %%
# compute accuracy metrics by different avg_rating quantile range
compare_metrics_by_rating_level(df_test, 0, 0.25)
compare_metrics_by_rating_level(df_test, 0.25, 0.5)
compare_metrics_by_rating_level(df_test, 0.5, 0.75)
compare_metrics_by_rating_level(df_test, 0.75, 1)


# %%
# compute accuracy metrics for top 10 categories
for cat in dummies[:20]:
    compare_metrics_by_cat(df_test, cat)

# %%
# compute accuracy metrics for rare categories
for cat in dummies[90:100]:
    compare_metrics_by_cat(df_test, cat)


# %%
# plot the learning curve of GLM
learning_curve_plot(
    GLM_cv.best_estimator_, "GLM", dataset, y, save_path=figure_path
)
learning_curve_plot(
    LGBM_cv.best_estimator_, "LGBM", dataset, y, save_path=figure_path
)
