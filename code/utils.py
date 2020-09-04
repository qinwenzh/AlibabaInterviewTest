import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
import time

warnings.simplefilter(action='ignore', category=FutureWarning)


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.savefig('results/lgbm_importances01.png')

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def recorrect_columes_names(df):
    for feature_name in df.columns:
        if ('/' in feature_name or ':' in feature_name or ',' in feature_name or ' ' in feature_name):
            x = feature_name.replace('/', '_')
            x = x.replace(',', '_')
            x = x.replace(':', '_')
            x = x.replace(' ', '_')
            df = df.rename(columns={feature_name: x})

    return df