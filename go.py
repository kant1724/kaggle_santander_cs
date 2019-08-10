import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np

train_df = pd.read_csv('./train.csv')
y = train_df['TARGET']
x = train_df.drop('TARGET', axis=1)

test_df = pd.read_csv('./test.csv')

clf = LGBMClassifier(
    nthread=4,
    n_estimators=10000,
    learning_rate=0.02,
    num_leaves=34,
    colsample_bytree=0.9497036,
    subsample=0.8715623,
    max_depth=8,
    reg_alpha=0.041545473,
    reg_lambda=0.0735294,
    min_split_gain=0.0222415,
    min_child_weight=39.3259775,
    silent=-1,
    verbose=-1, )

num_folds = 10
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
sub_preds = np.zeros(test_df.shape[0])

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(x, y)):
    train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
    valid_x, valid_y = x.iloc[valid_idx], y.iloc[valid_idx]

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
        eval_metric='auc', verbose=200, early_stopping_rounds=200)

    sub_preds += clf.predict_proba(test_df, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits


output = pd.DataFrame({'ID': test_df.ID, 'TARGET': sub_preds})
output.to_csv('submission.csv', index=False)

