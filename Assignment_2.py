import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay as cmd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV


# ============================================================
# Choose different vectorization methods
# ============================================================

# (1) CountVectorizer
# vectorizer = CountVectorizer(
#     ngram_range=(1, 2),
#     stop_words='english',
#     min_df=20
# )


# (2) HashingVectorizer
# from sklearn.feature_extraction.text import HashingVectorizer
# vectorizer = HashingVectorizer(
#     ngram_range=(1, 2),
#     n_features=200
# )


# (3) TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    min_df=20,
    norm='l2',
    smooth_idf=True,
    use_idf=True,
    ngram_range=(1, 1),
    stop_words='english'
)


# ============================================================
# Split into train/test set
# ============================================================

url = 'https://raw.githubusercontent.com/zhenliangma/Applied-AI-in-Transportation/master/Exercise_4_Text_classification/Pakistani%20Traffic%20sentiment%20Analysis.csv'

df = pd.read_csv(url)

x = df['Text']
y = df['Sentiment']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=0
)


# ============================================================
# Apply the vectorizer
# ============================================================

x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)


# ============================================================
# Choose model and parameter grid
# ============================================================


#(1) Logistic Regression

# model = LogisticRegression(
#     max_iter=1000,
#     random_state=0
# )

# param_grid = {
#     'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]
# }


# ------------------------------------------------------------

# (2) KNN

# model = KNeighborsClassifier()

# param_grid = {
#     'n_neighbors': [7, 9, 11, 13, 15, 17],
#     'weights': ['distance']
# }


# ------------------------------------------------------------

# (3) Random Forest

model = RandomForestClassifier(
    random_state=0
)

param_grid = {
    'n_estimators': [250, 300, 350, 400],
    'max_depth': [20, 30, 40, None],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [1, 2]
}


# ------------------------------------------------------------

# (4) XGBoost

# model = XGBClassifier()

# param_grid = {
#     'learning_rate': [0.05, 0.075, 0.1, 0.125, 0.15],
#     'n_estimators': [150, 175, 200, 225, 250],
#     'max_depth': [3, 4, 5]
# }


# ------------------------------------------------------------

# (5) SVM

# model = SVC(
#     probability=True
# )

# param_grid = {
#     'kernel': ['rbf'],
#     'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]
# }


# ------------------------------------------------------------

# (6) Naive Bayes

# model = BernoulliNB()

# param_grid = {
#     'alpha': [0.5, 0.75, 1, 1.25, 1.5, 2],
#     'force_alpha': [True]
# }


# ============================================================
# Grid Search
# ============================================================

# Perform grid search using 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)


# Train all parameter combinations
grid_search.fit(
    x_train_vectorized,
    y_train
)

model = grid_search.best_estimator_

# ============================================================
# Best parameters and cross-validation score
# ============================================================

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best parameters:", best_params)
print("Best cross-validation accuracy:", best_score)


# Use the best model found by GridSearchCV
model = grid_search.best_estimator_


# ============================================================
# Confusion Matrix
# ============================================================

cmd.from_estimator(
    model,
    x_test_vectorized,
    y_test,
    display_labels=['Positive', 'Negative'],
    cmap='Blues',
    xticks_rotation='vertical'
)

plt.title("Confusion Matrix for Random Forest")
plt.tight_layout()
plt.show()

# ============================================================
# Test accuracy
# ============================================================

y_pred = model.predict(x_test_vectorized)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("The accuracy of the model is:", accuracy)