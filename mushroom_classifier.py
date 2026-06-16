"""
Sistema Inteligente — Classificação de Cogumelos (venenoso vs. comestível)

TAREFA: CLASSIFICAÇÃO (supervisionada). O dataset tem a coluna-alvo
`mushroom_type` rotulada ('p' = venenoso, 'e' = comestível) e o objetivo é
prever uma categoria para um cogumelo novo — não clusterização.

stalk-root incompleta: '?' é tratado como categoria própria ("missing"),
pois a ausência pode ser informativa e descartar a coluna/linhas perderia
~30% dos dados. A imputação pela moda é avaliada como comparação.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42

df = pd.read_csv("/mnt/user-data/uploads/mushroom.csv", sep=";")
print("Dimensões do dataset:", df.shape)
print("\nDistribuição da classe-alvo:")
print(df["mushroom_type"].value_counts())
print("  e = edible (comestível)  |  p = poisonous (venenoso)")

n_missing = (df["stalk-root"] == "?").sum()
print(f"\n[stalk-root] valores ausentes ('?'): {n_missing} "
      f"({100*n_missing/len(df):.1f}% das linhas)")
df["stalk-root"] = df["stalk-root"].replace("?", "missing")
print("Estratégia: '?' tratado como categoria 'missing'.")

y = (df["mushroom_type"] == "p").astype(int)   # 1 = venenoso, 0 = comestível
X = df.drop(columns=["mushroom_type"])

const_cols = [c for c in X.columns if X[c].nunique() <= 1]
if const_cols:
    print("Removendo colunas constantes:", const_cols)
    X = X.drop(columns=const_cols)
cat_cols = X.columns.tolist()

preprocess = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
)
tree_clf = Pipeline(steps=[
    ("prep", preprocess),
    ("model", DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
tree_clf.fit(X_train, y_train)
y_pred = tree_clf.predict(X_test)

print("\n" + "=" * 60)
print("RESULTADOS — Árvore de Decisão (max_depth=8)")
print("=" * 60)
print(f"Acurácia (teste): {accuracy_score(y_test, y_pred):.4f}")
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred,
      target_names=["comestível (e)", "venenoso (p)"]))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(tree_clf, X, y, cv=cv, scoring="accuracy")
print(f"Acurácia média (CV 5-fold): {cv_scores.mean():.4f} "
      f"(+/- {cv_scores.std():.4f})")

cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusão [linhas=real, colunas=previsto]:")
print("              prev_comestível  prev_venenoso")
print(f"real_comestível    {cm[0,0]:>8}        {cm[0,1]:>6}")
print(f"real_venenoso      {cm[1,0]:>8}        {cm[1,1]:>6}")
fn = cm[1, 0]  # venenoso previsto como comestível: erro crítico
print(f"\n*** Falsos negativos (venenoso previsto como comestível): {fn} ***")

df_imp = pd.read_csv("/mnt/user-data/uploads/mushroom.csv", sep=";")
mode_val = df_imp.loc[df_imp["stalk-root"] != "?", "stalk-root"].mode()[0]
df_imp["stalk-root"] = df_imp["stalk-root"].replace("?", mode_val)
X2 = df_imp.drop(columns=["mushroom_type"] + const_cols)
y2 = (df_imp["mushroom_type"] == "p").astype(int)
cv_scores_imp = cross_val_score(tree_clf, X2, y2, cv=cv, scoring="accuracy")
print("\n[Comparação stalk-root]")
print(f"  '?' como categoria 'missing' -> CV acc: {cv_scores.mean():.4f}")
print(f"  '?' imputado pela moda ('{mode_val}') -> CV acc: {cv_scores_imp.mean():.4f}")

ohe = tree_clf.named_steps["prep"].named_transformers_["cat"]
feat_names = ohe.get_feature_names_out(cat_cols)
importances = tree_clf.named_steps["model"].feature_importances_
top = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(10)
print("\nTop 10 atributos mais decisivos:")
for name, imp in top.items():
    print(f"  {name:<40} {imp:.3f}")

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["comestível", "venenoso"]).plot(
    ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Matriz de Confusão — Árvore de Decisão")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/confusion_matrix.png", dpi=120)
print("\nGráfico salvo: confusion_matrix.png")
