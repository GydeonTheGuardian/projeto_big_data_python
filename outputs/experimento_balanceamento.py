import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from modelagem import carregar_dados, dividir_dados, INPUT_CSV

X, y, _ = carregar_dados(INPUT_CSV)
X, _, y, _ = dividir_dados(X, y)
cats = X.columns.difference(["BMI", "MentHlth", "PhysHlth"]).tolist()
models = {
    "DT": DecisionTreeClassifier(max_depth=12,min_samples_split=20,min_samples_leaf=10,random_state=42),
    "RF": RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_split=20,min_samples_leaf=10,random_state=42,n_jobs=-1),
    "ADA": AdaBoostClassifier(n_estimators=50,learning_rate=1.0,random_state=42),
}
results = []
for fold, (tr, va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(X,y), 1):
    xt, yt, xv, yv = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]
    n = int((yt == 0).sum())
    half = {1: n // 2, 2: n // 2}
    strategies = {
        "Pesos": None,
        "Under 50%": RandomUnderSampler(sampling_strategy={0: n // 2},random_state=42),
        "Over 50%": RandomOverSampler(sampling_strategy=half,random_state=42),
        "Over 100%": RandomOverSampler(sampling_strategy={1:n,2:n},random_state=42),
        "SMOTENC 50%": SMOTENC(categorical_features=cats,sampling_strategy=half,random_state=42),
    }
    for strategy, sampler in strategies.items():
        a, b = (xt,yt) if sampler is None else sampler.fit_resample(xt,yt)
        for name, estimator in models.items():
            model = clone(estimator)
            options = {}
            if sampler is None:
                if name == "ADA":
                    options["sample_weight"] = compute_sample_weight("balanced",b)
                else:
                    model.set_params(class_weight="balanced")
            model.fit(a,b,**options)
            pred = model.predict(xv)
            results.append([fold,name,strategy,f1_score(yv,pred,average="macro"),
                            precision_score(yv==1,pred==1,zero_division=0),
                            recall_score(yv==1,pred==1)])
        print("Concluído",fold,strategy,flush=True)
df = pd.DataFrame(results, columns=["fold","modelo","estrategia","f1","prec","recall"])
print(df.groupby(["modelo","estrategia"])[["f1","prec","recall"]].agg(["mean","std"]).to_string())
