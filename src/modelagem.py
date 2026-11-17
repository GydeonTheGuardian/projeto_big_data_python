"""
=============================================================================
PROJETO BIG DATA - ETAPA DE MODELAGEM PREDITIVA MULTICLASSE
=============================================================================
Dataset: data/dados_tratados_multiclasse.csv
Variável-Alvo: Target_Diabetes (0 = Não Diabético, 1 = Pré-diabético, 2 = Diabético)

Objetivo:
  Construir, treinar e avaliar modelos de Machine Learning multiclasse
  para o rastreamento epidemiológico de diabetes, tratando de forma estrita
  o severo desbalanceamento de classes através de:
    - Amostragem estratificada (stratify=y)
    - Oversampling no treino de cada fold na busca dos modelos obrigatórios
    - Ponderação na execução de referência com --sem-grid-search
    - Métricas especializadas para classes raras (Balanced Accuracy, F1 Macro, Matriz de Confusão)

Modelos Implementados:
  1. DecisionTreeClassifier
  2. RandomForestClassifier
  3. AdaBoostClassifier
  Opcional: LogisticRegression com StandardScaler
=============================================================================
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# Suporte à codificação UTF-8 no console do Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configurações de exibição do Pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

# Definição segura dos caminhos do projeto via pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_CSV = DATA_DIR / "dados_tratados_multiclasse.csv"

NOMES_CLASSES = ["Não Diabético", "Pré-diabético", "Diabético"]
MAPA_CLASSES = {0: "Não Diabético", 1: "Pré-diabético", 2: "Diabético"}


def carregar_dados(caminho_csv: Path) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Carrega o dataset tratado e separa as variáveis preditoras (X) do alvo (y).

    Args:
        caminho_csv (Path): Caminho para o arquivo CSV tratado.

    Returns:
        Tuple[pd.DataFrame, pd.Series, List[str]]:
            - X: DataFrame com as variáveis preditoras disponíveis no CSV tratado
            - y: Series com o alvo multiclasse (0, 1, 2)
            - features: Lista com os nomes dos atributos
    """
    print("=" * 80)
    print("1. CARREGAMENTO DOS DADOS TRATADOS")
    print("=" * 80)
    print(f"-> Localizando arquivo: {caminho_csv.resolve()}")

    if not caminho_csv.exists():
        raise FileNotFoundError(
            f"Erro crítico: Arquivo tratado não encontrado em '{caminho_csv}'.\n"
            f"Execute previamente o script 'src/preprocessamento.py' para gerar a base."
        )

    df = pd.read_csv(caminho_csv)
    print(f"-> Base carregada com sucesso: {df.shape[0]:,} instâncias x {df.shape[1]} colunas.")

    colunas_excluir = ["Status_Diabetes", "Target_Diabetes"]
    features = [c for c in df.columns if c not in colunas_excluir]

    X = df[features].copy()
    y = df["Target_Diabetes"].astype(int).copy()

    print(f"• Total de variáveis preditoras (features): {len(features)}")
    print(f"• Atributos selecionados: {', '.join(features)}\n")
    return X, y, features


def dividir_dados(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Realiza a divisão treino/teste com estratificação estrita por classe.

    Args:
        X (pd.DataFrame): Matriz de preditoras.
        y (pd.Series): Vetor da variável-alvo.
        test_size (float): Proporção do conjunto de teste (default: 0.20).
        seed (int): Semente de aleatoriedade para reprodutibilidade.

    Returns:
        Tuple com X_train, X_test, y_train, y_test.
    """
    print("=" * 80)
    print("2. DIVISÃO ESTRATIFICADA DE TREINO E TESTE (stratify=y)")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    total_treino = len(y_train)
    total_teste = len(y_test)
    pct_treino = (total_treino / len(y)) * 100
    pct_teste = (total_teste / len(y)) * 100

    print(f"• Proporção da divisão: {pct_treino:.0f}% Treino ({total_treino:,} registros) | {pct_teste:.0f}% Teste ({total_teste:,} registros)")
    print(f"• Semente de aleatoriedade: {seed}\n")

    # Auditoria de estratificação (garantia de preservação das proporções)
    dist_treino = y_train.value_counts(normalize=True).reindex([0, 1, 2]) * 100
    dist_teste = y_test.value_counts(normalize=True).reindex([0, 1, 2]) * 100

    tabela_estratificacao = pd.DataFrame({
        "Classe": NOMES_CLASSES,
        "Treino (Qtd)": [f"{y_train.value_counts()[i]:,}" for i in [0, 1, 2]],
        "Treino (%)": [f"{dist_treino[i]:.2f}%" for i in [0, 1, 2]],
        "Teste (Qtd)": [f"{y_test.value_counts()[i]:,}" for i in [0, 1, 2]],
        "Teste (%)": [f"{dist_teste[i]:.2f}%" for i in [0, 1, 2]],
    })
    print("• Auditoria da Estratificação por Conjunto:")
    print(tabela_estratificacao.to_string(index=False))
    print()
    return X_train, X_test, y_train, y_test


def treinar_arvore_decisao(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42
) -> Tuple[DecisionTreeClassifier, float]:
    """Treina uma árvore CART com profundidade limitada e pesos por classe."""
    print("-> Treinando DecisionTreeClassifier...")
    t0 = time.time()
    arvore = DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=seed
    )
    arvore.fit(X_train, y_train)
    return arvore, time.time() - t0


def treinar_adaboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42
) -> Tuple[AdaBoostClassifier, float]:
    """Treina stumps em sequência com pesos iniciais balanceados por classe."""
    print("-> Treinando AdaBoostClassifier...")
    t0 = time.time()
    # O estimador padrão do AdaBoost é uma árvore de profundidade 1.
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=seed)
    pesos = compute_sample_weight(class_weight="balanced", y=y_train)
    ada.fit(X_train, y_train, sample_weight=pesos)
    return ada, time.time() - t0


def treinar_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42
) -> Tuple[RandomForestClassifier, float]:
    """
    Treina o modelo RandomForestClassifier com ponderação de classes.

    Args:
        X_train (pd.DataFrame): Dados de treino.
        y_train (pd.Series): Rótulos de treino.
        seed (int): Semente aleatória.

    Returns:
        Tuple[RandomForestClassifier, float]: Modelo treinado e tempo de treino em segundos.
    """
    print("-" * 80)
    print("-> Treinando RandomForestClassifier (class_weight='balanced')...")
    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    tempo = time.time() - t0
    print(f"   Treinamento concluído em {tempo:.2f} segundos!")
    return rf, tempo


def metade_da_classe_majoritaria(y: np.ndarray) -> Dict[int, int]:
    """Eleva as classes abaixo da metade da frequência majoritária."""
    contagens = pd.Series(y).value_counts()
    minimo = int(contagens.max()) // 2
    return {classe: minimo for classe, quantidade in contagens.items() if quantidade < minimo}


def otimizar_modelo(chave, X_train, y_train, seed):
    """Seleciona por F1 macro e retreina no conjunto completo de treino."""
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline as PipelineReamostragem

    grades = {
        "dt": (
            DecisionTreeClassifier(random_state=seed, min_samples_split=20),
            {"max_depth": [4, 6, 12], "criterion": ["gini", "entropy"],
             "min_samples_leaf": [10, 50]},
        ),
        "rf": (
            RandomForestClassifier(random_state=seed, min_samples_split=20, n_jobs=-1),
            {"n_estimators": [100, 150], "max_depth": [8, 12, 18],
             "min_samples_leaf": [10, 30]},
        ),
        "ada": (
            AdaBoostClassifier(random_state=seed),
            {"n_estimators": [50, 100, 150], "learning_rate": [0.5, 1.0]},
        ),
    }
    estimador, grade = grades[chave]
    pipeline = PipelineReamostragem([
        ("oversampling", RandomOverSampler(
            sampling_strategy=metade_da_classe_majoritaria, random_state=seed
        )),
        ("modelo", estimador),
    ])
    busca = GridSearchCV(
        pipeline,
        {f"modelo__{parametro}": valores for parametro, valores in grade.items()},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        refit=True,
        error_score="raise",
    )
    print(f"-> Grid Search ({chave}): 3 folds estratificados, F1 macro, oversampling de 50%.")
    inicio = time.perf_counter()
    busca.fit(X_train, y_train)
    tempo = time.perf_counter() - inicio
    print(f"   Melhores hiperparâmetros: {busca.best_params_}")
    print(f"   F1 macro médio na validação: {busca.best_score_:.6f}")
    print(f"   Tempo de busca e retreino: {tempo:.2f} s")
    return busca.best_estimator_.named_steps["modelo"], busca.refit_time_


def treinar_regressao_logistica(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42
) -> Tuple[Pipeline, float]:
    """
    Treina o modelo LogisticRegression em um Pipeline com StandardScaler
    e ponderação de classes para convergência linear otimizada.

    Args:
        X_train (pd.DataFrame): Dados de treino.
        y_train (pd.Series): Rótulos de treino.
        seed (int): Semente aleatória.

    Returns:
        Tuple[Pipeline, float]: Pipeline treinado e tempo de treino em segundos.
    """
    print("-" * 80)
    print("-> Treinando LogisticRegression com StandardScaler (class_weight='balanced')...")
    t0 = time.time()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=seed
        ))
    ])
    pipeline.fit(X_train, y_train)
    tempo = time.time() - t0
    print(f"   Treinamento concluído em {tempo:.2f} segundos!")
    return pipeline, tempo


def formatar_matriz_confusao(y_true: pd.Series, y_pred: np.ndarray) -> str:
    """
    Gera uma representação textual tabular rica da matriz de confusão,
    contendo contagem absoluta e percentual normalizado por linha (Recall Real).
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true") * 100

    linhas = []
    header = f"{'Classe Real':<15} | " + " | ".join([f"Pred: {c[:8]:<8}" for c in NOMES_CLASSES]) + " | Total Real"
    divisor = "-" * len(header)
    linhas.append(divisor)
    linhas.append(header)
    linhas.append(divisor)

    for i, nome in enumerate(NOMES_CLASSES):
        vals = [f"{cm[i, j]:>6,} ({cm_norm[i, j]:4.1f}%)" for j in range(3)]
        linha_txt = f"{nome:<15} | " + " | ".join(vals) + f" | {cm[i].sum():>8,}"
        linhas.append(linha_txt)

    linhas.append(divisor)
    return "\n".join(linhas)


def avaliar_modelo(
    modelo: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    nome_modelo: str,
    tempo_treino: float,
    features: List[str]
) -> Dict[str, Any]:
    """
    Executa a avaliação completa do modelo no conjunto de teste,
    calculando e exibindo métricas voltadas a problemas desbalanceados.

    Args:
        modelo (Any): Modelo ou Pipeline scikit-learn treinado.
        X_test (pd.DataFrame): Dados de teste.
        y_test (pd.Series): Rótulos reais de teste.
        nome_modelo (str): Nome legível do modelo.
        tempo_treino (float): Tempo de treinamento em segundos.
        features (List[str]): Nomes dos atributos preditores.

    Returns:
        Dict[str, Any]: Dicionário com as métricas consolidadas.
    """
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    macro_recall = recall_score(y_test, y_pred, average="macro")
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 80)
    print(f"AVALIAÇÃO DO MODELO: {nome_modelo.upper()}")
    print("=" * 80)
    print(f"• Tempo de treinamento: {tempo_treino:.2f} s")
    print(f"• Acurácia Pura (Global):       {acc * 100:.2f}%  (Métrica enganosa sob desbalanceamento)")
    print(f"• Acurácia Balanceada:          {bal_acc * 100:.2f}%  (Média do Recall de cada classe)")
    print(f"• Macro F1-Score:               {macro_f1:.4f}   (Média harmônica não-ponderada)")
    print(f"• Weighted F1-Score:            {weighted_f1:.4f}")
    print(f"• Macro Recall (Sensibilidade): {macro_recall * 100:.2f}%")
    print(f"• Macro Precision:              {macro_prec * 100:.2f}%")
    print("-" * 80)

    # 1. Relatório detalhado por classe
    print("\n1. RELATÓRIO DE CLASSIFICAÇÃO DETALHADO POR CLASSE (Classification Report):")
    report_dict = classification_report(
        y_test, y_pred,
        target_names=NOMES_CLASSES,
        output_dict=True,
        digits=4,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).T
    print(report_df.to_string())
    print("-" * 80)

    # 2. Matriz de Confusão detalhada
    print("\n2. MATRIZ DE CONFUSÃO (Contagens e Recall Normalizado por Linha):")
    print(formatar_matriz_confusao(y_test, y_pred))
    print("-" * 80)

    # 3. Importância de Features (se aplicável ao modelo)
    if hasattr(modelo, "feature_importances_"):
        print("\n3. IMPORTÂNCIA DAS VARIÁVEIS PREDITORAS (redução de impureza):")
        fi_series = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=False)
        fi_df = pd.DataFrame({
            "Atributo": fi_series.index,
            "Importância Relativa (%)": (fi_series.values * 100).round(2),
            "Impacto Cumulativo (%)": (fi_series.cumsum().values * 100).round(2)
        })
        print(fi_df.to_string(index=False))
        print("-" * 80)
    elif hasattr(modelo, "named_steps") and hasattr(modelo.named_steps.get("clf"), "coef_"):
        print("\n3. COEFICIENTES DA REGRESSÃO LOGÍSTICA MULTICLASSE (Pesos por Atributo):")
        coefs = modelo.named_steps["clf"].coef_
        coef_df = pd.DataFrame(coefs, index=NOMES_CLASSES, columns=features).T
        print(coef_df.to_string())
        print("-" * 80)

    return {
        "Modelo": nome_modelo,
        "Tempo (s)": tempo_treino,
        "Acurácia Global (%)": acc * 100,
        "Acurácia Balanceada (%)": bal_acc * 100,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Recall Pré-diabético (%)": report_dict["Pré-diabético"]["recall"] * 100,
        "Recall Diabético (%)": report_dict["Diabético"]["recall"] * 100,
        "Recall Não Diabético (%)": report_dict["Não Diabético"]["recall"] * 100,
    }


def comparar_modelos(resultados: List[Dict[str, Any]]) -> None:
    """
    Exibe uma tabela comparativa final de desempenho entre os modelos avaliados.
    """
    print("\n" + "=" * 80)
    print("QUADRO COMPARATIVO CONSOLIDADO DE MODELOS MULTICLASSE")
    print("=" * 80)
    tabela = pd.DataFrame(resultados)
    print(tabela.to_string(index=False))
    print("=" * 80 + "\n")


def parse_args():
    """Configura argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Pipeline de Modelagem Preditiva Multiclasse - Projeto Big Data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_CSV,
        help="Caminho do CSV tratado (default: data/dados_tratados_multiclasse.csv)"
    )
    parser.add_argument(
        "--modelo",
        type=str,
        choices=["dt", "rf", "ada", "obrigatorios", "lr", "ambos"],
        default="obrigatorios",
        help="dt: Decision Tree, rf: Random Forest, ada: AdaBoost, "
             "obrigatorios: os três (padrão), lr: regressão logística, ambos: rf e lr"
    )
    parser.add_argument(
        "--sem-grid-search",
        action="store_true",
        help="Executa os modelos com a configuração de referência e ponderação de classes."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proporção da partição de teste (default: 0.20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória para reprodutibilidade (default: 42)"
    )
    return parser.parse_args()


def main():
    """Fluxo principal de orquestração do treinamento e avaliação."""
    args = parse_args()

    print("\n" + "#" * 80)
    print("   INÍCIO DO PIPELINE DE MODELAGEM PREDITIVA MULTICLASSE - BIG DATA")
    print("#" * 80 + "\n")

    # 1. Carregamento dos dados tratados
    X, y, features = carregar_dados(args.input)

    # 2. Particionamento Estratificado de Treino e Teste
    X_train, X_test, y_train, y_test = dividir_dados(
        X, y,
        test_size=args.test_size,
        seed=args.seed
    )

    modelos = {
        "dt": ("Decision Tree Classifier", treinar_arvore_decisao),
        "rf": ("Random Forest Classifier", treinar_random_forest),
        "ada": ("AdaBoost Classifier", treinar_adaboost),
        "lr": ("Logistic Regression", treinar_regressao_logistica),
    }
    grupos = {"obrigatorios": ["dt", "rf", "ada"], "ambos": ["rf", "lr"]}
    resultados = []
    for chave in grupos.get(args.modelo, [args.modelo]):
        nome, treinar = modelos[chave]
        if args.sem_grid_search or chave == "lr" or args.modelo == "ambos":
            modelo, tempo = treinar(X_train, y_train, seed=args.seed)
        else:
            modelo, tempo = otimizar_modelo(chave, X_train, y_train, args.seed)
        resultados.append(avaliar_modelo(modelo, X_test, y_test, nome, tempo, features))

    # 5. Quadro Comparativo Consolidado
    if len(resultados) > 1:
        comparar_modelos(resultados)

    print("#" * 80)
    print("   PIPELINE DE MODELAGEM PREDITIVA CONCLUÍDO COM SUCESSO!")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
