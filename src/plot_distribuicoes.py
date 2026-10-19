"""
=============================================================================
PROJETO BIG DATA - VISUALIZAÇÃO E ANÁLISE EXPLORATÓRIA MULTICLASSE
=============================================================================
Dataset: CDC Behavioral Risk Factor Surveillance System (BRFSS)
Arquivo: Diabetes Health Indicators.csv

Escopo:
Classificação Multiclasse (3 categorias oficiais):
  - 'Não Diabético' (Código CDC: 3)
  - 'Pré-diabético' (Código CDC: 4)
  - 'Diabético'     (Códigos CDC: 1 e 2 - Diabético e Gestacional)

Fatores analisados em relação ao diagnóstico:
  1. Índice de Massa Corporal (BMI / IMC)
  2. Faixa Etária (_AGEG5YR / Age)
  3. Percepção do Estado Geral de Saúde (GENHLTH / GenHlth)

Saídas:
  Gráficos salvos em alta resolução (300 DPI) na pasta 'outputs/'
=============================================================================
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Backend não interativo para geração e salvamento direto
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Suporte à codificação UTF-8 no console do Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configuração de caminhos do projeto via pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Diabetes Health Indicators.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# Configurações visuais padronizadas (design acadêmico e corporativo moderno)
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10.5,
    "figure.titlesize": 15,
})

# Paleta de cores semântica para o problema multiclasse de 3 categorias:
# - Não Diabético: Azul sóbrio (#2b5c8f) -> Baseline / Saudável
# - Pré-diabético: Âmbar / Ouro (#e69f00) -> Alerta / Condição limítrofe
# - Diabético:     Carmesim (#c0392b)     -> Diagnóstico clínico confirmado
CORES_STATUS = {
    "Não Diabético": "#2b5c8f",
    "Pré-diabético": "#e69f00",
    "Diabético": "#c0392b"
}
CATEGORIAS_STATUS = ["Não Diabético", "Pré-diabético", "Diabético"]


def carregar_e_tratar_dados(caminho_csv: Path) -> pd.DataFrame:
    """
    Carrega o dataset e realiza o tratamento e engenharia de atributos:
    - DIABETE4 -> 'Status_Diabetes' (3: Não Diabético, 4: Pré-diabético, 1 e 2: Diabético; filtra 7, 9 e NaN)
    - _BMI5    -> 'BMI' (_BMI5 / 100 com duas casas decimais implícitas)
    - _AGEG5YR -> 'Faixa_Etaria' (13 faixas etárias ordenadas; filtra 14)
    - GENHLTH  -> 'Saude_Geral' (1: Excelente até 5: Ruim; filtra 7 e 9)
    """
    if not caminho_csv.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado em: {caminho_csv}")

    print(f"-> Carregando dados de: {caminho_csv}")
    df = pd.read_csv(caminho_csv)
    print(f"   Registros brutos carregados: {len(df):,}")

    # 1. Tratamento da Variável-Alvo Multiclasse: DIABETE4
    # Códigos CDC BRFSS:
    # 1 = Diabético | 2 = Gestacional (ambos agrupados em 'Diabético')
    # 3 = Não Diabético
    # 4 = Pré-diabético ou limítrofe
    # Descarte de códigos não informativos (7 = Não sabe, 9 = Recusou responder e NaN)
    df_valid = df[df["DIABETE4"].isin([1, 2, 3, 4])].copy()

    mapa_target = {
        3.0: "Não Diabético",
        4.0: "Pré-diabético",
        1.0: "Diabético",
        2.0: "Diabético"
    }
    df_valid["Status_Diabetes"] = df_valid["DIABETE4"].map(mapa_target)
    df_valid["Status_Diabetes"] = pd.Categorical(
        df_valid["Status_Diabetes"],
        categories=CATEGORIAS_STATUS,
        ordered=True
    )

    # 2. Tratamento do BMI: _BMI5
    # Documentação CDC BRFSS: _BMI5 possui 2 casas decimais implícitas (ex: 2847 = 28.47)
    df_valid["BMI"] = df_valid["_BMI5"] / 100.0

    # 3. Tratamento da Idade: _AGEG5YR (Faixas quinquenais)
    mapa_idades = {
        1: "18-24",
        2: "25-29",
        3: "30-34",
        4: "35-39",
        5: "40-44",
        6: "45-49",
        7: "50-54",
        8: "55-59",
        9: "60-64",
        10: "65-69",
        11: "70-74",
        12: "75-79",
        13: "80+"
    }
    # Filtra 14 (Don't know/Refused/Missing)
    df_valid = df_valid[df_valid["_AGEG5YR"].isin(mapa_idades.keys())].copy()
    df_valid["Faixa_Etaria"] = df_valid["_AGEG5YR"].map(mapa_idades)
    df_valid["Faixa_Etaria"] = pd.Categorical(
        df_valid["Faixa_Etaria"],
        categories=list(mapa_idades.values()),
        ordered=True
    )

    # 4. Tratamento do Estado Geral de Saúde: GENHLTH
    mapa_saude = {
        1: "Excelente",
        2: "Muito Boa",
        3: "Boa",
        4: "Regular",
        5: "Ruim"
    }
    # Filtra 7 e 9 (Don't know/Refused)
    df_valid = df_valid[df_valid["GENHLTH"].isin(mapa_saude.keys())].copy()
    df_valid["Saude_Geral"] = df_valid["GENHLTH"].map(mapa_saude)
    df_valid["Saude_Geral"] = pd.Categorical(
        df_valid["Saude_Geral"],
        categories=list(mapa_saude.values()),
        ordered=True
    )

    total_validos = len(df_valid)
    print(f"   Registros tratados e válidos: {total_validos:,} ({(total_validos / len(df)) * 100:.2f}% do original)")
    for cat in CATEGORIAS_STATUS:
        qtd = (df_valid["Status_Diabetes"] == cat).sum()
        pct = (qtd / total_validos) * 100
        print(f"   - {cat:<15}: {qtd:>7,} ({pct:5.2f}%)")
    print()

    return df_valid


def plot_distribuicao_bmi(df: pd.DataFrame, pasta_saida: Path) -> None:
    """
    Gera e salva visualizações da distribuição de BMI (IMC) para as 3 classes:
    - Gráfico 1: Curvas de Densidade Estimada (KDE) com linhas de corte da OMS e medianas de cada classe
    - Gráfico 2: Boxplot comparativo com indicação de quartis, medianas e médias para as 3 classes
    """
    df_bmi = df.dropna(subset=["BMI"]).copy()
    
    # Filtro de corte para valores extremos severos (> 70) para visualização focada
    df_plot = df_bmi[df_bmi["BMI"] <= 70]

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    # --- Subplot 1: Curvas KDE Multiclasse ---
    sns.kdeplot(
        data=df_plot,
        x="BMI",
        hue="Status_Diabetes",
        palette=CORES_STATUS,
        common_norm=False,
        fill=True,
        alpha=0.30,
        linewidth=2,
        ax=axes[0]
    )

    # Coleta handles e rótulos originais do KDE
    kde_leg = axes[0].get_legend()
    kde_handles = list(kde_leg.legend_handles) if kde_leg else []
    kde_labels = [t.get_text() for t in kde_leg.get_texts()] if kde_leg else []

    # Linhas de referência de corte da OMS
    axes[0].axvline(25.0, color="#7f8c8d", linestyle="--", linewidth=1.2, label="Sobrepeso (≥ 25)")
    axes[0].axvline(30.0, color="#2c3e50", linestyle=":", linewidth=1.5, label="Obesidade (≥ 30)")

    # Cálculo das medianas dos 3 grupos
    med_nao = df_bmi[df_bmi["Status_Diabetes"] == "Não Diabético"]["BMI"].median()
    med_pre = df_bmi[df_bmi["Status_Diabetes"] == "Pré-diabético"]["BMI"].median()
    med_dia = df_bmi[df_bmi["Status_Diabetes"] == "Diabético"]["BMI"].median()

    axes[0].axvline(med_nao, color=CORES_STATUS["Não Diabético"], linestyle="-.", linewidth=1.4,
                    label=f"Med. Não Diabético ({med_nao:.1f})")
    axes[0].axvline(med_pre, color=CORES_STATUS["Pré-diabético"], linestyle="-.", linewidth=1.4,
                    label=f"Med. Pré-diabético ({med_pre:.1f})")
    axes[0].axvline(med_dia, color=CORES_STATUS["Diabético"], linestyle="-.", linewidth=1.4,
                    label=f"Med. Diabético ({med_dia:.1f})")

    line_handles, line_labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles=kde_handles + line_handles,
        labels=kde_labels + line_labels,
        loc="upper right",
        frameon=True,
        fontsize=8.5
    )

    axes[0].set_title("Distribuição de Densidade do IMC (BMI) por Classe", fontweight="bold")
    axes[0].set_xlabel("Índice de Massa Corporal (kg/m²)")
    axes[0].set_ylabel("Densidade Estimada")
    axes[0].set_xlim(15, 60)

    # --- Subplot 2: Boxplot Multiclasse ---
    sns.boxplot(
        data=df_plot,
        x="Status_Diabetes",
        y="BMI",
        hue="Status_Diabetes",
        palette=CORES_STATUS,
        order=CATEGORIAS_STATUS,
        width=0.45,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": 7},
        fliersize=2,
        flierprops={"alpha": 0.25},
        legend=False,
        ax=axes[1]
    )

    # Cálculo das médias dos 3 grupos
    media_nao = df_bmi[df_bmi["Status_Diabetes"] == "Não Diabético"]["BMI"].mean()
    media_pre = df_bmi[df_bmi["Status_Diabetes"] == "Pré-diabético"]["BMI"].mean()
    media_dia = df_bmi[df_bmi["Status_Diabetes"] == "Diabético"]["BMI"].mean()

    texto_estatisticas = (
        f"Não Diabético:\n"
        f"  • Mediana: {med_nao:.1f} kg/m² | Média: {media_nao:.1f} kg/m²\n\n"
        f"Pré-diabético:\n"
        f"  • Mediana: {med_pre:.1f} kg/m² | Média: {media_pre:.1f} kg/m²\n\n"
        f"Diabético:\n"
        f"  • Mediana: {med_dia:.1f} kg/m² | Média: {media_dia:.1f} kg/m²\n\n"
        f"(♦ Diamante amarelo indica a média)"
    )
    axes[1].text(
        0.03, 0.96, texto_estatisticas,
        transform=axes[1].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        fontsize=9
    )

    axes[1].set_title("Boxplot do IMC por Diagnóstico Multiclasse", fontweight="bold")
    axes[1].set_xlabel("Categoria de Diabetes")
    axes[1].set_ylabel("Índice de Massa Corporal (kg/m²)")
    axes[1].set_ylim(15, 60)

    fig.suptitle("Relação entre Índice de Massa Corporal (BMI) e Categorias de Diabetes", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    arquivo_saida = pasta_saida / "01_distribuicao_bmi.png"
    fig.savefig(arquivo_saida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Gráfico salvo: {arquivo_saida}")


def plot_distribuicao_idade(df: pd.DataFrame, pasta_saida: Path) -> None:
    """
    Gera e salva visualizações da distribuição de Faixa Etária por status de diabetes:
    - Gráfico 1: Composição etária dentro de cada uma das 3 classes (%)
    - Gráfico 2: Taxa de prevalência (%) de Pré-diabetes e Diabetes por faixa etária
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.2))

    # --- Subplot 1: Composição Etária Dentro de Cada Categoria (%) ---
    tabela_cruzada = pd.crosstab(
        df["Faixa_Etaria"],
        df["Status_Diabetes"],
        normalize="columns"
    ) * 100

    tabela_plot = tabela_cruzada.reset_index().melt(
        id_vars="Faixa_Etaria",
        value_name="Percentual",
        var_name="Status_Diabetes"
    )

    sns.barplot(
        data=tabela_plot,
        x="Faixa_Etaria",
        y="Percentual",
        hue="Status_Diabetes",
        palette=CORES_STATUS,
        hue_order=CATEGORIAS_STATUS,
        ax=axes[0]
    )
    axes[0].set_title("Composição Etária Dentro de Cada Categoria (%)", fontweight="bold")
    axes[0].set_xlabel("Faixa Etária (anos)")
    axes[0].set_ylabel("Proporção Dentro da Classe (%)")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(title="Status de Diabetes", frameon=True)

    # --- Subplot 2: Taxa de Prevalência por Faixa Etária (%) ---
    # Foco epidemiológico nas categorias clínicas de interesse (Pré-diabético e Diabético)
    ct_idade = pd.crosstab(df["Faixa_Etaria"], df["Status_Diabetes"], normalize="index") * 100
    prev_idade = ct_idade[["Pré-diabético", "Diabético"]].reset_index().melt(
        id_vars="Faixa_Etaria",
        value_name="Prevalencia_Pct",
        var_name="Status_Diabetes"
    )

    barras = sns.barplot(
        data=prev_idade,
        x="Faixa_Etaria",
        y="Prevalencia_Pct",
        hue="Status_Diabetes",
        palette=CORES_STATUS,
        hue_order=["Pré-diabético", "Diabético"],
        ax=axes[1]
    )

    # Anotação dos valores no topo das barras
    for p in barras.patches:
        altura = p.get_height()
        if altura > 0:
            axes[1].annotate(
                f"{altura:.1f}%",
                (p.get_x() + p.get_width() / 2.0, altura),
                ha="center", va="bottom",
                fontsize=8,
                fontweight="bold",
                xytext=(0, 2),
                textcoords="offset points"
            )

    axes[1].set_title("Taxa de Prevalência por Faixa Etária: Pré-diabetes vs Diabetes (%)", fontweight="bold")
    axes[1].set_xlabel("Faixa Etária (anos)")
    axes[1].set_ylabel("Prevalência na População da Faixa (%)")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylim(0, prev_idade["Prevalencia_Pct"].max() * 1.18)
    axes[1].legend(title="Condição Clínica", frameon=True)

    fig.suptitle("Relação entre Faixa Etária (Age) e Diagnóstico Multiclasse de Diabetes", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    arquivo_saida = pasta_saida / "02_distribuicao_idade.png"
    fig.savefig(arquivo_saida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Gráfico salvo: {arquivo_saida}")


def plot_distribuicao_genhlth(df: pd.DataFrame, pasta_saida: Path) -> None:
    """
    Gera e salva visualizações da relação entre Saúde Geral (GENHLTH) e Diabetes:
    - Gráfico 1: Distribuição da autoavaliação de saúde por categoria de diabetes
    - Gráfico 2: Taxa de prevalência de Pré-diabetes e Diabetes segundo a percepção de saúde
    """
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2))

    # --- Subplot 1: Distribuição Relativa de Saúde Geral ---
    tabela_saude = pd.crosstab(
        df["Saude_Geral"],
        df["Status_Diabetes"],
        normalize="columns"
    ) * 100

    tabela_plot = tabela_saude.reset_index().melt(
        id_vars="Saude_Geral",
        value_name="Percentual",
        var_name="Status_Diabetes"
    )

    sns.barplot(
        data=tabela_plot,
        x="Saude_Geral",
        y="Percentual",
        hue="Status_Diabetes",
        palette=CORES_STATUS,
        hue_order=CATEGORIAS_STATUS,
        ax=axes[0]
    )
    axes[0].set_title("Distribuição da Saúde Autoavaliada Dentro de Cada Classe (%)", fontweight="bold")
    axes[0].set_xlabel("Saúde Geral Autoavaliada")
    axes[0].set_ylabel("Proporção Dentro da Classe (%)")
    axes[0].legend(title="Status de Diabetes", frameon=True)

    # --- Subplot 2: Taxa de Prevalência por Nível de Saúde Geral ---
    ct_saude = pd.crosstab(df["Saude_Geral"], df["Status_Diabetes"], normalize="index") * 100
    prev_saude = ct_saude[["Pré-diabético", "Diabético"]].reset_index().melt(
        id_vars="Saude_Geral",
        value_name="Prevalencia_Pct",
        var_name="Status_Diabetes"
    )

    barras = sns.barplot(
        data=prev_saude,
        x="Saude_Geral",
        y="Prevalencia_Pct",
        hue="Status_Diabetes",
        palette=CORES_STATUS,
        hue_order=["Pré-diabético", "Diabético"],
        ax=axes[1]
    )

    for p in barras.patches:
        altura = p.get_height()
        if altura > 0:
            axes[1].annotate(
                f"{altura:.1f}%",
                (p.get_x() + p.get_width() / 2.0, altura),
                ha="center", va="bottom",
                fontsize=9.5,
                fontweight="bold",
                xytext=(0, 2),
                textcoords="offset points"
            )

    axes[1].set_title("Prevalência por Nível de Saúde Geral: Pré-diabetes vs Diabetes (%)", fontweight="bold")
    axes[1].set_xlabel("Saúde Geral Autoavaliada")
    axes[1].set_ylabel("Prevalência Dentro do Nível de Saúde (%)")
    axes[1].set_ylim(0, prev_saude["Prevalencia_Pct"].max() * 1.18)
    axes[1].legend(title="Condição Clínica", frameon=True)

    fig.suptitle("Relação entre Saúde Geral Percebida (GenHlth) e Categorias de Diabetes", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    arquivo_saida = pasta_saida / "03_distribuicao_saude_geral.png"
    fig.savefig(arquivo_saida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Gráfico salvo: {arquivo_saida}")


def plot_painel_consolidado(df: pd.DataFrame, pasta_saida: Path) -> None:
    """
    Gera um painel executivo multiclasse consolidando os 3 principais fatores
    (BMI, Faixa Etária e Saúde Geral) em uma única figura em alta resolução.
    """
    fig, axes = plt.subplots(1, 3, figsize=(23, 6.5))

    # --- 1. Densidade do IMC ---
    df_plot = df.dropna(subset=["BMI"])
    df_plot = df_plot[df_plot["BMI"] <= 70]

    sns.kdeplot(
        data=df_plot,
        x="BMI",
        hue="Status_Diabetes",
        palette=CORES_STATUS,
        common_norm=False,
        fill=True,
        alpha=0.30,
        linewidth=2,
        ax=axes[0]
    )
    kde_leg = axes[0].get_legend()
    kde_handles = list(kde_leg.legend_handles) if kde_leg else []
    kde_labels = [t.get_text() for t in kde_leg.get_texts()] if kde_leg else []

    axes[0].axvline(30.0, color="#2c3e50", linestyle=":", linewidth=1.5, label="Obesidade (≥ 30)")
    line_handles, line_labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles=kde_handles + line_handles,
        labels=kde_labels + line_labels,
        loc="upper right",
        frameon=True,
        fontsize=9
    )

    axes[0].set_title("(A) Distribuição do IMC (BMI) por Classe", fontweight="bold")
    axes[0].set_xlabel("Índice de Massa Corporal (kg/m²)")
    axes[0].set_ylabel("Densidade Estimada")
    axes[0].set_xlim(15, 60)

    # --- 2. Prevalência por Faixa Etária ---
    ct_idade = pd.crosstab(df["Faixa_Etaria"], df["Status_Diabetes"], normalize="index") * 100
    prev_idade = ct_idade[["Pré-diabético", "Diabético"]].reset_index().melt(
        id_vars="Faixa_Etaria",
        value_name="Prevalencia_Pct",
        var_name="Status_Diabetes"
    )

    barras_age = sns.barplot(
        data=prev_idade,
        x="Faixa_Etaria",
        y="Prevalencia_Pct",
        hue="Status_Diabetes",
        palette=CORES_STATUS,
        hue_order=["Pré-diabético", "Diabético"],
        ax=axes[1]
    )
    for p in barras_age.patches:
        altura = p.get_height()
        if altura > 0:
            axes[1].annotate(
                f"{altura:.1f}%",
                (p.get_x() + p.get_width() / 2.0, altura),
                ha="center", va="bottom",
                fontsize=7.5,
                fontweight="bold",
                xytext=(0, 2),
                textcoords="offset points"
            )
    axes[1].set_title("(B) Prevalência por Faixa Etária (Age)", fontweight="bold")
    axes[1].set_xlabel("Faixa Etária")
    axes[1].set_ylabel("Taxa de Prevalência (%)")
    axes[1].tick_params(axis="x", rotation=40)
    axes[1].set_ylim(0, prev_idade["Prevalencia_Pct"].max() * 1.18)
    axes[1].legend(title="Condição", frameon=True, fontsize=8.5)

    # --- 3. Prevalência por Saúde Geral ---
    ct_saude = pd.crosstab(df["Saude_Geral"], df["Status_Diabetes"], normalize="index") * 100
    prev_saude = ct_saude[["Pré-diabético", "Diabético"]].reset_index().melt(
        id_vars="Saude_Geral",
        value_name="Prevalencia_Pct",
        var_name="Status_Diabetes"
    )

    barras_saude = sns.barplot(
        data=prev_saude,
        x="Saude_Geral",
        y="Prevalencia_Pct",
        hue="Status_Diabetes",
        palette=CORES_STATUS,
        hue_order=["Pré-diabético", "Diabético"],
        ax=axes[2]
    )
    for p in barras_saude.patches:
        altura = p.get_height()
        if altura > 0:
            axes[2].annotate(
                f"{altura:.1f}%",
                (p.get_x() + p.get_width() / 2.0, altura),
                ha="center", va="bottom",
                fontsize=9,
                fontweight="bold",
                xytext=(0, 2),
                textcoords="offset points"
            )
    axes[2].set_title("(C) Prevalência por Saúde Geral (GenHlth)", fontweight="bold")
    axes[2].set_xlabel("Saúde Geral Autoavaliada")
    axes[2].set_ylabel("Taxa de Prevalência (%)")
    axes[2].set_ylim(0, prev_saude["Prevalencia_Pct"].max() * 1.18)
    axes[2].legend(title="Condição", frameon=True, fontsize=8.5)

    fig.suptitle(
        "Painel Analítico Multiclasse: Impacto de BMI, Faixa Etária e Saúde Geral no Diagnóstico de Diabetes",
        fontsize=16,
        fontweight="bold",
        y=1.02
    )
    plt.tight_layout()

    arquivo_saida = pasta_saida / "04_painel_resumo_diabetes.png"
    fig.savefig(arquivo_saida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Painel consolidado salvo: {arquivo_saida}")


def main():
    print("=" * 75)
    print("GERAÇÃO DE GRÁFICOS MULTICLASSE - DIABETES HEALTH INDICATORS (CDC BRFSS)")
    print("=" * 75)

    # Garante que o diretório outputs/ exista
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Diretório de saída: {OUTPUT_DIR}\n")

    # Carrega e trata o dataset
    df = carregar_e_tratar_dados(DATA_PATH)

    print("Iniciando geração dos gráficos multiclasse em alta resolução (300 DPI)...")
    # 1. Gráfico de BMI (KDE e Boxplot multiclasse)
    plot_distribuicao_bmi(df, OUTPUT_DIR)

    # 2. Gráfico de Idade (Composição e Prevalência multiclasse)
    plot_distribuicao_idade(df, OUTPUT_DIR)

    # 3. Gráfico de Saúde Geral (Composição e Prevalência multiclasse)
    plot_distribuicao_genhlth(df, OUTPUT_DIR)

    # 4. Painel Integrado / Dashboard Executivo
    plot_painel_consolidado(df, OUTPUT_DIR)

    print("\n" + "=" * 75)
    print("Todos os gráficos multiclasse foram gerados e salvos com sucesso na pasta 'outputs/'!")
    print("=" * 75)


if __name__ == "__main__":
    main()
