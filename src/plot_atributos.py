"""Gera uma figura por atributo da base tratada, comparando as classes do alvo."""

from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from modelagem import INPUT_CSV, NOMES_CLASSES, carregar_dados


OUTPUT_DIR = INPUT_CSV.parent.parent / "outputs" / "atributos"
CORES = ("#23658B", "#AF5711", "#704799")
TITULOS = {
    "HighBP": "Hipertensão arterial",
    "HighChol": "Colesterol alto",
    "BMI": "Índice de massa corporal (IMC)",
    "GenHlth": "Saúde geral autorreferida",
    "HeartDiseaseorAttack": "Doença coronariana ou infarto",
    "Stroke": "Histórico de AVC",
    "DiffWalk": "Dificuldade para caminhar",
    "PhysActivity": "Atividade física",
    "Smoker": "Fumou pelo menos 100 cigarros na vida",
    "Sex": "Sexo registrado",
    "Age": "Faixa etária",
    "Education": "Escolaridade",
    "Income": "Faixa de renda",
    "MentHlth": "Dias de saúde mental ruim",
    "PhysHlth": "Dias de saúde física ruim",
    "HasHealthcare": "Cobertura de saúde",
    "NoDocbcCost": "Deixou de consultar por causa do custo",
    "CholCheck": "Exame de colesterol nos últimos cinco anos",
}
ROTULOS = {
    "Sex": ("Masculino", "Feminino"),
    "GenHlth": ("Excelente", "Muito boa", "Boa", "Regular", "Ruim"),
    "Age": ("18–24", *(f"{a}–{a+4}" for a in range(25, 80, 5)), "80+"),
}


def plotar(atributo, valores, alvo):
    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    grupos = [valores[alvo == classe] for classe in range(len(NOMES_CLASSES))]
    for grupo, nome, cor in zip(grupos, NOMES_CLASSES, CORES):
        ax.plot([], [], color=cor, linewidth=5, label=f"{nome} (n={len(grupo):,})".replace(",", "."))

    if atributo in {"BMI", "MentHlth", "PhysHlth"}:
        limites = np.arange(-0.5, 31, 1) if atributo != "BMI" else np.arange(12, 72, 2)
        for grupo, cor in zip(grupos, CORES):
            ax.hist(grupo, bins=limites, weights=np.ones(len(grupo)) / len(grupo),
                    histtype="step", linewidth=2, color=cor)
        ax.set_xlabel("IMC (kg/m²)" if atributo == "BMI" else "Dias nos últimos 30 dias")
    else:
        categorias = sorted(valores.unique())
        posicoes = np.arange(len(categorias))
        largura = 0.24
        for i, (grupo, cor) in enumerate(zip(grupos, CORES)):
            proporcoes = grupo.value_counts(normalize=True).reindex(categorias, fill_value=0)
            ax.bar(posicoes + (i - 1) * largura, proporcoes, width=largura, color=cor)
        if atributo in ROTULOS:
            inicio = 0 if atributo == "Sex" else 1
            rotulos = [ROTULOS[atributo][valor - inicio] for valor in categorias]
        elif set(categorias) <= {0, 1}:
            rotulos = ["Sim" if valor == 1 else "Não" for valor in categorias]
        else:
            rotulos = [f"Faixa {valor}" for valor in categorias]
        ax.set_xticks(posicoes, rotulos)
        ax.set_ylim(0, 1.05)

    ax.set_title(fill(TITULOS[atributo], width=60), loc="left", fontsize=17, pad=16)
    ax.set_ylabel("Proporção dentro da classe")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", color="#dae1e8")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=3, frameon=False)
    fig.savefig(OUTPUT_DIR / f"{atributo}.png", dpi=150)
    plt.close(fig)


def main():
    X, alvo, atributos = carregar_dados(INPUT_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for atributo in atributos:
        plotar(atributo, X[atributo], alvo)
    print(f"{len(atributos)} gráficos salvos em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
