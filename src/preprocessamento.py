"""
=============================================================================
PROJETO BIG DATA - ITEM 8: PRÉ-PROCESSAMENTO E ENGENHARIA DE ATRIBUTOS
=============================================================================
Dataset: CDC Behavioral Risk Factor Surveillance System (BRFSS)
Arquivo de Entrada:  data/Diabetes Health Indicators.csv
Arquivo de Saída:    data/dados_tratados_multiclasse.csv

Objetivo:
  Estruturar um pipeline robusto e auditável de limpeza, padronização e
  engenharia de atributos para um problema de classificação MULTICLASSE
  em 3 categorias clínicas de Diabetes:
    - 0: 'Não Diabético' (Código original 3)
    - 1: 'Pré-diabético' (Código original 4)
    - 2: 'Diabético'     (Códigos originais 1 e 2 - Diabético e Gestacional)

Etapas Executadas:
  1. Carregamento seguro via pathlib a partir de data/
  2. Filtragem de ruídos e mapeamento estrito da variável-alvo (DIABETE4)
  3. Tratamento e padronização das variáveis preditoras clínicas e de estilo de vida:
     - BMI (_BMI5 / 100, expurgo de outliers biológicos e nulos)
     - HighBP (_RFHYPE6 -> 0: Não, 1: Sim)
     - HighChol (_RFCHOL3 -> 0: Não, 1: Sim)
     - GenHlth (GENHLTH -> Escala 1 a 5)
     - Age (_AGEG5YR -> Escala 1 a 13)
     - DiffWalk (DIFFWALK -> 0: Não, 1: Sim)
     - HeartDiseaseorAttack (_MICHD -> 0: Não, 1: Sim)
     - PhysActivity (EXERANY2 -> 0: Não, 1: Sim)
     - Smoker (SMOKE100 -> 0: Não, 1: Sim)
     - Stroke (CVDSTRK3 -> 0: Não, 1: Sim)
     - Sex (_SEX -> 0: Masculino, 1: Feminino)
     - Education (_EDUCAG -> Escala 1 a 4)
     - Income (_INCOMG1 -> Escala 1 a 7)
  4. Auditoria de integridade e relatório estatístico detalhado (evidenciando desbalanceamento)
  5. Exportação otimizada para novo CSV na pasta data/
=============================================================================
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

# Garante suporte a UTF-8 no console do Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configurações de exibição do Pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# Definição segura dos caminhos do projeto via pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_CSV = DATA_DIR / "Diabetes Health Indicators.csv"
OUTPUT_CSV = DATA_DIR / "dados_tratados_multiclasse.csv"

# Mapeamentos oficiais CDC BRFSS para variável-alvo
MAPA_TARGET_DESC = {
    3.0: "Não Diabético",
    4.0: "Pré-diabético",
    1.0: "Diabético",
    2.0: "Diabético"
}

MAPA_TARGET_NUM = {
    3.0: 0,
    4.0: 1,
    1.0: 2,
    2.0: 2
}

ORDEM_CLASSES = ["Não Diabético", "Pré-diabético", "Diabético"]


def carregar_dados(caminho_csv: Path) -> pd.DataFrame:
    """
    Carrega o dataset bruto com validação de caminho seguro pathlib.

    Args:
        caminho_csv (Path): Caminho absoluto/relativo do arquivo CSV.

    Returns:
        pd.DataFrame: Dataframe bruto carregado.
    """
    print("=" * 80)
    print("ETAPA 1: CARREGAMENTO DO DATASET BRUTO")
    print("=" * 80)
    print(f"-> Localizando arquivo em: {caminho_csv.resolve()}")

    if not caminho_csv.exists():
        raise FileNotFoundError(
            f"Erro crítico: O arquivo não foi encontrado em '{caminho_csv}'.\n"
            f"Verifique se 'Diabetes Health Indicators.csv' está na pasta 'data/'."
        )

    df = pd.read_csv(caminho_csv)
    print(f"-> Dataset carregado com sucesso!")
    print(f"   Dimensões brutas: {df.shape[0]:,} linhas x {df.shape[1]} colunas\n")
    return df


def processar_variavel_alvo(df: pd.DataFrame, coluna_alvo: str = "DIABETE4") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trata a variável-alvo multiclasse 'DIABETE4', eliminando ruídos
    (7: Don't know, 9: Refused e NaNs) e mapeando estritamente em 3 classes.

    Args:
        df (pd.DataFrame): Dataframe original.
        coluna_alvo (str): Nome da coluna da variável-alvo no BRFSS.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Dataframe filtrado e métricas da etapa.
    """
    print("=" * 80)
    print("ETAPA 2: PROCESSAMENTO DA VARIÁVEL-ALVO MULTICLASSE ('DIABETE4')")
    print("=" * 80)

    if coluna_alvo not in df.columns:
        raise KeyError(f"A coluna de destino '{coluna_alvo}' não existe no dataset.")

    total_inicial = len(df)

    # Identificação e auditoria de ruídos
    mascara_validos = df[coluna_alvo].isin([1.0, 2.0, 3.0, 4.0])
    total_ruidos = (~mascara_validos).sum()

    print(f"• Registros brutos avaliados: {total_inicial:,}")
    print(f"• Registros descartados (códigos 7, 9 ou NaN): {total_ruidos:,} ({(total_ruidos / total_inicial) * 100:.2f}%)")

    df_target = df[mascara_validos].copy()

    # Mapeamento descritivo e numérico
    df_target["Status_Diabetes"] = df_target[coluna_alvo].map(MAPA_TARGET_DESC)
    df_target["Status_Diabetes"] = pd.Categorical(
        df_target["Status_Diabetes"],
        categories=ORDEM_CLASSES,
        ordered=True
    )
    df_target["Target_Diabetes"] = df_target[coluna_alvo].map(MAPA_TARGET_NUM).astype(int)

    metricas_target = {
        "total_inicial": total_inicial,
        "total_validos": len(df_target),
        "total_descartados": total_ruidos
    }

    print(f"• Registros preservados após validação do alvo: {len(df_target):,} ({(len(df_target) / total_inicial) * 100:.2f}%)\n")
    return df_target, metricas_target


def processar_preditoras(
    df: pd.DataFrame,
    limite_bmi_min: float = 12.0,
    limite_bmi_max: float = 70.0,
    incluir_socioeconomicos: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Realiza o tratamento e engenharia de atributos das principais preditoras:
      - BMI: conversão de _BMI5 (/100) com corte de outliers biológicos.
      - HighBP, HighChol: binarização de indicadores CDC (1: Não -> 0, 2: Sim -> 1).
      - GenHlth: escala ordinal 1 a 5 (limpeza de 7, 9).
      - Age: escala ordinal 1 a 13 (limpeza de 14).
      - DiffWalk, HeartDiseaseorAttack, PhysActivity, Smoker, Stroke: binarizações estritas.
      - Sex, Education, Income: fatores demográficos e socioeconômicos.

    Args:
        df (pd.DataFrame): Dataframe com a variável-alvo tratada.
        limite_bmi_min (float): Limite inferior aceito para IMC biológico.
        limite_bmi_max (float): Limite superior aceito para IMC biológico.
        incluir_socioeconomicos (bool): Se inclui e limpa educação e renda.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Dataframe limpo final e tabela de funil de auditoria.
    """
    print("=" * 80)
    print("ETAPA 3: PROCESSAMENTO E LIMPEZA DAS VARIÁVEIS PREDITORAS")
    print("=" * 80)

    passos_funil = []
    linhas_correntes = len(df)
    passos_funil.append({"Etapa / Filtro": "0. Variável-Alvo Válida (DIABETE4)", "Registros Restantes": linhas_correntes, "Perda Absoluta": 0, "Retenção (%)": 100.0})

    df_clean = df.copy()

    # 1. Tratamento de BMI (_BMI5)
    # CDC BRFSS: _BMI5 possui 2 decimais implícitas. Ex.: 2750 -> 27.50
    df_clean["BMI"] = (df_clean["_BMI5"] / 100.0).round(2)
    mascara_bmi = df_clean["BMI"].notna() & (df_clean["BMI"] >= limite_bmi_min) & (df_clean["BMI"] <= limite_bmi_max)
    perda_bmi = linhas_correntes - mascara_bmi.sum()
    df_clean = df_clean[mascara_bmi].copy()
    linhas_correntes = len(df_clean)
    passos_funil.append({
        "Etapa / Filtro": f"1. BMI Válido ({limite_bmi_min:.0f} <= IMC <= {limite_bmi_max:.0f})",
        "Registros Restantes": linhas_correntes,
        "Perda Absoluta": perda_bmi,
        "Retenção (%)": (linhas_correntes / passos_funil[0]["Registros Restantes"]) * 100
    })

    # 2. HighBP (_RFHYPE6): 1 = Não (0), 2 = Sim (1); 9 = Missing
    mascara_highbp = df_clean["_RFHYPE6"].isin([1, 2])
    perda_highbp = linhas_correntes - mascara_highbp.sum()
    df_clean = df_clean[mascara_highbp].copy()
    df_clean["HighBP"] = (df_clean["_RFHYPE6"] == 2).astype(int)
    linhas_correntes = len(df_clean)
    passos_funil.append({
        "Etapa / Filtro": "2. HighBP (_RFHYPE6 in [1, 2])",
        "Registros Restantes": linhas_correntes,
        "Perda Absoluta": perda_highbp,
        "Retenção (%)": (linhas_correntes / passos_funil[0]["Registros Restantes"]) * 100
    })

    # 3. HighChol (_RFCHOL3): 1 = Não (0), 2 = Sim (1); 9/NaN = Missing
    mascara_highchol = df_clean["_RFCHOL3"].isin([1.0, 2.0])
    perda_highchol = linhas_correntes - mascara_highchol.sum()
    df_clean = df_clean[mascara_highchol].copy()
    df_clean["HighChol"] = (df_clean["_RFCHOL3"] == 2.0).astype(int)
    linhas_correntes = len(df_clean)
    passos_funil.append({
        "Etapa / Filtro": "3. HighChol (_RFCHOL3 in [1, 2])",
        "Registros Restantes": linhas_correntes,
        "Perda Absoluta": perda_highchol,
        "Retenção (%)": (linhas_correntes / passos_funil[0]["Registros Restantes"]) * 100
    })

    # 4. GenHlth (GENHLTH): 1 a 5 (Excelente a Ruim); 7/9 = Missing
    mascara_genhlth = df_clean["GENHLTH"].isin([1.0, 2.0, 3.0, 4.0, 5.0])
    perda_genhlth = linhas_correntes - mascara_genhlth.sum()
    df_clean = df_clean[mascara_genhlth].copy()
    df_clean["GenHlth"] = df_clean["GENHLTH"].astype(int)
    linhas_correntes = len(df_clean)
    passos_funil.append({
        "Etapa / Filtro": "4. GenHlth (GENHLTH in [1..5])",
        "Registros Restantes": linhas_correntes,
        "Perda Absoluta": perda_genhlth,
        "Retenção (%)": (linhas_correntes / passos_funil[0]["Registros Restantes"]) * 100
    })

    # 5. Faixa Etária (_AGEG5YR): 1 a 13 (18-24 até 80+); 14 = Missing
    mascara_age = df_clean["_AGEG5YR"].isin(list(range(1, 14)))
    perda_age = linhas_correntes - mascara_age.sum()
    df_clean = df_clean[mascara_age].copy()
    df_clean["Age"] = df_clean["_AGEG5YR"].astype(int)
    linhas_correntes = len(df_clean)
    passos_funil.append({
        "Etapa / Filtro": "5. Age (_AGEG5YR in [1..13])",
        "Registros Restantes": linhas_correntes,
        "Perda Absoluta": perda_age,
        "Retenção (%)": (linhas_correntes / passos_funil[0]["Registros Restantes"]) * 100
    })

    # 6. DiffWalk (DIFFWALK): 1 = Sim (1), 2 = Não (0); 7/9/NaN = Missing
    mascara_diffwalk = df_clean["DIFFWALK"].isin([1.0, 2.0])
    perda_diffwalk = linhas_correntes - mascara_diffwalk.sum()
    df_clean = df_clean[mascara_diffwalk].copy()
    df_clean["DiffWalk"] = (df_clean["DIFFWALK"] == 1.0).astype(int)
    linhas_correntes = len(df_clean)
    passos_funil.append({
        "Etapa / Filtro": "6. DiffWalk (DIFFWALK in [1, 2])",
        "Registros Restantes": linhas_correntes,
        "Perda Absoluta": perda_diffwalk,
        "Retenção (%)": (linhas_correntes / passos_funil[0]["Registros Restantes"]) * 100
    })

    # 7. HeartDiseaseorAttack (_MICHD): 1 = Sim (1), 2 = Não (0); NaN = Missing
    mascara_heart = df_clean["_MICHD"].isin([1.0, 2.0])
    perda_heart = linhas_correntes - mascara_heart.sum()
    df_clean = df_clean[mascara_heart].copy()
    df_clean["HeartDiseaseorAttack"] = (df_clean["_MICHD"] == 1.0).astype(int)
    linhas_correntes = len(df_clean)
    passos_funil.append({
        "Etapa / Filtro": "7. HeartDiseaseorAttack (_MICHD in [1, 2])",
        "Registros Restantes": linhas_correntes,
        "Perda Absoluta": perda_heart,
        "Retenção (%)": (linhas_correntes / passos_funil[0]["Registros Restantes"]) * 100
    })

    # 8. PhysActivity (EXERANY2): 1 = Sim (1), 2 = Não (0); 7/9 = Missing
    mascara_phys = df_clean["EXERANY2"].isin([1.0, 2.0])
    perda_phys = linhas_correntes - mascara_phys.sum()
    df_clean = df_clean[mascara_phys].copy()
    df_clean["PhysActivity"] = (df_clean["EXERANY2"] == 1.0).astype(int)
    linhas_correntes = len(df_clean)
    passos_funil.append({
        "Etapa / Filtro": "8. PhysActivity (EXERANY2 in [1, 2])",
        "Registros Restantes": linhas_correntes,
        "Perda Absoluta": perda_phys,
        "Retenção (%)": (linhas_correntes / passos_funil[0]["Registros Restantes"]) * 100
    })

    # 9. Smoker (SMOKE100): 1 = Sim (1), 2 = Não (0); 7/9/NaN = Missing
    mascara_smoker = df_clean["SMOKE100"].isin([1.0, 2.0])
    perda_smoker = linhas_correntes - mascara_smoker.sum()
    df_clean = df_clean[mascara_smoker].copy()
    df_clean["Smoker"] = (df_clean["SMOKE100"] == 1.0).astype(int)
    linhas_correntes = len(df_clean)
    passos_funil.append({
        "Etapa / Filtro": "9. Smoker (SMOKE100 in [1, 2])",
        "Registros Restantes": linhas_correntes,
        "Perda Absoluta": perda_smoker,
        "Retenção (%)": (linhas_correntes / passos_funil[0]["Registros Restantes"]) * 100
    })

    # 10. Stroke (CVDSTRK3): 1 = Sim (1), 2 = Não (0); 7/9 = Missing
    mascara_stroke = df_clean["CVDSTRK3"].isin([1.0, 2.0])
    perda_stroke = linhas_correntes - mascara_stroke.sum()
    df_clean = df_clean[mascara_stroke].copy()
    df_clean["Stroke"] = (df_clean["CVDSTRK3"] == 1.0).astype(int)
    linhas_correntes = len(df_clean)
    passos_funil.append({
        "Etapa / Filtro": "10. Stroke (CVDSTRK3 in [1, 2])",
        "Registros Restantes": linhas_correntes,
        "Perda Absoluta": perda_stroke,
        "Retenção (%)": (linhas_correntes / passos_funil[0]["Registros Restantes"]) * 100
    })

    # 11. Sex (_SEX): 1 = Masculino (0), 2 = Feminino (1)
    mascara_sex = df_clean["_SEX"].isin([1, 2])
    df_clean = df_clean[mascara_sex].copy()
    df_clean["Sex"] = (df_clean["_SEX"] == 2).astype(int)

    # 12 e 13. Socioeconômicos: Education e Income
    if incluir_socioeconomicos:
        # Education (_EDUCAG): 1..4 (9 = Missing)
        mascara_edu = df_clean["_EDUCAG"].isin([1, 2, 3, 4])
        perda_edu = len(df_clean) - mascara_edu.sum()
        df_clean = df_clean[mascara_edu].copy()
        df_clean["Education"] = df_clean["_EDUCAG"].astype(int)
        linhas_correntes = len(df_clean)
        passos_funil.append({
            "Etapa / Filtro": "11. Education (_EDUCAG in [1..4])",
            "Registros Restantes": linhas_correntes,
            "Perda Absoluta": perda_edu,
            "Retenção (%)": (linhas_correntes / passos_funil[0]["Registros Restantes"]) * 100
        })

        # Income (_INCOMG1): 1..7 (9 = Don't know/Refused)
        mascara_income = df_clean["_INCOMG1"].isin([1, 2, 3, 4, 5, 6, 7])
        perda_income = len(df_clean) - mascara_income.sum()
        df_clean = df_clean[mascara_income].copy()
        df_clean["Income"] = df_clean["_INCOMG1"].astype(int)
        linhas_correntes = len(df_clean)
        passos_funil.append({
            "Etapa / Filtro": "12. Income (_INCOMG1 in [1..7])",
            "Registros Restantes": linhas_correntes,
            "Perda Absoluta": perda_income,
            "Retenção (%)": (linhas_correntes / passos_funil[0]["Registros Restantes"]) * 100
        })

    # Seleção final e ordenação lógica das colunas tratadas
    colunas_finais = [
        # Variáveis-Alvo
        "Status_Diabetes",
        "Target_Diabetes",
        # Fatores Clínicos
        "HighBP",
        "HighChol",
        "BMI",
        "GenHlth",
        "HeartDiseaseorAttack",
        "Stroke",
        "DiffWalk",
        # Fatores Comportamentais
        "PhysActivity",
        "Smoker",
        # Demográficos e Socioeconômicos
        "Sex",
        "Age",
    ]
    if incluir_socioeconomicos:
        colunas_finais.extend(["Education", "Income"])

    df_final = df_clean[colunas_finais].copy()
    funil_df = pd.DataFrame(passos_funil)

    print("• Funil de Limpeza de Dados (Auditoria Sequencial):")
    print(funil_df.to_string(index=False))
    print()

    return df_final, funil_df


def gerar_relatorio_estatistico(df_bruto: pd.DataFrame, df_limpo: pd.DataFrame) -> None:
    """
    Exibe um relatório estatístico completo no terminal:
      - Auditoria comparativa de volume de dados (bruto vs limpo)
      - Distribuição detalhada da variável-alvo multiclasse e cálculo de desbalanceamento
      - Resumo descritivo univariado de todas as variáveis tratadas
      - Validação de nulos (garantia de integridade pré-Machine Learning)

    Args:
        df_bruto (pd.DataFrame): Dataframe original antes de qualquer tratamento.
        df_limpo (pd.DataFrame): Dataframe final limpo e pronto para modelagem.
    """
    print("=" * 80)
    print("ETAPA 4: RELATÓRIO ESTATÍSTICO DA BASE TRATADA (ITEM 8)")
    print("=" * 80)

    total_bruto = len(df_bruto)
    total_limpo = len(df_limpo)
    total_descartado = total_bruto - total_limpo
    pct_retencao = (total_limpo / total_bruto) * 100
    pct_descarte = (total_descartado / total_bruto) * 100

    print("1. RESUMO DE VOLUMETRIA:")
    print(f"• Total de registros no dataset bruto:      {total_bruto:>10,}")
    print(f"• Total de registros tratados e válidos:    {total_limpo:>10,} ({pct_retencao:6.2f}%)")
    print(f"• Total de instâncias descartadas (ruídos): {total_descartado:>10,} ({pct_descarte:6.2f}%)")
    print(f"• Total de atributos finais no dataset:     {df_limpo.shape[1]:>10}")
    print("-" * 80)

    print("\n2. DISTRIBUIÇÃO DA VARIÁVEL-ALVO MULTICLASSE (Desbalanceamento):")
    contagem_status = df_limpo["Status_Diabetes"].value_counts().reindex(ORDEM_CLASSES)
    pct_status = (df_limpo["Status_Diabetes"].value_counts(normalize=True).reindex(ORDEM_CLASSES)) * 100
    contagem_num = df_limpo["Target_Diabetes"].value_counts().reindex([0, 1, 2])

    tabela_target = pd.DataFrame({
        "Código (Target)": [0, 1, 2],
        "Classe": ORDEM_CLASSES,
        "Total de Instâncias": [f"{v:,}" for v in contagem_status.values],
        "Proporção (%)": [f"{p:.2f}%" for p in pct_status.values],
        "Razão Imbalance (vs Minoria)": [
            f"{contagem_status['Não Diabético'] / contagem_status[c]:.1f}x" for c in ORDEM_CLASSES
        ]
    })
    print(tabela_target.to_string(index=False))

    print("\n  [ALERTA DE DESBALANCEAMENTO DE CLASSES]")
    print(f"  • A classe majoritária ('Não Diabético') representa {pct_status['Não Diabético']:.1f}% da base.")
    print(f"  • A classe minoritária ('Pré-diabético') representa apenas {pct_status['Pré-diabético']:.2f}% da base.")
    print(f"  • Desproporção entre classes: há ~{contagem_status['Não Diabético'] / contagem_status['Pré-diabético']:.1f} Não Diabéticos para cada Pré-diabético.")
    print("  • Recomendações para ML: Utilizar estratificação em treino/teste (stratify=y),")
    print("    avaliar métricas como F1-Score Macro, PR-AUC e ponderar classes (class_weight='balanced').")
    print("-" * 80)

    print("\n3. AUDITORIA DE AUSENTES / NULOS NA BASE FINAL:")
    nulos = df_limpo.isnull().sum()
    if nulos.sum() == 0:
        print("  -> Sucesso: 0 valores nulos identificados em todas as colunas tratadas.")
    else:
        print(nulos[nulos > 0].to_string())
    print("-" * 80)

    print("\n4. ESTATÍSTICAS DESCRITIVAS DAS VARIÁVEIS TRATADAS (.describe().T):")
    cols_numericas = df_limpo.select_dtypes(include=[np.number])
    desc_df = cols_numericas.describe().T[["count", "mean", "std", "min", "50%", "max"]]
    desc_df.rename(columns={"50%": "median"}, inplace=True)
    print(desc_df.to_string())
    print("=" * 80 + "\n")


def salvar_dataset(df_limpo: pd.DataFrame, caminho_saida: Path) -> Path:
    """
    Exporta o dataframe limpo e tratado para arquivo CSV na pasta data/.

    Args:
        df_limpo (pd.DataFrame): Dataframe final limpo.
        caminho_saida (Path): Caminho do arquivo CSV de saída.

    Returns:
        Path: Caminho do arquivo salvo.
    """
    print("=" * 80)
    print("ETAPA 5: EXPORTAÇÃO DO DATASET TRATADO")
    print("=" * 80)

    # Garante que a pasta pai exista
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    print(f"-> Exportando para: {caminho_saida.resolve()}")
    df_limpo.to_csv(caminho_saida, index=False, encoding="utf-8")
    tamanho_mb = caminho_saida.stat().st_size / (1024 * 1024)

    print(f"-> Concluído com sucesso! Arquivo gerado:")
    print(f"   • Tamanho no disco: {tamanho_mb:.2f} MB")
    print(f"   • Total de linhas:  {len(df_limpo):,}")
    print(f"   • Total de colunas: {df_limpo.shape[1]}")
    print("=" * 80 + "\n")
    return caminho_saida


def parse_args():
    """Configura e processa argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Pipeline de Pré-processamento e Engenharia de Atributos - Projeto Big Data (Item 8)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_CSV,
        help="Caminho do CSV de entrada (default: data/Diabetes Health Indicators.csv)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Caminho do CSV de saída tratado (default: data/dados_tratados_multiclasse.csv)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Se informado, apenas executa o pipeline e exibe o relatório sem salvar o CSV em disco."
    )
    parser.add_argument(
        "--sem-socioeconomicos",
        action="store_true",
        help="Se informado, exclui da filtragem estrita os atributos socioeconômicos (Education e Income)."
    )
    return parser.parse_args()


def main():
    """Ponto de entrada principal de execução do pipeline."""
    args = parse_args()

    print("\n" + "#" * 80)
    print("   INÍCIO DO PIPELINE DE PRÉ-PROCESSAMENTO MULTICLASSE (ITEM 8) - BIG DATA")
    print("#" * 80 + "\n")

    # 1. Carregamento do dataset bruto
    df_bruto = carregar_dados(args.input)

    # 2. Processamento da variável-alvo multiclasse
    df_target, _ = processar_variavel_alvo(df_bruto, coluna_alvo="DIABETE4")

    # 3. Processamento e limpeza das preditoras
    df_limpo, _ = processar_preditoras(
        df_target,
        limite_bmi_min=12.0,
        limite_bmi_max=70.0,
        incluir_socioeconomicos=not args.sem_socioeconomicos
    )

    # 4. Relatório estatístico detalhado
    gerar_relatorio_estatistico(df_bruto, df_limpo)

    # 5. Exportação do arquivo tratado
    if not args.no_save:
        salvar_dataset(df_limpo, args.output)
    else:
        print("-> Flag --no-save informada: salvamento do arquivo em disco ignorado.\n")

    print("#" * 80)
    print("   PIPELINE DE PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
