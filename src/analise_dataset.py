"""
=============================================================================
PROJETO BIG DATA - ANÁLISE EXPLORATÓRIA E ENGENHARIA DE VARIÁVEL-ALVO
=============================================================================
Dataset: CDC BRFSS (Behavioral Risk Factor Surveillance System)
Arquivo: Diabetes Health Indicators.csv

Escopo do Problema:
Classificação multiclasse (3 classes oficiais) para rastreamento de Diabetes:
  - "Não Diabético" (Código original CDC: 3)
  - "Pré-diabético" (Código original CDC: 4)
  - "Diabético"     (Códigos originais CDC: 1 e 2 - Diabético e Gestacional)

Etapas executadas:
  1. Carregamento seguro com caminhos relativos via pathlib.
  2. Dimensões (shape) do dataset bruto.
  3. Auditoria detalhada de valores nulos (contagem absoluta e percentual).
  4. Estatísticas descritivas univariadas (.describe().T).
  5. Mapeamento estrito da variável-alvo ('DIABETE4') e descarte de ruídos.
  6. Distribuição das classes multiclasse e contagem de registros válidos.
=============================================================================
"""

import sys
from pathlib import Path
import pandas as pd

# Garante suporte adequado à codificação UTF-8 no console do Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configurações de exibição do Pandas no terminal para máxima legibilidade
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# Definição segura dos caminhos do projeto via pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Diabetes Health Indicators.csv"

# Metadados e mapeamentos oficiais do CDC BRFSS para DIABETE4:
# 1: Diabético
# 2: Diabetes gestacional (durante gravidez)
# 3: Não diabético
# 4: Pré-diabetes ou limítrofe
# 7: Não sabe / Não tem certeza (ruído / label ausente)
# 9: Recusou responder (ruído / label ausente)
MAPA_MULTICLASSE = {
    3.0: "Não Diabético",
    4.0: "Pré-diabético",
    1.0: "Diabético",
    2.0: "Diabético"
}

ORDEM_CLASSES = ["Não Diabético", "Pré-diabético", "Diabético"]


def carregar_dados(caminho_csv: Path) -> pd.DataFrame:
    """
    Carrega o dataset a partir de um caminho seguro pathlib com validação prévia.
    """
    print("=" * 75)
    print("1. CARREGAMENTO DO DATASET")
    print("=" * 75)
    print(f"-> Caminho do arquivo: {caminho_csv}")

    if not caminho_csv.exists():
        raise FileNotFoundError(
            f"Erro crítico: O arquivo não foi encontrado no caminho especificado: {caminho_csv}\n"
            f"Certifique-se de que o arquivo 'Diabetes Health Indicators.csv' está na pasta 'data/'."
        )

    df = pd.read_csv(caminho_csv)
    print("-> Status: Dataset carregado com sucesso!\n")
    return df


def exibir_shape(df: pd.DataFrame) -> None:
    """
    Exibe a contagem de linhas e colunas (shape) do dataset bruto.
    """
    linhas, colunas = df.shape
    print("=" * 75)
    print("2. DIMENSÕES DO DATASET BRUTO (Shape)")
    print("=" * 75)
    print(f"• Total de Instâncias (Linhas): {linhas:,}")
    print(f"• Total de Atributos (Colunas): {colunas}")
    print(f"• Shape estrutural: ({linhas:,}, {colunas})\n")


def auditar_valores_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audita integralmente a presença de valores ausentes por coluna,
    retornando contagem absoluta, porcentagem relativa e tipo de dado.
    """
    print("=" * 75)
    print("3. AUDITORIA COMPLETA DE VALORES NULOS POR ATRIBUTO")
    print("=" * 75)

    contagem_nulos = df.isnull().sum()
    percentual_nulos = (contagem_nulos / len(df)) * 100
    tipos_dados = df.dtypes

    auditoria_df = pd.DataFrame({
        "Tipo": tipos_dados.astype(str),
        "Valores Nulos": contagem_nulos,
        "Porcentagem (%)": percentual_nulos
    })

    print(auditoria_df.to_string())

    total_nulos = contagem_nulos.sum()
    colunas_com_nulos = (contagem_nulos > 0).sum()
    print("-" * 75)
    print(f"• Total consolidado de valores nulos no dataset: {total_nulos:,}")
    print(f"• Colunas contendo ao menos 1 valor nulo: {colunas_com_nulos} de {df.shape[1]}")
    print(f"• Percentual global de células faltantes: {(total_nulos / (df.shape[0] * df.shape[1])) * 100:.2f}%\n")

    return auditoria_df


def gerar_estatisticas_descritivas(df: pd.DataFrame) -> None:
    """
    Gera e formata as estatísticas descritivas básicas das variáveis numéricas (.describe().T).
    """
    print("=" * 75)
    print("4. ESTATÍSTICAS DESCRITIVAS BÁSICAS (.describe().T)")
    print("=" * 75)
    
    colunas_numericas = df.select_dtypes(include=["number"])
    desc_df = colunas_numericas.describe().T
    
    # Adiciona a contagem formatada para facilitar a visualização
    print(desc_df.to_string())
    print("\n")


def processar_variavel_alvo_multiclasse(df: pd.DataFrame, coluna_alvo: str = "DIABETE4") -> pd.DataFrame:
    """
    Processa e converte a variável-alvo 'DIABETE4' estritamente para o escopo
    multiclasse de 3 categorias oficiais:
      - 'Não Diabético' (Código original 3)
      - 'Pré-diabético' (Código original 4)
      - 'Diabético'     (Códigos originais 1 e 2: diabetes tipo 1/2 e gestacional)

    Filtra instâncias não informativas (7 = Não Sabe, 9 = Recusou Responder e NaN),
    garantindo que o dataset resultante contenha apenas rótulos válidos.
    """
    print("=" * 75)
    print(f"5. PROCESSAMENTO E MAPEAMENTO DA VARIÁVEL-ALVO MULTICLASSE ('{coluna_alvo}')")
    print("=" * 75)

    if coluna_alvo not in df.columns:
        raise KeyError(f"A coluna de destino '{coluna_alvo}' não existe no dataset fornecido.")

    # 1. Auditoria prévia dos códigos brutos originais do CDC BRFSS
    print("Distribuição Bruta dos Códigos Originais no BRFSS:")
    mapa_legenda_brfss = {
        1.0: "1 - Sim (Diabético)",
        2.0: "2 - Sim (Apenas durante gestação)",
        3.0: "3 - Não (Não Diabético)",
        4.0: "4 - Pré-diabetes ou limítrofe",
        7.0: "7 - Não sabe / Sem certeza (descartado)",
        9.0: "9 - Recusou responder (descartado)"
    }
    bruto_counts = df[coluna_alvo].value_counts(dropna=False)
    bruto_pct = (df[coluna_alvo].value_counts(dropna=False, normalize=True) * 100)
    
    tabela_bruta = pd.DataFrame({
        "Significado Original CDC": [
            mapa_legenda_brfss.get(val, "Valores Ausentes (NaN)") for val in bruto_counts.index
        ],
        "Quantidade": [f"{v:,}" for v in bruto_counts.values],
        "Proporção Bruta (%)": bruto_pct.values
    }, index=bruto_counts.index)
    tabela_bruta.index.name = "Código"
    print(tabela_bruta.to_string())
    print()

    # 2. Aplicação do mapeamento para o problema multiclasse de 3 categorias
    df_processado = df.copy()
    df_processado["TARGET_MULTICLASSE"] = df_processado[coluna_alvo].map(MAPA_MULTICLASSE)

    # 3. Filtragem de registros válidos para modelagem preditiva
    df_valido = df_processado.dropna(subset=["TARGET_MULTICLASSE"]).copy()
    
    # Categorização explícita e ordenada para consistência analítica
    df_valido["TARGET_MULTICLASSE"] = pd.Categorical(
        df_valido["TARGET_MULTICLASSE"],
        categories=ORDEM_CLASSES,
        ordered=True
    )

    # Registro de descartes de integridade
    total_bruto = len(df)
    total_valido = len(df_valido)
    total_descartados = total_bruto - total_valido

    print("-" * 75)
    print(f"• Registros brutos originais:       {total_bruto:,} (100.00%)")
    print(f"• Registros válidos (3 classes):    {total_valido:,} ({(total_valido / total_bruto) * 100:.2f}%)")
    print(f"• Registros descartados (7, 9, NaN): {total_descartados:,} ({(total_descartados / total_bruto) * 100:.2f}%)")
    print("-" * 75)
    print()

    return df_valido


def exibir_distribuicao_multiclasse(df_valido: pd.DataFrame, coluna_target: str = "TARGET_MULTICLASSE") -> None:
    """
    Exibe a contagem absoluta e percentual consolidada das 3 classes oficiais
    e o total de registros válidos prontos para treinamento e modelagem.
    """
    print("=" * 75)
    print("6. DISTRIBUIÇÃO CONSOLIDADA DAS 3 CLASSES MULTICLASSE (Dataset Válido)")
    print("=" * 75)

    contagem = df_valido[coluna_target].value_counts().reindex(ORDEM_CLASSES)
    proporcao = df_valido[coluna_target].value_counts(normalize=True).reindex(ORDEM_CLASSES) * 100

    mapeamento_detalhe = {
        "Não Diabético": "Código 3 (Sem diagnóstico)",
        "Pré-diabético": "Código 4 (Limítrofe / Pré-diabetes)",
        "Diabético": "Códigos 1 e 2 (Diabetes regular e gestacional)"
    }

    tabela_final = pd.DataFrame({
        "Categoria Multiclasse": ORDEM_CLASSES,
        "Origem no BRFSS (DIABETE4)": [mapeamento_detalhe[c] for c in ORDEM_CLASSES],
        "Contagem Absoluta": [f"{contagem[c]:,}" for c in ORDEM_CLASSES],
        "Proporção Válida (%)": [f"{proporcao[c]:.2f}%" for c in ORDEM_CLASSES]
    })
    tabela_final.set_index("Categoria Multiclasse", inplace=True)

    print(tabela_final.to_string())
    print("-" * 75)
    print(f"TOTAL DE REGISTROS VÁLIDOS PARA O MODELO MULTICLASSE: {len(df_valido):,} instâncias")
    print("=" * 75 + "\n")


def main():
    """
    Pipeline principal de execução da análise exploratória multiclasse.
    """
    print("\n" + "#" * 75)
    print("   INÍCIO DO PIPELINE DE ANÁLISE EXPLORATÓRIA MULTICLASSE - BIG DATA")
    print("#" * 75 + "\n")

    # 1. Carregamento do arquivo CSV via pathlib
    df = carregar_dados(DATA_PATH)

    # 2. Exibição das dimensões (shape)
    exibir_shape(df)

    # 3. Auditoria de valores nulos
    auditar_valores_nulos(df)

    # 4. Estatísticas descritivas numéricas
    gerar_estatisticas_descritivas(df)

    # 5. Processamento e mapeamento estrito da variável-alvo multiclasse
    df_valido = processar_variavel_alvo_multiclasse(df, coluna_alvo="DIABETE4")

    # 6. Exibição da distribuição final das 3 classes e total válido
    exibir_distribuicao_multiclasse(df_valido, coluna_target="TARGET_MULTICLASSE")

    print("#" * 75)
    print("   PIPELINE DE ANÁLISE EXPLORATÓRIA FINALIZADO COM SUCESSO!")
    print("#" * 75 + "\n")


if __name__ == "__main__":
    main()
