# Classificação de diabetes com Python

Projeto da disciplina **Tópicos em Big Data em Python**. A tarefa é classificar registros da pesquisa BRFSS em **sem diabetes**, **pré-diabetes** e **diabetes** usando indicadores de saúde, hábitos, dados demográficos e acesso a serviços de saúde.

- [Abrir o simulador](https://gydeontheguardian.github.io/projeto_big_data_python/simulador/)
- [Consultar métricas e matrizes de confusão](output/desempenho_modelos.html)
- [Ler o relatório em PDF](output/pdf/resultados_modelagem_diabetes.pdf)

O simulador é uma demonstração acadêmica. As previsões são executadas no navegador e não substituem avaliação clínica.

## Como o projeto foi desenvolvido

1. **Análise exploratória.** O script `src/analise_dataset.py` inspeciona os 433.323 registros e as 20 colunas do CSV de entrada, audita valores ausentes e descreve a distribuição da variável-alvo.
2. **Definição do alvo.** `DIABETE4` foi recodificado para as três classes. O projeto agrupou diabetes gestacional com diabetes e descartou respostas desconhecidas, recusadas ou ausentes. `DIABTYPE` ficou fora dos preditores porque contém informação diretamente relacionada ao desfecho.
3. **Pré-processamento.** `src/preprocessamento.py` valida os códigos das respostas, converte o IMC para sua unidade original e registra as perdas por filtro. A base final tem 266.972 registros e 18 preditores, incluindo dias de saúde mental e física ruim, cobertura de saúde, barreira de custo e exame de colesterol. A exclusão de respostas inválidas mantém entradas completas, mas reduz e pode alterar a composição da amostra.
4. **Visualização.** `src/plot_atributos.py` gera um gráfico por preditor. As proporções são calculadas dentro de cada classe para permitir a comparação apesar do desbalanceamento. O script `src/plot_distribuicoes.py` contém gráficos exploratórios anteriores.
5. **Divisão dos dados.** A partição estratificada 80/20, com semente 42, reservou 213.577 registros para treino e 53.395 para teste. Os três modelos usam a mesma partição.
6. **Balanceamento.** A equipe comparou ponderação, undersampling, Random Oversampling e SMOTENC. O Random Oversampling de 50% obteve o maior F1 macro na comparação para os três modelos. Em cada treino, a classe minoritária abaixo do limite é aumentada até metade da classe majoritária. O teste mantém a distribuição original.
7. **Treinamento e ajuste.** `src/modelagem.py` compara Decision Tree, Random Forest e AdaBoost. O Grid Search usa 3 folds estratificados e seleciona pelo F1 macro, que atribui igual peso às três classes. A reamostragem fica dentro do pipeline, aplicada somente à parte de treino de cada fold. Após a seleção, o pipeline retreina no treino completo. O Random Search exploratório encontrou as mesmas configurações nesta grade com menos avaliações. A entrega usa Grid Search conforme a diretriz.
8. **Avaliação.** O teste reservado foi usado para calcular as métricas finais e as matrizes de confusão. A precisão e a sensibilidade de pré-diabetes ajudam a interpretar os resultados além da média global.
9. **Simulador.** O formulário transforma peso e altura em IMC e apresenta faixas de renda domiciliar anual em reais, estimadas por posição relativa na PNAD Contínua 2023. O AdaBoost apresenta a primeira previsão e o usuário pode comparar os outros dois modelos. A adaptação das faixas é aproximada e o desempenho na população brasileira ainda não foi avaliado.

## Resultados

| Modelo | F1 macro na validação | F1 macro no teste | Precisão em pré-diabetes | Sensibilidade em pré-diabetes |
|---|---:|---:|---:|---:|
| Decision Tree | 0,4404 | 0,4448 | 5,20% | 10,81% |
| Random Forest | 0,4542 | 0,4563 | 5,84% | 6,04% |
| AdaBoost | 0,4531 | 0,4563 | 5,77% | 10,96% |

Random Forest e AdaBoost empataram a quatro casas decimais no F1 macro do teste. O AdaBoost identificou 147 dos 1.341 casos de pré-diabetes, contra 81 da Random Forest. A baixa precisão limita a interpretação das previsões individuais.

## Como executar

Na raiz do repositório, com Python 3.12:

```sh
python -m pip install -r requirements.txt
python src/analise_dataset.py
python src/preprocessamento.py
python src/plot_atributos.py
python src/modelagem.py
```

Abra `output/simulador/index.html` para usar o simulador localmente. Mantenha os arquivos da pasta juntos. O Grid Search pode levar alguns minutos. A opção `python src/modelagem.py --sem-grid-search` roda a configuração de referência com ponderação. Os gráficos são gerados em `outputs/atributos/`.

## Fontes

- [CDC · BRFSS 2023, dados e documentação](https://www.cdc.gov/brfss/annual_data/annual_2023.html)
- [IBGE · PNAD Contínua anual, microdados da primeira visita](https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Anual/Microdados/Visita/Visita_1/)
