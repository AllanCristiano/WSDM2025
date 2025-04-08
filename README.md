# WSDM Cup Multilingual Chatbot Arena - Ensemble Pipeline

Este repositório contém um pipeline completo para o treinamento de um ensemble de classificadores LightGBM aplicados ao desafio [WSDM Cup Multilingual Chatbot Arena](https://www.kaggle.com/c/wsdm-cup-multilingual-chatbot-arena). O código realiza pré-processamento de texto, extração de características (incluindo vetorização TF-IDF, similaridade de cosseno e estatísticas de texto), treinamento de modelos e geração do arquivo de submissão para Kaggle.

---

## Sumário

- [Descrição](#descrição)
- [Dependências](#dependências)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Como Utilizar](#como-utilizar)
- [Detalhes do Pipeline](#detalhes-do-pipeline)
  - [Pré-processamento dos Dados](#pré-processamento-dos-dados)
  - [Extração de Características](#extração-de-características)
  - [Treinamento do Modelo](#treinamento-do-modelo)
  - [Avaliação e Submissão](#avaliação-e-submissão)
- [Observações](#observações)

---

## Descrição

O objetivo deste projeto é prever o "vencedor" entre duas respostas de chatbot (model_a e model_b) para cada prompt fornecido, utilizando um conjunto de dados multilíngue. O pipeline compreende:

1. **Carregamento dos dados:** Leitura dos arquivos `train.parquet`, `test.parquet` e `sample_submission.csv` fornecidos pelo Kaggle.
2. **Pré-processamento:** Normalização dos textos e extração de características numéricas (por exemplo, comprimento das respostas e número de palavras únicas).
3. **Vetorização com TF-IDF:** Transformação dos textos (prompt, response_a e response_b) em representações numéricas.
4. **Cálculo de Similaridade:** Medição da similaridade de cosseno entre o prompt e cada resposta.
5. **Combinação de Features:** Junção das features extraídas com as representações TF-IDF.
6. **Treinamento do Modelo:** Configuração de quatro modelos LightGBM com diferentes hiperparâmetros e criação de um ensemble utilizando `VotingClassifier` com votação suave.
7. **Avaliação:** Cálculo da acurácia e geração de um relatório de classificação para o conjunto de validação.
8. **Submissão:** Geração do arquivo `submission.csv` mapeando as previsões de volta para os rótulos originais.

---

## Dependências

Certifique-se de ter instaladas as seguintes bibliotecas:

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [lightgbm](https://lightgbm.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)
- [scipy](https://www.scipy.org/)

Você pode instalá-las utilizando o `pip`:

```bash
pip install pandas numpy lightgbm scikit-learn scipy
