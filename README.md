# Prediction Challenge 2026: nota de reviews

Solução para um desafio de predição: dado o título e o texto de uma review de produto em
português, prever a nota de 1 a 5. Classificador linear sobre features de texto, sem modelos
pré-treinados.

## Abordagem

- **Texto:** `CountVectorizer` de palavras (1 e 2-gramas) mais um de caracteres (`char_wb`,
  3 a 5-gramas) para pegar erros de digitação e variações de escrita.
- **Features manuais:** tamanho, pontuação, palavras positivas e negativas, conectivos de
  antítese ("mas", "porém"), que viram a polaridade da frase (`features.py`).
- **Modelo:** `LinearSVC` multiclasse, tudo num `ColumnTransformer` + `Pipeline`.

## Evolução

| Versão | Mudança | Score no desafio |
|---|---|---|
| 1 | Baseline: TF-IDF + regressão logística | não registrado |
| 2 | Sem IDF, pré-processamento, stopwords que preservam antítese, features melhores | 0,46 |
| 3 | N-gramas de caractere em paralelo aos de palavra, features numéricas enxutas | **0,4673** |

Detalhes de cada mudança e o efeito esperado em
[`docs/mudancas-do-modelo.md`](docs/mudancas-do-modelo.md).

## Como rodar

```sh
pip install -r requirements.txt
python main.py                  # treina com data/train.csv e grava submission.csv
python main.py --cv-folds 5     # inclui validação cruzada estratificada
```

Os padrões do `main.py` são a configuração da versão 3. Outras opções: `--model
linear_svm|logreg|lightgbm`, `--feature-mode full|minimal`, `--no-char-ngrams`, `--svm-c`,
`--svm-class-weight none|balanced`.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `main.py` | Linha de comando: treino, validação e geração da submissão |
| `preprocessing.py` | Leitura, limpeza do texto e stopwords |
| `features.py` | Features numéricas extraídas do texto |
| `model.py` | Montagem do pipeline e dos estimadores |
| `utils.py` | Métricas e log |
| `notebook_final_4951.ipynb` | Pipeline completo (pré-processamento, modelo e treino) num notebook único e comentado |
| `data/` | Dados do desafio: 41 mil reviews de treino e 10 mil de teste |
| `submissions/` | Submissão enviada (versão 3) |
| `docs/` | Notas das mudanças do modelo |

## Licença

[MIT](LICENSE)
