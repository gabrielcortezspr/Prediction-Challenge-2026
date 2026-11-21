# Prediction Challenge 2026: nota de reviews

Solução para um desafio de predição: dado o título e o texto de uma review de produto em
português, prever a nota de 1 a 5. Classificador linear sobre features de texto, sem modelos
pré-treinados.

## Abordagem

- **Texto:** `CountVectorizer` de palavras (1 e 2-gramas) e, opcionalmente, de caracteres
  (`char_wb`, 3 a 5-gramas) para pegar erros de digitação e variações de escrita.
- **Features manuais:** tamanho, pontuação, palavras positivas e negativas (`features.py`).
- **Modelo:** `LinearSVC` multiclasse, tudo num `ColumnTransformer` + `Pipeline`.

O histórico das mudanças e o efeito esperado de cada uma está em
[`docs/mudancas-do-modelo.md`](docs/mudancas-do-modelo.md). A melhor submissão marcou
**0,4673** na métrica do desafio.

## Como rodar

```sh
pip install -r requirements.txt
python main.py                                  # treina em train.csv e gera submission.csv
python main.py --use-char-ngrams --cv-folds 5   # com n-gramas de caractere e validação cruzada
```

Principais opções: `--model`, `--feature-mode full|minimal`, `--svm-c`,
`--svm-class-weight none|balanced`, `--count-max-features`.

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `main.py` | Linha de comando: treino, validação e geração da submissão |
| `preprocessing.py` | Leitura, normalização do texto e stopwords |
| `features.py` | Features numéricas extraídas do texto |
| `model.py` | Montagem do pipeline e dos estimadores |
| `utils.py` | Métricas e log |
| `train.csv`, `test.csv` | Dados do desafio (41 mil e 10 mil reviews) |
| `docs/` | Notas das mudanças e material de estudo sobre SVM |

## Licença

[MIT](LICENSE)
