# Modelo Atual e Mudancas Aplicadas

## Resumo do modelo atual
O pipeline final usa um classificador tradicional (`LinearSVC`) com features textuais e numericas:

- Vetorizacao de palavras com `CountVectorizer` (ngrams de 1 a 2).
- Vetorizacao de caracteres opcional com `CountVectorizer` (`char_wb`, ngrams de 3 a 5).
- Features numericas manuais (modo `full`) ou conjunto enxuto (modo `minimal`).
- Concatenacao de todas as features via `ColumnTransformer`.
- Classificacao multiclasse (1 a 5) com `LinearSVC`.

## Mudancas significativas implementadas

### 1. Pipeline hibrido de texto (word + char n-grams)
Arquivo: `model.py`

Foi adicionado suporte a dois vetorizadores em paralelo:
- `word_vectorizer`: capta palavras e bigramas.
- `char_vectorizer` (opcional): capta padroes sublexicais, erros ortograficos, variacoes de escrita e morfologia.

Impacto esperado:
- Melhor robustez para texto ruidoso.
- Ganhos em classes intermediarias (2, 3 e 4), que costumam ter fronteiras mais ambiguas.

### 2. Controle de regularizacao do SVM
Arquivo: `model.py`

`LinearSVC` agora aceita:
- `svm_c` (forca da regularizacao).
- `svm_class_weight` (`none` ou `balanced`).

Tambem foram ajustados:
- `dual=False`
- `max_iter=3000`

Impacto esperado:
- Mais estabilidade no treino e melhor ajuste do trade-off bias/variance sem mudar de familia de modelo.

### 3. Modo de features numericas (`full`/`minimal`)
Arquivo: `features.py`

`get_numeric_feature_columns(mode=...)` agora permite:
- `full`: conjunto completo de sinais manuais.
- `minimal`: apenas sinais mais robustos e menos sujeitos a ruido:
  - `word_count`
  - `sentiment_score`
  - `antithesis_norm_count`
  - `asin_freq`

Impacto esperado:
- Reducao de overfitting por excesso de features manuais.
- Baseline mais simples para comparacoes de experimento.

### 4. Validacao cruzada opcional
Arquivo: `main.py`

Foi adicionado `--cv-folds` para calcular media e desvio padrao de:
- `f1_macro`
- `accuracy`

Impacto esperado:
- Estimativa mais confiavel de generalizacao do que apenas um unico holdout.

### 5. Novos argumentos de experimento
Arquivo: `main.py`

Novos argumentos adicionados:
- `--feature-mode {full,minimal}`
- `--use-char-ngrams`
- `--char-max-features`
- `--svm-c`
- `--svm-class-weight {none,balanced}`
- `--cv-folds`

Isso facilita comparacoes reproduziveis sem editar codigo.

## Exemplo de execucao

Com configuracao hibrida e conjunto minimo:

```bash
/home/gabrielcortezspr/Documents/Prediction-Challenge-2026/.venv/bin/python main.py \
  --model linear_svm \
  --feature-mode minimal \
  --use-char-ngrams \
  --count-max-features 5000 \
  --char-max-features 3000
```

## Resultado observado na validacao holdout (teste rapido)

Configuracao usada:
- `linear_svm`
- `feature_mode=minimal`
- `use_char_ngrams=true`
- `count_max_features=5000`
- `char_max_features=3000`

Metricas:
- `f1_macro`: **0.46209**
- `accuracy`: **0.46580**

Tambem foi gerado com sucesso:
- `submission.csv`

## Como evoluir a partir daqui

Sugestoes de proxima iteracao:
1. Fazer busca de hiperparametros de `svm_c` (ex.: 0.5, 1.0, 1.5, 2.0) com `--cv-folds 5`.
2. Comparar `feature_mode=full` vs `minimal` com mesma configuracao de texto.
3. Ajustar `count_max_features` e `char_max_features` em grid pequeno para equilibrar desempenho e tempo.
4. Testar `--svm-class-weight balanced` para avaliar impacto nas classes 2-4.
