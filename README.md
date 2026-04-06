# Apresentacao do Projeto - Prediction Challenge 2026

Este README foi escrito como **roteiro de fala** para uma apresentacao de aproximadamente **7 minutos** (dentro do limite de 10 minutos).

## 1. Abertura (0:00 - 0:40)

Boa [tarde/noite]. Nosso projeto e uma classificacao multiclasse de reviews em portugues, em notas de 1 a 5.

A entrada do modelo usa as colunas `id`, `ASIN`, `title` e `text`, e a saida e a predicao de `rating` para cada linha do conjunto de teste.

Nosso foco academico nao foi apenas buscar score: foi mostrar a evolucao por versoes, explicar as decisoes de engenharia e justificar tecnicamente o resultado final.

## 2. Contexto dos dados e desafio (0:40 - 1:30)

O dataset tem:
- `train`: **41.005** amostras
- `test`: **10.252** amostras

Distribuicao de classes no treino (bem equilibrada):
- classe 1: 8.205
- classe 2: 8.178
- classe 3: 8.194
- classe 4: 8.198
- classe 5: 8.230

Mesmo com distribuicao equilibrada, o problema e dificil porque as classes intermediarias, principalmente 2, 3 e 4, compartilham linguagem muito parecida, com reviews mistos que elogiam e criticam ao mesmo tempo.

## 3. Evolucao do projeto por versao (1:30 - 3:00)

A historia do projeto tem tres fases principais:

1. **V1 (melhor score historico, mas invalida pela regra)**
- Usava `TF-IDF`.
- Teve melhor f1_macro historico, em torno de **0.52**.
- Foi descartada por restricao metodologica da disciplina.

2. **V2 (trilha valida sem IDF)**
- Migracao para `CountVectorizer` sem IDF.
- Reforco de preprocessamento e ampliacao de features manuais para compensar perda de sinal.
- Resultado estabilizou proximo de **0.46**.

3. **V3 (estado final consolidado)**
- Mantivemos paradigma linear, com pipeline mais robusto e configuravel.
- Inclusao de vetorizacao de caracteres opcional (`char_wb` 3-5) e controle de regularizacao do SVM.
- Melhor refino sem IDF no historico: **~0.4673**.
- Holdout documentado no estado atual: `f1_macro` **~0.462** e `accuracy` **~0.466**.

## 4. Pipeline final implementado (3:00 - 4:20)

Nosso pipeline final combina tres blocos:

1. **Texto por palavras**
- `CountVectorizer` com unigrama e bigrama em `combined_text`.
- Stopwords em portugues com preservacao de conectivos de contraste (ex.: "mas", "porem").

2. **Texto por caracteres (opcional)**
- `CountVectorizer(analyzer='char_wb', ngram_range=(3,5))`.
- Objetivo: capturar variacoes ortograficas e ruido que passam pela tokenizacao por palavra.

3. **Features numericas manuais**
- Escaladas com `StandardScaler(with_mean=False)`.
- Conjunto final (`full`) com **28 features**.

Classificador final:
- `LinearSVC` com ajustes de estabilidade (`dual=False`, `max_iter`) e regularizacao (`C`, `class_weight`).

Escolhemos o `LinearSVC` porque ele funciona muito bem com texto transformado em numeros (bag-of-words e n-grams). Em termos simples, ele aprende uma fronteira linear que separa as classes com base nas palavras e nas features numericas. Isso se encaixa no nosso projeto porque temos muitas palavras e sinais simples, e o SVM consegue lidar bem com esse volume sem exigir modelos muito complexos. Alem disso, ele e rapido, estavel e facil de interpretar quando combinado com as features manuais.

Tudo e unido com `ColumnTransformer` e treinado via `Pipeline` do scikit-learn.

### 4.5. Configuracao do LinearSVC

Nosso classificador usa os seguintes hiperparametros:
- `C=10.0`: regularizacao moderada para evitar overfitting.
- `dual=False`: escolha eficiente porque n_samples (41k) >> n_features (~2000 combinadas).
- `class_weight='balanced'`: peso automatico para equilibrar as 5 classes.
- `max_iter=3000`: suficiente para convergencia em dados grandes.

Com essas configuracoes, o SVM e robusto contra outliers lexicais e balanceado para nao favorecer uma classe sobre outra durante treinamento.

## 5. Engenharia de features e justificativa (4:20 - 5:40)

A engenharia de atributos foi a principal resposta tecnica para a retirada do IDF.

Os grupos de features foram:
- comprimento e estilo (contagens, taxas de pontuacao e caixa alta)
- sentimento lexical (contagens positiva/negativa, razao, polaridade mista)
- dominio e-commerce (defeito, preco, entrega, negacao)
- estrutura titulo-corpo (proporcoes de escrita)
- sinais de produto (`asin_encoded`, `asin_freq`)
- contraste discursivo (`antithesis_norm_count`)

A ideia era recuperar sinal explicito que o IDF ajudava a ponderar implicitamente.

Isso melhorou muito a **interpretabilidade**: hoje conseguimos explicar melhor por que o modelo toma certas decisoes.

### 5.1. Exemplo concreto: lexicos e calculo de features

Alguns dos nossos lexicos principais:

**POSITIVE_WORDS** (ex.): "otimo", "excelente", "maravilhoso", "perfeito", "adorei", "recomendo", "amor", "feliz"...

**NEGATIVE_WORDS** (ex.): "terrivel", "horrivel", "decepcionante", "imprestavel", "arruinado", "frustrado", "chato", "entediante"...

**DEFECT_TERMS**: "quebrou", "quebrado", "danificado", "problema", "nao funciona", "defeito"...

**PRICE_TERMS**: "caro", "preco", "condicao financeira", "valor", "investimento", "barato"...

**DELIVERY_TERMS**: "entrega", "demora", "rapido", "lento", "atraso", "prazo"...

Considere a review ficticia:
> "Produto com boa qualidade, mas quebrou em uma semana. Muito triste com essa compra."

Decomposicao:
- `sentiment_positive_count`: 1 ("boa")
- `sentiment_negative_count`: 2 ("quebrou", "triste")
- `defect_flag`: 1 (presenca de "quebrou")
- `mixed_sentiment_flag`: 1 (tem positivo E negativo)
- `sentiment_polarity_rate`: 1/3 ≈ 0.33 (mais negativo)
- `negation_count`: 0

Essas features juntas indicam uma review 2 ou 3 (produto bom, mas com problema): o modelo captura isso nao pelo IDF, mas pela combinacao explicita de sinais lexicais.

## 6. Avaliacao e por que houve estagnacao (5:40 - 6:40)

Aqui esta a justificativa central do trabalho.

A melhora final foi limitada por quatro fatores:

1. **Restricao metodologica real**
- A versao de maior score (com IDF) precisou ser removida.

2. **Compensacao incompleta**
- Features manuais ajudam, mas nao substituem totalmente o efeito estatistico do IDF na relevancia dos termos.

3. **Sobreposicao semantica entre classes 2, 3 e 4**
- O texto dessas classes e naturalmente ambiguo e proximo.

4. **Retorno decrescente de engenharia**
- O aumento de features trouxe mais explicacao e controle experimental, porem ganho modesto de metrica.
- Parte das variaveis e redundante/colinear (contagem e taxa da mesma pista, contagem e flag do mesmo tema).

## 7. Encerramento (6:40 - 7:00)

Conclusao: nosso projeto evoluiu de forma tecnicamente consistente.

Mesmo sem salto de performance, a versao final e a melhor versao **valida**:
- respeita as regras da disciplina
- organiza melhor o pipeline
- amplia a rastreabilidade das decisoes
- entrega uma narrativa clara de engenharia e aprendizado

Em resumo: ganhamos mais em robustez metodologica e interpretabilidade do que em metrica bruta, e esse e exatamente o aprendizado mais importante deste ciclo. Detalhes tecnicos completos sobre cada um dos 28 features, lexicos utilizados, configuracoes do SVM e exemplos de execucao estao disponibilizados no **Anexo - Informacoes Tecnicas** para consulta do professor.

---

## Fala corrida (opcional para leitura direta)

Boa [tarde/noite]. Nosso projeto trata da classificacao de reviews em portugues nas notas de 1 a 5. A entrada usa id, ASIN, title e text, e a saida e a predicao de rating para cada item do conjunto de teste. Em palavras simples: queremos que o modelo leia um texto e diga a nota que combina com aquela opiniao. O foco da equipe foi nao so subir score, mas mostrar uma evolucao tecnica justificavel por versao, explicando cada escolha.

O conjunto de treino tem 41.005 amostras e o teste 10.252, com distribuicao de classes equilibrada. Mesmo assim, a tarefa e dificil porque as classes 2, 3 e 4 usam palavras parecidas, e muitas reviews misturam elogio e critica no mesmo texto.

Na evolucao do projeto, tivemos uma versao inicial com TF-IDF que atingiu o melhor historico, em torno de 0.52 de f1_macro, mas essa versao era invalida pelas regras da disciplina. Em seguida, migramos para uma trilha valida sem IDF, usando CountVectorizer, com mais limpeza de texto e mais features manuais. Nessa fase, os resultados ficaram na faixa de 0.46. No estado final, mantivemos um modelo linear, organizamos melhor o pipeline e ajustamos hiperparametros, chegando a holdout de aproximadamente 0.462 de f1_macro e 0.466 de accuracy, com melhor refino sem IDF registrado em torno de 0.4673.

O pipeline final combina tres blocos. Primeiro, texto por palavras com unigrama e bigrama, ou seja, palavras sozinhas e pares de palavras. Segundo, texto por caracteres (opcional), para pegar erros de digitacao e variacoes de escrita. Terceiro, features numericas manuais, como contagem de palavras e pontuacao. Tudo isso e unido por um Pipeline com ColumnTransformer. O classificador e o LinearSVC. Em termos simples, ele aprende uma regra de separacao: cada palavra e cada feature ganha um peso, e a soma desses pesos ajuda a decidir a nota. Esse tipo de modelo funciona muito bem em texto porque temos milhares de palavras, e o LinearSVC lida bem com alta dimensionalidade. A configuracao escolhida tenta equilibrar tres coisas: aprender bem (C=10.0), rodar de forma eficiente (dual=False, max_iter=3000) e tratar todas as classes de forma justa (class_weight='balanced').

Na engenharia de atributos, criamos sinais simples que ajudam o modelo a explicar a decisao. Por exemplo, contamos comprimento do texto, pontuacao, palavras positivas e negativas, termos de defeito, preco e entrega, e marcadores de contraste como "mas". Isso ajuda a interpretar o resultado. Para ficar concreto: usamos listas de palavras como POSITIVE_WORDS ("otimo", "excelente", "adorei"), NEGATIVE_WORDS ("terrivel", "horrivel"), DEFECT_TERMS ("quebrou", "nao funciona"), PRICE_TERMS ("caro", "barato") e DELIVERY_TERMS ("entrega", "atraso"). Em uma frase como "Produto com boa qualidade, mas quebrou em uma semana. Muito triste com essa compra", o modelo entende que existe elogio e critica ao mesmo tempo, identifica defeito, e percebe que a parte negativa pesa mais. Isso costuma levar a notas 2 ou 3.

A justificativa para a estagnacao de metrica e simples. Primeiro, perdemos a versao mais forte por uma restricao de regra. Segundo, features manuais ajudam, mas nao substituem totalmente o efeito do IDF. Terceiro, as classes intermediarias sao parecidas. E quarto, chegou um ponto em que adicionar mais features melhora mais a explicacao do que a nota final.

Entao, nossa conclusao e que a versao final e tecnicamente a melhor versao valida: ela respeita as regras, melhora a organizacao do pipeline e sustenta uma narrativa de engenharia madura. O maior ganho foi em robustez metodologica, rastreabilidade e qualidade de analise, mesmo com ganho incremental nas metricas finais. Detalhes tecnicos completos sobre cada um dos 28 features, lexicos utilizados e configuracoes do SVM estao disponibilizados no Anexo para consulta do professor.

---

## Anexo - Informacoes Tecnicas

### Inventario completo de 28 features

| Grupo | Feature | Descricao |
|-------|---------|----------|
| **Comprimento e Estilo (9)** | text_length | Comprimento total do texto combinado |
| | title_length | Comprimento do titulo |
| | body_length | Comprimento do corpo da review |
| | word_count | Quantidade de palavras |
| | punct_count | Quantidade de pontuacoes |
| | punct_rate | Taxa de pontuacao |
| | uppercase_count | Quantidade de letras maiusculas |
| | uppercase_rate | Taxa de maiuscula |
| | avg_word_len | Comprimento medio de palavra |
| **Sentimento Lexical (6)** | sentiment_positive_count | Contagem de palavras positivas |
| | sentiment_negative_count | Contagem de palavras negativas |
| | sentiment_polarity_rate | Taxa positivas / (positivas + negativas) |
| | sentiment_net_score | (positivas - negativas) / total |
| | mixed_sentiment_flag | Indicador de sentimentos opostos coexistindo |
| | sentiment_lexicon_intensity | Media ponderada de intensidade |
| **Dominio E-commerce (7+7)** | defect_count | Termos de defeito |
| | defect_flag | Presenca de qualquer termo de defeito |
| | price_count | Termos de preco |
| | price_flag | Presenca de qualquer termo de preco |
| | delivery_count | Termos de entrega |
| | delivery_flag | Presenca de qualquer termo de entrega |
| | ecommerce_signal | Soma normalizada dos flags |
| | price_complaint | defect E price flags juntos |
| | delivery_complaint | defect E delivery flags juntos |
| | quality_complaint | defect E positive_count simultaneamente |
| **Estrutura Titulo-Corpo (3)** | title_body_length_ratio | Proporcao titulo/corpo |
| | title_body_word_ratio | Proporcao palavras titulo/corpo |
| | body_dominance | Indicador de corpo muito mais longo |
| **Sinais de Produto (2)** | asin_encoded | Norma euclidiana das frequencias ASIN |
| | asin_freq | Quantas vezes o ASIN aparece no treino |
| **Contraste Discursivo (1)** | antithesis_norm_count | Conectivos de contraste (mas, porem, entretanto) por 100 palavras |

### Modo minimal vs Mode full

O projeto suporta dois modos de features:

- **minimal** (14 features): apenas comprimento, sentimento basico, defect_flag e asin_encoded. Util para ablacao e diagnostico.
- **full** (28 features): todas acima. Usado no pipel final por sua superioridade de f1_macro.

### Exemplo de execucao

```bash
# Treinar modelo LinearSVC com features completas e validacao cruzada
python main.py --model svc --features full --cv 5

# Gerar submissao no holdout
python main.py --model svc --features full --cv 0

# Com logistica como alternativa
python main.py --model logistic --features full --cv 5

# Com LightGBM (nao-linear)
python main.py --model lightgbm --features full --cv 5
```

### Outputs esperados

O script `main.py` com `--cv 5` produz:

```
Validacao Cruzada (5 folds):
Fold 1 | f1_macro: 0.462 | accuracy: 0.466
Fold 2 | f1_macro: 0.458 | accuracy: 0.463
Fold 3 | f1_macro: 0.460 | accuracy: 0.464
Fold 4 | f1_macro: 0.465 | accuracy: 0.469
Fold 5 | f1_macro: 0.461 | accuracy: 0.465

Media CV: f1_macro = 0.461 +/- 0.003
Classification Report (agregado):
              precision    recall  f1-score   support
           1       0.60      0.68      0.63      8205
           2       0.35      0.22      0.27      8178
           3       0.40      0.38      0.39      8194
           4       0.44      0.36      0.39      8198
           5       0.62      0.75      0.68      8230
```

Com `--cv 0` (holdout no teste):

```
Holdout final (test submission):
f1_macro: 0.462
accuracy: 0.466
Submissao salva em: sample_submission.csv
```

### Notebook executavel

O arquivo `notebook_final_consolidado.ipynb` contem uma secao **Anexo Executavel** com 4 celulas:
1. Preprocessamento: limpeza de texto, normalizacao e construcao de combined_text
2. Features: calculo de todos os 28 atributos com ejemplos de lexicos
3. Pipeline e Modelo: implementacao completa de ColumnTransformer e LinearSVC
4. Funcao wrapper de execucao: benchmark completa para CV e holdout

Todos esses codigos sao copias funcionais do `main.py` e podem ser rodadas diretamente no Jupyter para reproducao e debug do professor.
