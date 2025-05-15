# Notas:

## Parser:
### preprocess_train
- Inicialmente removemos variavies nao importante como Id

- Transformação de variáveis categóricas em variáveis binárias (dummies)
- Como temos drop_first=True ou seja vamos ter mais N(numero de atributos)-1 de colunas 

- Variaveis categoricas:
    - person_home_ownership:
        - own
        - Mortage
        - Own
    - loan_intent:
        - Education
        - Medical
        - Personal
        - Venture
        - Debtconsolidation
    - loan_grade:
        - A
        - B
        - C
    - cb_person_default_on_file:
        - Y
        - N
- Dentro do treino dividir entre treino e teste,80% para treino real (X_train, y_train) e  20% para validação (X_val, y_val). Desta forma serve para testar o modelo antes de usar no teste oficial

## Metodos para equilibrar resultados:
- balance_classes:
    - Este método ajusta automaticamente o peso de cada classe com base na sua frequência inversa no conjunto de treino.

    - Ou seja, dá mais peso às classes minoritárias e menos peso às majoritárias, sem alterar os dados em si.

## Feature engenering:
- Criar novas variáveis que combinam informação relevante para ajudar o modelo a detetar padrões mais facilmente — especialmente padrões relacionados com capacidade financeira e risco.
### Variaveis usadas:
- Correção de valores 0 na coluna person_emp_length (anos de experiência profissional):
    Os zeros são substituídos pela mediana da coluna para evitar valores inválidos que podem prejudicar os cálculos subsequentes, nomeadamente divisões por zero. Isto melhora a qualidade e a robustez dos novos atributos criado

- **financial_burden**:
- Calcula uma estimativa do custo total dos juros do empréstimo, dado pelo produto do montante do empréstimo        (loan_amnt)    pela taxa de juro (loan_int_rate). Esta variável ajuda a captar o peso financeiro do empréstimo para o   indivíduo, que pode estar correlacionado com a probabilidade de incumprimento.

- **income_per_year_emp**:
- Mede o rendimento médio anual do indivíduo, obtido ao dividir o rendimento anual (person_income) pelo tempo de experiência profissional (person_emp_length). Esta variável reflete a capacidade financeira ajustada pelo tempo no mercado de trabalho, podendo indicar estabilidade ou crescimento da carreira.

- **int_per_year_emp**:
- Calcula a taxa de juro anual ajustada pelo tempo de experiência, dividindo a taxa de juro do empréstimo pela experiência profissional. Esta métrica pode evidenciar se o custo do empréstimo é elevado relativamente à experiência financeira da pessoa, o que pode impactar no risco de incumprimento.


## Calculo do tempo é com o treino +teste


## logisticRegression:
## Treino:
- LogisticRegression(max_iter=1000, random_state=42)
- Resultados:
```
Exatidão (validação): 0.911842441810896

Classification report:
               precision    recall  f1-score   support

           0       0.93      0.97      0.95     10087 
           1       0.77      0.53      0.63      1642 

    accuracy                           0.91     11729 
   macro avg       0.85      0.75      0.79     11729 
weighted avg       0.90      0.91      0.90     11729 

Matriz de confusão:
[[9832  255]
 [ 779  863]]
Tempo de treino: 0.0700 segundos
Tempo de predição (validação): 0.0000 segundos
Submission saved on 'submission.csv'
Tempo para processar teste: 0.0810 segundos
[LogisticRegression] Tempo de execução: 0.37 segundos
```

- Conclusao:
- Bons resultados para um modelo tão simples porem existe um possivel fator problematico que é ataxa de recall para load aproval 1 que é de 0,53 significa que o modelo só identifica 53% dos clientes problemáticos.
- Oque pode ser causado pelo um maior numero de 0 do que 1
- Solução: 
    - usar tecnicas para equilibrar os dados

## Treino com balance_classes:

- Resultados:
```
Exatidão (validação): 0.8397987893256033

Classification report:
               precision    recall  f1-score   support

           0       0.97      0.84      0.90     10087
           1       0.46      0.83      0.59      1642

    accuracy                           0.84     11729
   macro avg       0.71      0.84      0.75     11729
weighted avg       0.90      0.84      0.86     11729

Matriz de confusão:
[[8484 1603]
 [ 276 1366]]
Tempo de treino: 0.0780 segundos
Tempo de predição (validação): 0.0000 segundos
Submission saved on 'submission.csv'
Tempo para processar teste: 0.0900 segundos
[LogisticRegression] Tempo de execução: 0.39 segundos

```
- Melhorou o recall porem pirou outras metricas talvez mais importantes como a precisao e o f1-score, levando um aumento do numero de falsos negativos
- ISto aconteceu pois o modelo dá mais importância à classe minoritária (no teu caso, o loan_status = 1). 



## Uso de Feature engenering
```
Exatidão (validação): 0.9149970159433882

Classification report:
               precision    recall  f1-score   support

           0       0.93      0.97      0.95     10087
           1       0.78      0.55      0.64      1642

    accuracy                           0.91     11729
   macro avg       0.85      0.76      0.80     11729
weighted avg       0.91      0.91      0.91     11729

Matriz de confusão:
[[9828  259]
 [ 738  904]]
Tempo de treino: 0.1280 segundos
Tempo de predição (validação): 0.0010 segundos
Submission saved on 'submission.csv'
Tempo para processar teste: 0.0950 segundos
[LogisticRegression] Tempo de execução: 0.48 segundos
```
- O recall subiu de 0.53 → 0.55

- O f1-score subiu de 0.63 → 0.64

- Isto significa que o modelo passou a identificar um pouco melhor quem não paga o empréstimo, sem perder qualidade nos outros caso

## Interpretação

- Class weights também estão a ajudar o modelo a tratar a classe 1 com mais importância.

- O teste (sem nada) obtém maior exatidão  mas ignora a classe 1, o que é problematico em problemas de fraude ou risco.

## RandomForestClassifier:
## Treino:
- emodelUsed = RandomForestClassifier(n_estimators=100, random_state=42)
- Resultados:
```
Exatidão (validação): 0.9517435416489044

Classification report:
               precision    recall  f1-score   support

           0       0.96      0.99      0.97     10087
           1       0.92      0.72      0.81      1642

    accuracy                           0.95     11729
   macro avg       0.94      0.86      0.89     11729
weighted avg       0.95      0.95      0.95     11729

Matriz de confusão:
[[9977  110]
 [ 456 1186]]
Tempo de treino: 4.2620 segundos
Tempo de predição (validação): 0.1930 segundos
Submission saved on 'submission.csv'
Tempo para processar teste: 0.6830 segundos
[RandomForest] Tempo de execução: 5.36 segundos
```
- ##Conclusao:

- Foi especialmente superior na classe minoritária, onde o modelo de regressão logística teve um desempenho mais fraco (recall 0.53 vs 0.72).

- Não foi necessário Equilibrio explícito ( class_weight) para a Random Forest ter um bom desempenho — isso é esperado, porque Random Forest lida melhor com desEquilibrio ao não ser um modelo linear e ao fazer bootstraping com vários subconjuntos dos dados.

## RandomForestClassifier com class_weight:

- Resultados:
```
Exatidão (validação): 0.9514877653678916

Classification report:
               precision    recall  f1-score   support

           0       0.95      0.99      0.97     10087
           1       0.92      0.71      0.80      1642

    accuracy                           0.95     11729
   macro avg       0.94      0.85      0.89     11729
weighted avg       0.95      0.95      0.95     11729

Matriz de confusão:
[[9989   98]
 [ 471 1171]]
Tempo de treino: 4.3159 segundos
Tempo de predição (validação): 0.1670 segundos
Submission saved on 'submission.csv'
Tempo para processar teste: 0.6889 segundos
[RandomForest] Tempo de execução: 5.38 segundos

```

- ## Conclusao:
- Random Forest com class_weight não tem qualquer melhoria mas ja era esperado   porque Random Forest lida melhor com desEquilibrio ao não ser um modelo linear e ao fazer bootstraping com vários subconjuntos dos dados.


## RandomForestClassifier com Feature engineering:

```
Exatidão (validação): 0.9470543098303351

Classification report:
               precision    recall  f1-score   support

           0       0.95      0.99      0.97     10087
           1       0.89      0.71      0.79      1642

    accuracy                           0.95     11729
   macro avg       0.92      0.85      0.88     11729
weighted avg       0.95      0.95      0.94     11729

Matriz de confusão:
[[9950  137]
 [ 484 1158]]
Tempo de treino: 8.4121 segundos
Tempo de predição (validação): 0.1850 segundos
Submission saved on 'submission.csv'
Tempo para processar teste: 0.7240 segundos
[RandomForest] Tempo de execução: 9.57 segundos

```
- Estas features podema acrescentar alguma redundancia desbalancear a árvore ao fazer splits em features artificiais menos relevantes


## XGBoost:
## Treino:
- XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
    )
- Resultados:
```
Exatidão (validação): 0.9534487168556569

Classification report:
               precision    recall  f1-score   support

           0       0.96      0.99      0.97     10087
           1       0.92      0.74      0.82      1642

    accuracy                           0.95     11729
   macro avg       0.94      0.86      0.89     11729
weighted avg       0.95      0.95      0.95     11729

Matriz de confusão:
[[9975  112]
 [ 434 1208]]
Tempo de treino: 0.4500 segundos
Tempo de predição (validação): 0.0080 segundos
Submission saved on 'submission.csv'
Tempo para processar teste: 0.1390 segundos
[XGBoost] Tempo de execução: 0.87 segundos

```


## Treino com balance_classes:

- Resultados:
```
Exatidão (validação): 0.9252280671839032

Classification report:
               precision    recall  f1-score   support

           0       0.97      0.94      0.96     10087
           1       0.70      0.83      0.76      1642

    accuracy                           0.93     11729
   macro avg       0.83      0.89      0.86     11729
weighted avg       0.93      0.93      0.93     11729

Matriz de confusão:
[[9489  598]
 [ 279 1363]]
Tempo de treino: 0.3060 segundos
Tempo de predição (validação): 0.0070 segundos
Submission saved on 'submission.csv'
Tempo para processar teste: 0.1350 segundos
[XGBoost] Tempo de execução: 0.73 segundos
```
- Melhorou o recall porem pirou outras metricas talvez mais importantes como a precisao e o f1-score, levando um aumento do numero de falsos negativos
- ISto aconteceu pois o modelo dá mais importância à classe minoritária (no teu caso, o loan_status = 1). 


## Uso de Feature engenering
```
Exatidão (validação): 0.9499531076818143

Classification report:
               precision    recall  f1-score   support

           0       0.96      0.99      0.97     10087
           1       0.90      0.72      0.80      1642

    accuracy                           0.95     11729
   macro avg       0.93      0.86      0.89     11729
weighted avg       0.95      0.95      0.95     11729

Matriz de confusão:
[[9953  134]
 [ 453 1189]]
Tempo de treino: 0.3670 segundos
Tempo de predição (validação): 0.0080 segundos
Submission saved on 'submission.csv'
Tempo para processar teste: 0.1570 segundos
[XGBoost] Tempo de execução: 0.83 segundos
```


## Interpretação
- O modelo base é forte para a classe majoritária, mas perde alguns positivos.

- Equilibrio melhora o recall da classe minoritária, mas prejudica precisão e f1-score, aumentando falsos positivos.


- Feature engineering ajudou pouco, mantendo resultados similares.

- Dependendo do objetivo, pode-se escolher entre maior precisão (modelo base) ou maior recall (Equilibrio).