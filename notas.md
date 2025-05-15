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
- Smote:
    -   Gera novos exemplos sintéticos da classe minoritária com base na interpolação entre vizinhos próximos no espaço de atributos.

    -   Exemplo: se tens poucos exemplos com loan_status = 1, o SMOTE gera mais exemplos parecidos com esses, para equilibrar com a maioria (loan_status = 0).


## Feature engenering:
- Criar novas variáveis que combinam informação relevante para ajudar o modelo a detetar padrões mais facilmente — especialmente padrões relacionados com capacidade financeira e risco.
### Variaveis usadas:
- Trata valores 0 na coluna person_emp_length (anos de experiência):
    -   Substitui os valores 0 por a mediana da coluna.
    -   Isto evita divisões por zero em cálculos futuros.

- Cria a feature financial_burden:
    - Representa o custo total dos juros do empréstimo.

    - Pode indicar se a pessoa está a assumir um empréstimo muito pesado.
- Cria a feature income_per_year_emp:
    - Calcula o rendimento médio por ano de trabalho.
    - Pode mostrar a estabilidade ou progressão da carreira da pessoa.
    - O + 1e-5 evita divisão por zero.
- Cria a feature int_per_year_emp


## Calculo do tempo é com o treino +teste


## logisticRegression:
## Treino:
- LogisticRegression(max_iter=1000, random_state=42)
- Resultados:
```
exatidão : 0.911842441810896
              precision    recall  f1-score   support

           0       0.93      0.97      0.95     10087
           1       0.77      0.53      0.63      1642

    accuracy                           0.91     11729
   macro avg       0.85      0.75      0.79     11729
weighted avg       0.90      0.91      0.90     11729
```
- Tempo de execução: 0.41 segundos
- Conclusao:
- Bons resultados para um modelo tão simples porem existe um possivel fator problematico que é ataxa de recall para load aproval 1 que é de 0,53 significa que o modelo só identifica 53% dos clientes problemáticos.
- Oque pode ser causado pelo um maior numero de 0 do que 1
- Solução: 
    - usar tecnicas para equilibrar os dados

## Treino com balance_classes:

- Resultados:
```
exatidão : 0.8399693068462785
              precision    recall  f1-score   support

           0       0.97      0.84      0.90     10087
           1       0.46      0.83      0.59      1642

    accuracy                           0.84     11729
   macro avg       0.71      0.84      0.75     11729
weighted avg       0.90      0.84      0.86     11729
```
- Melhorou o recall porem pirou outras metricas talvez mais importantes como a precisao e o f1-score, levando um aumento do numero de falsos negativos
- ISto aconteceu pois o modelo dá mais importância à classe minoritária (no teu caso, o loan_status = 1). 

## Uso De smote:
- Resultados:
```
exatidão : 0.8399693068462785
              precision    recall  f1-score   support

           0       0.97      0.84      0.90     10087
           1       0.46      0.83      0.59      1642

    accuracy                           0.84     11729
   macro avg       0.71      0.84      0.75     11729
weighted avg       0.90      0.84      0.86     11729
```
-  O uso de smote tem um resultado identico do balance_classes

## Uso de Feature engenering
```
exatidão  (validação): 0.9143149458606872
              precision    recall  f1-score   support

           0       0.93      0.97      0.95     10087
           1       0.78      0.55      0.64      1642

    accuracy                           0.91     11729
   macro avg       0.85      0.76      0.80     11729
weighted avg       0.91      0.91      0.91     11729
```
- O recall subiu de 0.53 → 0.55

- O f1-score subiu de 0.63 → 0.64

- Isto significa que o modelo passou a identificar um pouco melhor quem não paga o empréstimo, sem perder qualidade nos outros caso

## Interpretação
-  SMOTE  está a funcionar: aumentou bastante o recall da classe minoritária (classe 1).

- Class weights também estão a ajudar o modelo a tratar a classe 1 com mais importância.

- O teste (sem nada) obtém maior exatidão  mas ignora a classe 1, o que é problematico em problemas de fraude ou risco.

## RandomForestClassifier:
## Treino:
- emodelUsed = RandomForestClassifier(n_estimators=100, random_state=42)
- Resultados:
```
exatidão  (validação): 0.951828800409242
              precision    recall  f1-score   support

           0       0.96      0.99      0.97     10087
           1       0.92      0.72      0.81      1642

    accuracy                           0.95     11729
   macro avg       0.94      0.86      0.89     11729
weighted avg       0.95      0.95      0.95     11729
```

- Tempo:5.31 segundos
- ##Conclusao:
- Random Forest sem SMOTE nem class_weight teve melhor desempenho em todos os aspetos em comparacao com o caso base logisticRegression .

- Foi especialmente superior na classe minoritária, onde o modelo de regressão logística teve um desempenho mais fraco (recall 0.53 vs 0.72).

- Não foi necessário balanceamento explícito (nem SMOTE, nem class_weight) para a Random Forest ter um bom desempenho — isso é esperado, porque Random Forest lida melhor com desbalanceamento ao não ser um modelo linear e ao fazer bootstraping com vários subconjuntos dos dados.

## RandomForestClassifier com class_weight:

- Resultados:
```
exatidão  (validação): 0.9515730241282292
              precision    recall  f1-score   support

           0       0.95      0.99      0.97     10087
           1       0.92      0.71      0.80      1642

    accuracy                           0.95     11729
   macro avg       0.94      0.85      0.89     11729
weighted avg       0.95      0.95      0.95     11729

```

- ## Conclusao:
- Random Forest com class_weight não tem qualquer melhoria mas ja era esperado   porque Random Forest lida melhor com desbalanceamento ao não ser um modelo linear e ao fazer bootstraping com vários subconjuntos dos dados.

## RandomForestClassifier com Smote:
- Este teste não é necessario pois como nao é influenciado pelo class_weight o smote nao tera efeito

## RandomForestClassifier com Feature engineering:

```
exatidão (validação): 0.9472248273510103
              precision    recall  f1-score   support

           0       0.95      0.99      0.97     10087
           1       0.89      0.71      0.79      1642

    accuracy                           0.95     11729
   macro avg       0.92      0.85      0.88     11729
weighted avg       0.95      0.95      0.94     11729
```
- Estas features podema acrescentar alguma redundancia desbalancear a árvore ao fazer splits em features artificiais menos relevantes