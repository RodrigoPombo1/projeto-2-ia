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


## logisticRegression:
## Treino:
- LogisticRegression(max_iter=1000, random_state=42)
- Resultados:
```
Acurácia: 0.911842441810896
              precision    recall  f1-score   support

           0       0.93      0.97      0.95     10087
           1       0.77      0.53      0.63      1642

    accuracy                           0.91     11729
   macro avg       0.85      0.75      0.79     11729
weighted avg       0.90      0.91      0.90     11729
```
- Conclusao:
- Bons resultados para um modelo tão simples porem existe um possivel fator problematico que é ataxa de recall para load aproval 1 que é de 0,53 significa que o modelo só identifica 53% dos clientes problemáticos.
- Oque pode ser causado pelo um maior numero de 0 do que 1
- Solução: 
    - usar tecnicas para equilibrar os dados
    - Desta forma adicionei uma flag ao preprocess_train **balance_classes** quando true calcula o class_weight que depois pode ser passado para o LogisticRegression
## Treino com balance_classes:

- Resultados:
```
Acurácia: 0.8399693068462785
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
- Smote permite criar amostrar sintetica para balanciar os dados
- Resultados:
```
Acurácia: 0.8399693068462785
              precision    recall  f1-score   support

           0       0.97      0.84      0.90     10087
           1       0.46      0.83      0.59      1642

    accuracy                           0.84     11729
   macro avg       0.71      0.84      0.75     11729
weighted avg       0.90      0.84      0.86     11729
```
- Indica que o uso de smote tem um resultado identico do balance_classes

## Interpretação
-  SMOTE  está a funcionar: aumentou bastante o recall da classe minoritária (classe 1).

- Class weights também estão a ajudar o modelo a tratar a classe 1 com mais importância.

- O teste (sem nada) obtém maior acurácia mas ignora a classe 1, o que é problematico em problemas de fraude ou risco.