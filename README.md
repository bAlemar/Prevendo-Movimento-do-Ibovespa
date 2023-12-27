# Previsão do Movimento do Índice Ibovespa

## Descrição
Esse código foi utilizado para o meu trabalho de conclusão de curso (Economia pela Universidade Federal Fluminense (UFF)).
- **Objetivo principal (Acadêmico):** Prever movimento do Ibovespa com certa acurácia (> 50%).
- **Objetivo Secundário (Prático):** Ajudar na tomada de decisão de investimentos.
- **Futuros Trabalhos:** Criação de um sistema de trading utilizando Apis das corretoras.

## Estrutura do Repositório
- **Modelos:** Contém os modelos de ML salvos em arquivo .pkl.
- **Relatórios:** Contém as previsões dos modelos (y_pred, Retorno_Modelo, etc) em um arquivo .csv.
- **Resultados_treino:** Contém resultados dos Modelos de GridSearch ou os resultados da Validação Cruzada(CPCV) para modelos sem otimização de parâmetros.
- **CPCV.py:** Script da validação cruzada personalizada que pode ser utilizada no sklearn.
- **Dados.py:** Extração/Tratamento/Processamento de dados.
- **Modelo.py:** Treinamento do modelo e geração de resultados.
- **requirements.txt:** Lista de bibliotecas e versões necessárias.
- **Teste.ipynb:** Notebook Jupyter para testar funções e gerar gráficos.
## Como Usar
- O arquivo Teste.ipynb tem um exemplo para rodar arquivo sem utilização de GridSearch. 
- Para utilização de GridSearch basta incrementar a função CPCV antes, por exemplo:
```bash
from Modelo import *
from Dados import *
from CPCV import *
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

funcao_modelo = Modelo('binária')
modelo = GridSearchCV(estimator=DecisionTreeClassifier()
                    ,cv=cpcv()
                    ,param_grid=params)
funcao_modelo.gridsearch_model(modelo,'GridSearchModel')
funcao_modelo.relatorio_model()
```

## Contato
https://linktr.ee/bernardoalemar


# Executar o Script em sua máquina local
## Pré-requisitos:

Antes de começar, certifique-se de ter o seguinte instalado em sua máquina:

- Python 3.10.12
- pip (gerenciador de pacotes Python)
- Git (ferramenta de controle de versão)

Uma vez que você tenha isso instalado, abra um terminal em sua máquina local e execute os seguintes comandos:

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/bAlemar/Prevendo-Movimento-do-Ibovespa.git

2. **Navegue até o diretório do repositório clonado:**
   ```bash
   cd Prevendo-Movimento-do-Ibovespa
 

3. **Crie um ambiente virtual:**
   ```bash
    python -m venv ambiente_virtual

4. **Ative o ambiente virtual:**

   **4.1 Linux**
   ```bash
    source ambiente_virtual/bin/activate
   ```
   **4.2 Windows**
   ```bash
    source ambiente_virtual\Scripts\activate

5. **Instale as Dependências:**
   ```bash
    pip install -r requeriments.txt

