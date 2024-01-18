from Dados import *
from CPCV import *
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
import numpy as np
import os

class Modelo():
    def __init__(self,tipo_variavel,ativo_name) -> None:
        #Funcao que irá carregar os dados
        funcao_dados = function_Dados()
        funcao_dados.Carregar_dados(ativo_name=ativo_name,
                                    start='2002-01-01',
                                    end= '2023-12-01'
                                    )
        funcao_dados.Variaveis(tipo_variavel)
        #Definindo as observações de treino e teste.
        self.X_train, self.X_test, self.y_train , self.y_test = funcao_dados.train_test_split()
        self.dados = funcao_dados.data_og
    
    def gridsearch_model(self,modelo,nome_modelo):
        modelo.fit(self.X_train,self.y_train)
        results = modelo.cv_results_['mean_test_score']
        params = modelo.cv_results_['params']
        df_result = pd.DataFrame({'params':params,
                                'mean roc auc':results})
        # Salvando Resultados
        df_result.to_csv(f'./resultados_treino/GridSearch/{nome_modelo}.csv')

        # Salvando Modelo
        with open(f'./Modelos/{nome_modelo}.pkl','wb') as file:
            pickle.dump(modelo,file)
        self.modelo = modelo

    def model(self,modelo,nome_modelo,n_splits=5,n_test_splits=2):
        modelo.fit(self.X_train,self.y_train)
        #Aplicando validação cruzada 
        cv_purge = cpcv(n_splits=n_splits,n_test_splits=n_test_splits)
        score = cross_val_score(modelo,self.X_train,self.y_train,cv=cv_purge,scoring='roc_auc',n_jobs=-1)
        media_score = score.mean()

        #Obtendo Parâmetros do Modelo:
        pos_1 = str(modelo).find('(') + 1
        pos_2 = str(modelo).find(')')
        params = str(modelo)[pos_1:pos_2]

        df_result = pd.DataFrame({
                        'params':params,
                        'mean roc auc':media_score,
                        'scores': [score]
                    })
        # Salvando Resultados
        df_result.to_csv(f'./resultados_treino/CPCV/{nome_modelo}.csv')

        # Salvando Modelo
        with open(f'./Modelos/{nome_modelo}.pkl','wb') as file:
            pickle.dump(modelo,file)
        self.modelo = modelo
        self.nome_modelo = nome_modelo
    def relatorio_model(self):
        #Prevendo Y
        y_pred = self.modelo.predict(self.X_test)

        #Relatório Modelo:
        df_resultados = pd.DataFrame({
                                    'y_test': self.y_test.values,
                                    'y_pred': y_pred},
                                    index=self.y_test.index)
        
        dados = self.dados[['Adj Close']]

        #Calculo de retorno do Ibovespa em t
        dados['Retorno_t'] = (dados['Adj Close']/dados['Adj Close'].shift()) - 1

        #Merge Results e Dados
        merge = pd.merge(df_resultados,dados,on='Date')

        # Retorno Acumulado do Modelo:
        # Regra: Caso modelo indique alta em t+1, entrará comprado, caso contrário, o modelo não irá fazer nada ou venderá o ativo caso esteja comprado.
        merge['Retorno_Modelo'] = np.where(merge['y_pred']== 1, merge['Retorno_t'].shift(-1),0)
        merge['Retorno_Acumulado_Modelo'] = (1 + merge['Retorno_Modelo']).cumprod() - 1
        merge['Retorno_Acumulado_Mercado'] = (1 + merge['Retorno_t']).cumprod() - 1

        #Criar Pasta do Modelo:
        nome_pasta = f'./relatorios/{self.nome_modelo}'
        if not os.path.exists(nome_pasta):
            os.makedirs(nome_pasta)
        merge.to_csv(f'{nome_pasta}/previsoes.csv')
        return
