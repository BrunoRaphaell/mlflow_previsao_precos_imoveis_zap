#Ignorar avisos de atualização, etc
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import mlflow
import sys

from urllib.error import HTTPError
from pprint import pp, pprint
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from yellowbrick.regressor import ResidualsPlot, PredictionError


np.random.seed(42)
# mlflow.set_experiment(experiment_name='mlflow_previsao_precos_imoveis_zap')

tags_initial = {
    "Projeto": "mlflow preço de casas",
    "team": "Data Science",
    "dataset": "dados_OneHotEncoder.csv"
}

def plot_residuals(model, X_train, y_train, X_test, y_test):
    visualizer_re = ResidualsPlot(model, hist=False, qqplot=True)
    visualizer_re.fit(X_train, y_train)
    visualizer_re.score(X_test, y_test)
    return visualizer_re


def plot_prediction_error(model, X_train, y_train, X_test, y_test):
    visualizer_pe = PredictionError(model, qqplot=False)
    visualizer_pe.fit(X_train, y_train)  
    visualizer_pe.score(X_test, y_test)  
    return visualizer_pe
    
def fetch_logged_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


def main():

    csv_url = "https://raw.githubusercontent.com/BrunoRaphaell/previsao_precos_imoveis_zap/master/data/processed/dados_OneHotEncoder.csv"

    try:
        df = pd.read_csv(csv_url)
    except HTTPError as http:
        print("Há algum erro na url. Verifique se foi digitada corretamente.")

    X, y = df.drop('price', axis=1), df['price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    max_depth = float(sys.argv[1]) if len(sys.argv) > 1 else 8
    learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

    with mlflow.start_run(run_name='GradientBoostingRegressor', tags=tags_initial) as run:
        mlflow.sklearn.autolog()

        model = GradientBoostingRegressor(
            max_depth=max_depth, learning_rate=learning_rate, min_samples_split=4, n_estimators=200)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
        # Salvando o gráfico de resíduos e qqplot:
        residuals = plot_residuals(model, X_train, y_train, X_test, y_test)
        temp_name_residuals = "residuals_" + str(5) + "_" + str(0.1).replace('.', '-') + ".png"
        residuals.show(outpath=temp_name_residuals)
        mlflow.log_artifact(temp_name_residuals, "regression graphs")
        
        # Salvando o gráfico de erro de predição:
        prediction_error = plot_prediction_error(model, X_train, y_train, X_test, y_test)
        temp_name_prediction_error = "prediction_error_" + str(5) + "_" + str(0.1).replace('.', '-') + ".png"
        prediction_error.show(outpath=temp_name_prediction_error)
        mlflow.log_artifact(temp_name_prediction_error, "regression graphs")
        
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
        
        print("Run ID:", run.info.run_id)
        print("GradientBoostingRegressor(alpha=%f, l1_ratio=%f):" %
              (max_depth, learning_rate), end="\n\n")
        
        print("============Métricas============")
        pprint(metrics)
        
        print("============Parâmetros============")
        pprint(params)
        
        print("============Tags============")
        pprint(tags)
        
        print("============Artefatos============")
        pprint(artifacts)
        
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(local_path='./train.py')


if __name__ == '__main__':
    main()
