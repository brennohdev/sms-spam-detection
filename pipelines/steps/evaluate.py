import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report

class Evaluator:
    def execute(self, model, X_test, y_test, params: dict):
        mlflow.set_experiment("SMS Spam Detection - Tournament")
        
        with mlflow.start_run():
            predictions = model.predict(X_test)
            report = classification_report(y_test, predictions, output_dict=True)
            
            mlflow.log_params(params)
            mlflow.log_metric("precision", report['weighted avg']['precision'])
            mlflow.log_metric("recall", report['weighted avg']['recall'])
            
            mlflow.sklearn.log_model(model, "model")
            
            print(" [MLflow] Run registrada com sucesso!")
            return report