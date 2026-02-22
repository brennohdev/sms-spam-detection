import joblib
import os
from sklearn.model_selection import train_test_split

from pipelines.steps.ingest import DataIngestor
from pipelines.steps.train import ModelTrainer
from pipelines.steps.evaluate import Evaluator

class MasterTrainingPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.ingestor = DataIngestor()
        self.trainer = ModelTrainer()
        self.evaluator = Evaluator()

    def run(self):
        print("---  MASTERPIECE PIPELINE STARTED ---")
        
        # 1. INGESTION
        raw_data = self.ingestor.execute(self.data_path)
        
        # 2. SPLITTING (The 'Prep' work)
        X_train, X_test, y_train, y_test = train_test_split(
            raw_data['text'], 
            raw_data['label'], 
            test_size=0.2, 
            random_state=42, 
            stratify=raw_data['label']
        )
        
        # 3. TRAINING
        params = {"alpha": 0.01, "ngram_range": (1, 2)}
        trained_model = self.trainer.execute(X_train, y_train, params)
        
        # 4. EVALUATION
        metrics = self.evaluator.execute(trained_model, X_test, y_test, params)
        
        # 5. PERSISTENCE (The 'Warehouse')
        os.makedirs("models", exist_ok=True)
        save_path = "models/spam_detector_v1.joblib"
        joblib.dump(trained_model, save_path)
        
        print(f"---  SUCCESS: Model saved at {save_path} ---")

if __name__ == "__main__":
    pipeline = MasterTrainingPipeline(data_path="data/spam.csv")
    pipeline.run()