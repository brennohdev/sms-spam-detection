from sklearn.metrics import classification_report, confusion_matrix

class Evaluator:
    def execute(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        
        report = classification_report(y_test, predictions, output_dict=True)
        matrix = confusion_matrix(y_test, predictions)
        
        print("\n Classification Report:")
        print(classification_report(y_test, predictions))
        
        return {
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
            "f1": report['weighted avg']['f1-score'],
            "confusion_matrix": matrix
        }