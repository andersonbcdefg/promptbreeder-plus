import numpy as np
from sklearn.linear_model import PassiveAggressiveRegressor
from embeddings.onnx_backend import ONNXEmbeddingModel
from evolution import Unit

class ScoringModel:
    def __init__(self, embedding_model_name: str = "bge-micro-v2"):
        self.model = PassiveAggressiveRegressor(average=True)
        self.embedding_model = ONNXEmbeddingModel(model_name=embedding_model_name)

    def update(self, units: list[Unit]):
        """Update the scoring model with the given units"""
        prompts = [u.task_prompt for u in units]
        X = self.embedding_model.embed_batch(prompts)
        y = np.array([u.fitness for u in units])
        # first, measure prediction error (mae) if model has been trained
        try:
            preds = self.model.predict(X)
            mae = np.mean(np.abs(preds - y))
            print(f"Heuristic model MAE on current batch: {mae:.3f}")
        except Exception as e:
            print(f"Heuristic model not trained yet, skipping prediction error measurement.")
            mae = None
        # now, update the model
        self.model.partial_fit(X, y)

        return mae

    def predict(self, units: list[Unit]):
        """Predict the fitness of a task prompt"""
        prompts = [u.task_prompt for u in units]
        X = self.embedding_model.embed_batch(prompts)
        return self.model.predict(X)    

    