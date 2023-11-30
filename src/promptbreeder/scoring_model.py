from typing import Optional

import numpy as np
from sklearn.linear_model import PassiveAggressiveRegressor

from .embeddings.onnx_backend import ONNXEmbeddingModel
from .evolution import Unit


class ScoringModel:
    def __init__(self, diversity_factor = 0.5, embedding_model_name: str = "bge-micro-v2"):
        self.generation = 0
        self.diversity_factor = diversity_factor
        self.model = PassiveAggressiveRegressor(
            epsilon=0.05,
            # early_stopping=True,
            warm_start=True
        )
        self.embedding_model = ONNXEmbeddingModel(model_name=embedding_model_name)
        self.Xs = None
        self.ys = None

    def fit(self):
        """Fit the scoring model to the current data"""
        if self.Xs is None or self.ys is None:
            raise ValueError("No data to fit to!")
        elif len(self.ys) < 100:
            self.model.partial_fit(self.Xs, self.ys)
        else:
            self.model.fit(self.Xs, self.ys)

    def update(self, units: list[Unit], log_dir: Optional[str] = None, status = None):
        """Update the scoring model with the given units"""
        prompts = [u.task_prompt for u in units]
        status.update(f"Embedding {len(prompts)} prompts...")
        X = self.embedding_model.embed_batch(prompts)
        y = np.array([u.fitness for u in units])
        # first, measure metrics if model isn't brand new
        if self.Xs is not None:
            preds = self.model.predict(X)
            mae = np.mean(np.abs(preds - y))
            r = np.corrcoef(preds, y)[0, 1]
            r_squared = r ** 2
            metrics = {
                "mae": mae,
                "r": r,
                "r_squared": r_squared,
            }
            # scatterplot
            # if log_dir is not None:
            #     regression_line = np.poly1d(np.polyfit(preds, y, 1))
            #     if not os.path.exists(os.path.join(log_dir, "plots")):
            #         os.makedirs(os.path.join(log_dir, "plots"))
            #     plots_dir = os.path.join(log_dir, "plots")
            #     plt.scatter(preds, y, color="blue", alpha=0.5)
            #     plt.plot(preds, regression_line(preds), color="red")
            #     plt.xlabel("Predicted fitness")
            #     plt.ylabel("Actual fitness")
            #     plt.title(f"Scoring model performance (MAE: {mae:.3f}, R^2: {r_squared:.3f})")
            #     plt.savefig(f"{plots_dir}/scoring_model_gen_{self.generation}.png")
            #     plt.clf()
            #     plt.close()

        else:
            print("Heuristic model not trained yet, skipping metrics.")
            metrics = None
        # now, update the model
        status.update("Updating scoring model...")
        self.Xs = X if self.Xs is None else np.concatenate([self.Xs, X], axis=0)
        self.ys = y if self.ys is None else np.concatenate([self.ys, y], axis=0)
        self.fit()
        self.generation += 1
        return metrics

    def predict(self, units: list[Unit], return_embeddings: bool = False):
        """Predict the fitness of a list of units"""
        prompts = [u.task_prompt for u in units]
        X = self.embedding_model.embed_batch(prompts)
        if return_embeddings:
            return self.model.predict(X), X
        return self.model.predict(X)
    
    def select(self, units: list[Unit], num_to_select: int):
        """Select the best units from the given list, based on predicted fitness & diversity"""
        print(f"selecting {num_to_select} units from {len(units)} units")
        fitness_scores, embeddings = self.predict(units, return_embeddings=True)
        fitness_scores = np.clip(fitness_scores, 0, 1)

        # cosine similarity matrix
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(similarity_matrix, 0)

        # now, compute diversity scores (normalized to between 0 and 1)
        diversity_scores = np.add(-np.mean(similarity_matrix, axis=1), 1) / 2
        weights = (1 - self.diversity_factor) * fitness_scores  + self.diversity_factor * diversity_scores
        weights = weights / np.sum(weights)

        # total weight
        selected_units = np.random.choice(
            units,
            size=num_to_select,
            replace=False,
            p=weights,
        )
        
        return list(selected_units)
    