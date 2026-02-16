from omegamp.classifiers import AMPClassifier, HemolyticClassifier
from omegamp.constants import CLASSIFIER_MODELS

from typing import Literal
import numpy as np


def predict(
    sequences: list[str],
    *,
    checkpoint: Literal["broad-classifier"] = "broad-classifier",
    batch_size: int = 512,
    predict_proba: bool = True,
) -> np.ndarray:
    if checkpoint not in CLASSIFIER_MODELS:
        raise ValueError(
            f"Classifier {checkpoint} not found. Available classifiers: {', '.join(CLASSIFIER_MODELS.keys())}"
        )

    model_path = CLASSIFIER_MODELS[checkpoint]

    model = (
        AMPClassifier(model_path=model_path)
        if checkpoint != "hemolytic-classifier"
        else HemolyticClassifier(model_path=model_path)
    )
    model.eval()

    # Process sequences in batches
    all_predictions = []
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i : i + batch_size]

        predictions = (
            model.predict_proba(batch_sequences)
            if predict_proba
            else model(batch_sequences)
        )

        all_predictions.extend(predictions)

    return np.array(all_predictions)


if __name__ == "__main__":
    prediction = predict(["KKK", "RRR"])
    print(prediction)
