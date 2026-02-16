import torch.nn as nn
import xgboost as xgb
import pandas as pd

from .sequence_properties import (
    calculate_physchem_prop,
    calculate_aa_frequency,
    calculate_positional_encodings,
)


class PeptideClassifier(nn.Module):
    def __init__(self, model_path):
        super().__init__()

        if model_path is not None:
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
        else:
            self.model = xgb.XGBClassifier(
                eval_metric="logloss", early_stopping_rounds=50, n_estimators=5000
            )
        self.model.n_classes_ = 2
        self.decision_threshold = 0.5

    def get_input_features(self, sequences):
        """To be implemented by child classes"""
        raise NotImplementedError

    def train_classifier(
        self,
        input_features,
        labels,
        weight_balancing="balanced_with_adjustment_for_high_quality",
        mask_high_quality_idxs=[],
        return_feature_importances=False,
        verbose=True,
        objective="focal_loss",
    ):
        """To be implemented by child classes"""
        raise NotImplementedError

    def forward(self, sequences):
        input = self.get_input_features(sequences)
        probas = self.model.predict_proba(input)[:, 1]
        return (probas >= self.decision_threshold).astype(int)

    def predict_from_features(self, input_features, proba=False):
        probas = self.model.predict_proba(input_features)[:, 1]
        if proba:
            return probas
        return (probas >= self.decision_threshold).astype(int)

    def predict_proba(self, sequences):
        input = self.get_input_features(sequences)
        return self.model.predict_proba(input)[:, 1]

    def save(self, path):
        self.model.save_model(path)


class AMPClassifier(PeptideClassifier):
    def get_input_features(self, sequences):
        positional_encodings = pd.DataFrame(calculate_positional_encodings(sequences))
        properties = pd.DataFrame(calculate_physchem_prop(sequences, all_scales=True))
        frequencies = pd.DataFrame(calculate_aa_frequency(sequences))
        return pd.concat([properties, frequencies, positional_encodings], axis=1)


class HemolyticClassifier(PeptideClassifier):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.decision_threshold = 0.5

    def get_input_features(self, sequences):
        positional_encodings = pd.DataFrame(calculate_positional_encodings(sequences))
        properties = pd.DataFrame(calculate_physchem_prop(sequences, all_scales=True))
        frequencies = pd.DataFrame(calculate_aa_frequency(sequences))
        return pd.concat([properties, frequencies, positional_encodings], axis=1)
