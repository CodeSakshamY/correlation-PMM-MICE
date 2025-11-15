"""
Predictive Mean Matching (PMM) Imputer Module
Implements PMM algorithm for semi-parametric imputation
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class PMMImputer:
    """
    Predictive Mean Matching (PMM) imputer.

    PMM is a semi-parametric imputation method that:
    1. Fits a prediction model on observed data
    2. Predicts values for missing data
    3. Finds observed values with similar predictions
    4. Randomly selects from these "donor" values

    This preserves the distribution of the original data better than
    simple regression imputation.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        model_type: str = 'linear',
        random_state: Optional[int] = None
    ):
        """
        Initialize the PMM imputer.

        Parameters:
        -----------
        n_neighbors : int, default=5
            Number of nearest neighbors to consider for donor pool
        model_type : str, default='linear'
            Type of prediction model: 'linear', 'bayesian', or 'rf' (random forest)
        random_state : int, optional
            Random state for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

        # Initialize the prediction model
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'bayesian':
            self.model = BayesianRidge()
        elif model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.rng = np.random.RandomState(random_state)

    def fit_transform(
        self,
        data: pd.DataFrame,
        target_column: str,
        predictor_columns: List[str],
        predictor_weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Impute missing values in the target column using enhanced PMM.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset
        target_column : str
            Column to impute
        predictor_columns : List[str]
            Columns to use as predictors
        predictor_weights : Dict[str, float], optional
            Weights for each predictor (correlation strengths)

        Returns:
        --------
        imputed_values : np.ndarray
            The imputed column (complete, no missing values)
        """
        # Separate observed and missing data
        target = data[target_column]
        predictors = data[predictor_columns]

        # Handle case where predictors might have missing values
        # For now, we'll only use complete cases in predictors
        complete_mask = ~predictors.isnull().any(axis=1)
        observed_mask = ~target.isnull() & complete_mask
        missing_mask = target.isnull() & complete_mask

        if missing_mask.sum() == 0:
            # No missing values to impute
            return target.values

        if observed_mask.sum() < self.n_neighbors:
            # Not enough observed data for PMM, fall back to mean imputation
            mean_value = target[observed_mask].mean()
            result = target.copy()
            result[missing_mask] = mean_value
            return result.values

        # Get observed and missing predictor matrices
        X_observed = predictors[observed_mask]
        X_missing = predictors[missing_mask]
        y_observed = target[observed_mask].values

        # Create weight vector for predictors
        if predictor_weights:
            weight_vector = np.array([predictor_weights.get(col, 1.0) for col in predictor_columns])
            # Normalize to reasonable scale
            weight_vector = weight_vector / weight_vector.sum() * len(weight_vector)
        else:
            weight_vector = np.ones(len(predictor_columns))

        # Scale the predictors with weights
        X_observed_scaled = self.scaler.fit_transform(X_observed)
        X_missing_scaled = self.scaler.transform(X_missing)

        # Apply weights to scaled predictors
        X_observed_weighted = X_observed_scaled * weight_vector
        X_missing_weighted = X_missing_scaled * weight_vector

        # Fit the prediction model
        self.model.fit(X_observed_weighted, y_observed)

        # Predict for both observed and missing
        y_observed_pred = self.model.predict(X_observed_weighted)
        y_missing_pred = self.model.predict(X_missing_weighted)

        # Calculate residual standard deviation for uncertainty
        residuals = y_observed - y_observed_pred
        residual_std = np.std(residuals) if len(residuals) > 0 else 0

        # Enhanced stochastic component for PMM
        # Use bootstrap-style resampling for added uncertainty
        if residual_std > 0:
            # Draw from residual distribution with replacement (bootstrap)
            noise = self.rng.normal(0, residual_std * 0.3, len(y_missing_pred))
            y_missing_pred_noisy = y_missing_pred + noise
        else:
            y_missing_pred_noisy = y_missing_pred

        # For each missing value, find donors and select one
        imputed_values = np.zeros(missing_mask.sum())

        for i, (pred_value, x_missing) in enumerate(zip(y_missing_pred_noisy, X_missing_weighted)):
            # Enhanced distance calculation
            # 1. Distance in prediction space (primary criterion)
            pred_distances = np.abs(y_observed_pred - pred_value)

            # 2. Distance in predictor space (secondary criterion)
            predictor_distances = np.sqrt(np.sum((X_observed_weighted - x_missing) ** 2, axis=1))

            # Robust normalization using MAD (median absolute deviation)
            pred_median = np.median(pred_distances)
            pred_mad = np.median(np.abs(pred_distances - pred_median))
            pred_dist_norm = pred_distances / (1.4826 * pred_mad + 1e-10)

            predictor_median = np.median(predictor_distances)
            predictor_mad = np.median(np.abs(predictor_distances - predictor_median))
            predictor_dist_norm = predictor_distances / (1.4826 * predictor_mad + 1e-10)

            # Combined distance with emphasis on prediction matching
            combined_distances = 0.8 * pred_dist_norm + 0.2 * predictor_dist_norm

            # Find k nearest neighbors
            n_donors = min(self.n_neighbors, len(combined_distances))
            nearest_indices = np.argpartition(combined_distances, n_donors - 1)[:n_donors]

            # Enhanced donor selection with temperature-based sampling
            # This promotes diversity while still favoring closer donors
            donor_distances = combined_distances[nearest_indices]

            # Convert to similarity scores with temperature parameter
            temperature = 0.5  # Lower = more exploitation, higher = more exploration
            similarity = np.exp(-donor_distances / temperature)
            donor_weights = similarity / similarity.sum()

            # Randomly select from donor pool with weighted probability
            donor_idx = self.rng.choice(nearest_indices, p=donor_weights)
            imputed_values[i] = y_observed[donor_idx]

        # Create result array
        result = target.copy()
        result[missing_mask] = imputed_values

        return result.values

    def impute_column(
        self,
        data: pd.DataFrame,
        target_column: str,
        predictor_columns: Optional[List[str]] = None,
        predictor_weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Convenience method to impute a single column and return as Series.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset
        target_column : str
            Column to impute
        predictor_columns : List[str], optional
            Columns to use as predictors. If None, uses all other columns.
        predictor_weights : Dict[str, float], optional
            Weights for each predictor (correlation strengths)

        Returns:
        --------
        imputed_column : pd.Series
            The imputed column
        """
        if predictor_columns is None:
            predictor_columns = [col for col in data.columns if col != target_column]

        imputed_values = self.fit_transform(data, target_column, predictor_columns, predictor_weights)
        return pd.Series(imputed_values, index=data.index, name=target_column)


class AdaptivePMMImputer(PMMImputer):
    """
    Adaptive PMM imputer that adjusts n_neighbors based on data availability.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        min_neighbors: int = 3,
        model_type: str = 'linear',
        random_state: Optional[int] = None
    ):
        """
        Initialize adaptive PMM imputer.

        Parameters:
        -----------
        n_neighbors : int, default=5
            Target number of nearest neighbors
        min_neighbors : int, default=3
            Minimum number of neighbors to use
        model_type : str, default='linear'
            Type of prediction model
        random_state : int, optional
            Random state for reproducibility
        """
        super().__init__(n_neighbors, model_type, random_state)
        self.min_neighbors = min_neighbors

    def fit_transform(
        self,
        data: pd.DataFrame,
        target_column: str,
        predictor_columns: List[str],
        predictor_weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Impute with adaptive neighbor selection.
        """
        target = data[target_column]
        predictors = data[predictor_columns]

        complete_mask = ~predictors.isnull().any(axis=1)
        observed_mask = ~target.isnull() & complete_mask
        n_observed = observed_mask.sum()

        # Adapt n_neighbors based on available data
        if n_observed < self.min_neighbors:
            # Fall back to mean imputation
            mean_value = target[observed_mask].mean()
            result = target.copy()
            result[target.isnull()] = mean_value
            return result.values

        # Adjust n_neighbors to available data
        original_n = self.n_neighbors
        self.n_neighbors = min(self.n_neighbors, max(self.min_neighbors, n_observed // 3))

        # Call parent fit_transform
        result = super().fit_transform(data, target_column, predictor_columns, predictor_weights)

        # Restore original n_neighbors
        self.n_neighbors = original_n

        return result
