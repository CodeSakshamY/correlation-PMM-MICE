"""
Predictive Mean Matching (PMM) Imputer Module
Implements PMM algorithm for semi-parametric imputation
"""

import numpy as np
import pandas as pd
from typing import List, Optional
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
        predictor_columns: List[str]
    ) -> np.ndarray:
        """
        Impute missing values in the target column using PMM.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset
        target_column : str
            Column to impute
        predictor_columns : List[str]
            Columns to use as predictors

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

        # Scale the predictors
        X_observed_scaled = self.scaler.fit_transform(X_observed)
        X_missing_scaled = self.scaler.transform(X_missing)

        # Fit the prediction model
        self.model.fit(X_observed_scaled, y_observed)

        # Predict for both observed and missing
        y_observed_pred = self.model.predict(X_observed_scaled)
        y_missing_pred = self.model.predict(X_missing_scaled)

        # For each missing value, find donors and select one
        imputed_values = np.zeros(missing_mask.sum())

        for i, pred_value in enumerate(y_missing_pred):
            # Find the k nearest observed predictions
            distances = np.abs(y_observed_pred - pred_value)
            nearest_indices = np.argpartition(distances, min(self.n_neighbors, len(distances) - 1))[:self.n_neighbors]

            # Randomly select one donor from the nearest neighbors
            donor_idx = self.rng.choice(nearest_indices)
            imputed_values[i] = y_observed[donor_idx]

        # Create result array
        result = target.copy()
        result[missing_mask] = imputed_values

        return result.values

    def impute_column(
        self,
        data: pd.DataFrame,
        target_column: str,
        predictor_columns: Optional[List[str]] = None
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

        Returns:
        --------
        imputed_column : pd.Series
            The imputed column
        """
        if predictor_columns is None:
            predictor_columns = [col for col in data.columns if col != target_column]

        imputed_values = self.fit_transform(data, target_column, predictor_columns)
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
        predictor_columns: List[str]
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
        result = super().fit_transform(data, target_column, predictor_columns)

        # Restore original n_neighbors
        self.n_neighbors = original_n

        return result
