"""
Correlation Analyzer Module
Analyzes correlations between features to inform imputation strategy
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class CorrelationAnalyzer:
    """
    Analyzes correlations between features to determine optimal imputation strategies.
    Uses correlation coefficients to identify the most predictive features for each
    variable with missing values.
    """

    def __init__(self, correlation_threshold: float = 0.3):
        """
        Initialize the correlation analyzer.

        Parameters:
        -----------
        correlation_threshold : float, default=0.3
            Minimum absolute correlation coefficient to consider a feature
            as a potential predictor
        """
        self.correlation_threshold = correlation_threshold
        self.correlation_matrix = None
        self.predictor_sets = {}

    def fit(self, data: pd.DataFrame) -> 'CorrelationAnalyzer':
        """
        Compute correlation matrix and identify predictor sets.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to analyze

        Returns:
        --------
        self : CorrelationAnalyzer
            Fitted analyzer
        """
        # Compute correlation matrix
        self.correlation_matrix = data.corr(method='pearson')

        # For each column, identify highly correlated predictors
        for col in data.columns:
            correlations = self.correlation_matrix[col].drop(col)
            # Select features with correlation above threshold
            strong_correlates = correlations[
                abs(correlations) >= self.correlation_threshold
            ].sort_values(key=abs, ascending=False)

            self.predictor_sets[col] = list(strong_correlates.index)

        return self

    def get_predictors(self, target_column: str, max_predictors: int = None) -> List[str]:
        """
        Get the list of best predictor columns for a target column.

        Parameters:
        -----------
        target_column : str
            The column to find predictors for
        max_predictors : int, optional
            Maximum number of predictors to return

        Returns:
        --------
        predictors : List[str]
            List of predictor column names
        """
        if target_column not in self.predictor_sets:
            return []

        predictors = self.predictor_sets[target_column]

        if max_predictors is not None:
            predictors = predictors[:max_predictors]

        return predictors

    def get_correlation_strength(self, col1: str, col2: str) -> float:
        """
        Get the correlation coefficient between two columns.

        Parameters:
        -----------
        col1, col2 : str
            Column names

        Returns:
        --------
        correlation : float
            Pearson correlation coefficient
        """
        if self.correlation_matrix is None:
            raise ValueError("Analyzer not fitted. Call fit() first.")

        return self.correlation_matrix.loc[col1, col2]

    def get_imputation_order(self, columns_with_missing: List[str]) -> List[str]:
        """
        Determine optimal order for imputing columns based on correlations.
        Columns with more complete predictors should be imputed first.

        Parameters:
        -----------
        columns_with_missing : List[str]
            Columns that have missing values

        Returns:
        --------
        ordered_columns : List[str]
            Columns ordered by imputation priority
        """
        # Score each column by number of available predictors
        scores = {}
        for col in columns_with_missing:
            predictors = self.get_predictors(col)
            # Prefer columns that have many strong predictors
            scores[col] = len(predictors)

        # Sort by score (descending) - impute columns with fewer predictors first
        # This allows later imputations to benefit from earlier ones
        ordered = sorted(scores.items(), key=lambda x: x[1])

        return [col for col, _ in ordered]

    def visualize_correlations(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Create a heatmap visualization of the correlation matrix.

        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size for the plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=figsize)
            sns.heatmap(
                self.correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                fmt='.2f'
            )
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Visualization requires matplotlib and seaborn. Install with: pip install matplotlib seaborn")
