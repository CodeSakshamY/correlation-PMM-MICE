"""
Correlation Analyzer Module
Analyzes correlations between features to inform imputation strategy
Enhanced with multiple correlation methods and weighted scoring
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy import stats


class CorrelationAnalyzer:
    """
    Analyzes correlations between features to determine optimal imputation strategies.
    Uses multiple correlation methods (Pearson, Spearman, Kendall) with weighted scoring
    to identify the most predictive features for each variable with missing values.
    """

    def __init__(self, correlation_threshold: float = 0.3, use_mixed_correlations: bool = True):
        """
        Initialize the correlation analyzer.

        Parameters:
        -----------
        correlation_threshold : float, default=0.3
            Minimum absolute correlation coefficient to consider a feature
            as a potential predictor
        use_mixed_correlations : bool, default=True
            If True, combines Pearson, Spearman, and Kendall correlations
            for more robust correlation estimation
        """
        self.correlation_threshold = correlation_threshold
        self.use_mixed_correlations = use_mixed_correlations
        self.correlation_matrix = None
        self.pearson_matrix = None
        self.spearman_matrix = None
        self.kendall_matrix = None
        self.combined_matrix = None
        self.predictor_sets = {}
        self.correlation_weights = {}

    def fit(self, data: pd.DataFrame) -> 'CorrelationAnalyzer':
        """
        Compute correlation matrix and identify predictor sets.
        Uses multiple correlation methods for robust estimation.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to analyze

        Returns:
        --------
        self : CorrelationAnalyzer
            Fitted analyzer
        """
        if self.use_mixed_correlations:
            # Compute multiple correlation matrices
            self.pearson_matrix = data.corr(method='pearson')
            self.spearman_matrix = data.corr(method='spearman')

            # Kendall can be slow for large datasets, so we compute it selectively
            try:
                self.kendall_matrix = data.corr(method='kendall')
            except:
                self.kendall_matrix = None

            # Combine correlations with weighted average
            # Pearson: 50%, Spearman: 30%, Kendall: 20%
            if self.kendall_matrix is not None:
                self.combined_matrix = (
                    0.5 * self.pearson_matrix.abs() +
                    0.3 * self.spearman_matrix.abs() +
                    0.2 * self.kendall_matrix.abs()
                )
            else:
                # Without Kendall: Pearson 60%, Spearman 40%
                self.combined_matrix = (
                    0.6 * self.pearson_matrix.abs() +
                    0.4 * self.spearman_matrix.abs()
                )

            # Use Pearson for the main correlation matrix (for signed correlations)
            self.correlation_matrix = self.pearson_matrix
        else:
            # Simple Pearson correlation
            self.correlation_matrix = data.corr(method='pearson')
            self.combined_matrix = self.correlation_matrix.abs()

        # For each column, identify highly correlated predictors
        for col in data.columns:
            if self.use_mixed_correlations:
                # Use combined matrix for selection
                correlations_abs = self.combined_matrix[col].drop(col)
                correlations_signed = self.correlation_matrix[col].drop(col)
            else:
                correlations_abs = self.correlation_matrix[col].abs().drop(col)
                correlations_signed = self.correlation_matrix[col].drop(col)

            # Select features with correlation above threshold
            strong_mask = correlations_abs >= self.correlation_threshold
            strong_correlates = correlations_abs[strong_mask].sort_values(ascending=False)

            # Store predictor list
            self.predictor_sets[col] = list(strong_correlates.index)

            # Store correlation weights for each predictor (normalized)
            if len(strong_correlates) > 0:
                # Use signed correlations for weights
                weights = {}
                for pred in strong_correlates.index:
                    # Weight is the combined correlation strength
                    weights[pred] = strong_correlates[pred]

                # Normalize weights to sum to 1
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v / total_weight for k, v in weights.items()}

                self.correlation_weights[col] = weights
            else:
                self.correlation_weights[col] = {}

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

    def get_predictor_weights(self, target_column: str) -> Dict[str, float]:
        """
        Get normalized correlation weights for predictors of a target column.

        Parameters:
        -----------
        target_column : str
            The column to get predictor weights for

        Returns:
        --------
        weights : Dict[str, float]
            Dictionary mapping predictor names to their normalized weights
        """
        if target_column not in self.correlation_weights:
            return {}
        return self.correlation_weights[target_column]

    def get_imputation_order(self, columns_with_missing: List[str]) -> List[str]:
        """
        Determine optimal order for imputing columns based on correlations.
        Uses weighted scoring based on both quantity and quality of predictors.

        Parameters:
        -----------
        columns_with_missing : List[str]
            Columns that have missing values

        Returns:
        --------
        ordered_columns : List[str]
            Columns ordered by imputation priority
        """
        # Score each column by quality and quantity of available predictors
        scores = {}
        for col in columns_with_missing:
            predictors = self.get_predictors(col)
            weights = self.get_predictor_weights(col)

            if predictors:
                # Score based on both count and strength of correlations
                # Higher score = better predictors = impute later (use as predictor for others first)
                avg_weight = sum(weights.values()) / len(weights) if weights else 0
                score = len(predictors) * (1 + avg_weight)  # Weighted by average correlation strength
            else:
                score = 0

            scores[col] = score

        # Sort by score (ascending) - impute columns with weaker predictors first
        # This allows later imputations to benefit from strongly predicted columns
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
