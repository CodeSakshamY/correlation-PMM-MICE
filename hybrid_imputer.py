"""
Hybrid Imputer Module
Combines Correlation Analysis + PMM + MICE for advanced missing data imputation
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union
import warnings

from correlation_analyzer import CorrelationAnalyzer
from pmm_imputer import AdaptivePMMImputer


class HybridMICEImputer:
    """
    Hybrid imputation model combining:
    - Correlation Analysis: To identify optimal predictor sets
    - PMM (Predictive Mean Matching): For distribution-preserving imputation
    - MICE (Multivariate Imputation by Chained Equations): For iterative refinement

    This hybrid approach:
    1. Uses correlation analysis to determine which features best predict each missing variable
    2. Employs PMM to impute values while preserving data distribution
    3. Iterates using MICE framework to refine imputations based on newly imputed values
    """

    def __init__(
        self,
        n_iterations: int = 10,
        n_neighbors: int = 5,
        correlation_threshold: float = 0.3,
        max_predictors: int = 10,
        pmm_model_type: str = 'linear',
        convergence_threshold: float = 0.001,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize the Hybrid MICE imputer.

        Parameters:
        -----------
        n_iterations : int, default=10
            Maximum number of MICE iterations
        n_neighbors : int, default=5
            Number of neighbors for PMM
        correlation_threshold : float, default=0.3
            Minimum correlation to consider for predictor selection
        max_predictors : int, default=10
            Maximum number of predictors to use per variable
        pmm_model_type : str, default='linear'
            Model type for PMM: 'linear', 'bayesian', or 'rf'
        convergence_threshold : float, default=0.001
            Threshold for convergence detection
        random_state : int, optional
            Random state for reproducibility
        verbose : bool, default=False
            Whether to print progress information
        """
        self.n_iterations = n_iterations
        self.n_neighbors = n_neighbors
        self.correlation_threshold = correlation_threshold
        self.max_predictors = max_predictors
        self.pmm_model_type = pmm_model_type
        self.convergence_threshold = convergence_threshold
        self.random_state = random_state
        self.verbose = verbose

        # Components
        self.correlation_analyzer = CorrelationAnalyzer(
            correlation_threshold=correlation_threshold
        )
        self.imputer = AdaptivePMMImputer(
            n_neighbors=n_neighbors,
            model_type=pmm_model_type,
            random_state=random_state
        )

        # State
        self.missing_indicators = None
        self.columns_with_missing = []
        self.imputation_order = []
        self.convergence_history = []

    def _identify_missing_data(self, data: pd.DataFrame) -> None:
        """Identify columns with missing data and create missing indicators."""
        self.missing_indicators = data.isnull()
        self.columns_with_missing = [
            col for col in data.columns if self.missing_indicators[col].any()
        ]

        if self.verbose:
            print(f"Columns with missing data: {len(self.columns_with_missing)}")
            for col in self.columns_with_missing:
                n_missing = self.missing_indicators[col].sum()
                pct_missing = 100 * n_missing / len(data)
                print(f"  {col}: {n_missing} ({pct_missing:.2f}%)")

    def _initialize_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Initialize missing values with simple mean imputation.
        This provides a starting point for the MICE iterations.
        """
        imputed = data.copy()

        for col in self.columns_with_missing:
            if imputed[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Numerical: use mean
                mean_val = imputed[col].mean()
                imputed[col].fillna(mean_val, inplace=True)
            else:
                # Categorical: use mode
                mode_val = imputed[col].mode()[0] if len(imputed[col].mode()) > 0 else imputed[col].iloc[0]
                imputed[col].fillna(mode_val, inplace=True)

        return imputed

    def _compute_convergence_metric(
        self,
        data_current: pd.DataFrame,
        data_previous: pd.DataFrame
    ) -> float:
        """
        Compute convergence metric based on change in imputed values.
        """
        if data_previous is None:
            return float('inf')

        total_change = 0
        n_imputed = 0

        for col in self.columns_with_missing:
            missing_mask = self.missing_indicators[col]
            if missing_mask.any():
                current_vals = data_current.loc[missing_mask, col]
                previous_vals = data_previous.loc[missing_mask, col]

                # Normalize by standard deviation
                std = data_current[col].std()
                if std > 0:
                    change = np.mean(np.abs(current_vals - previous_vals) / std)
                else:
                    change = np.mean(np.abs(current_vals - previous_vals))

                total_change += change
                n_imputed += 1

        return total_change / n_imputed if n_imputed > 0 else 0

    def fit_transform(
        self,
        data: pd.DataFrame,
        columns_to_impute: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Impute missing values using the hybrid Correlation + PMM + MICE approach.

        Parameters:
        -----------
        data : pd.DataFrame
            Dataset with missing values
        columns_to_impute : List[str], optional
            Specific columns to impute. If None, imputes all columns with missing values.

        Returns:
        --------
        imputed_data : pd.DataFrame
            Dataset with imputed values
        """
        if data.isnull().sum().sum() == 0:
            if self.verbose:
                print("No missing data found. Returning original data.")
            return data.copy()

        # Identify missing data
        self._identify_missing_data(data)

        # Filter to requested columns
        if columns_to_impute is not None:
            self.columns_with_missing = [
                col for col in self.columns_with_missing if col in columns_to_impute
            ]

        if not self.columns_with_missing:
            if self.verbose:
                print("No columns to impute. Returning original data.")
            return data.copy()

        # Initialize with simple imputation
        imputed = self._initialize_imputation(data)

        # Fit correlation analyzer on complete cases first
        complete_data = data.dropna()
        if len(complete_data) > 0:
            self.correlation_analyzer.fit(complete_data)
        else:
            # If no complete cases, use initialized data
            self.correlation_analyzer.fit(imputed)

        # Determine imputation order based on correlations
        self.imputation_order = self.correlation_analyzer.get_imputation_order(
            self.columns_with_missing
        )

        if self.verbose:
            print(f"Imputation order: {self.imputation_order}")
            print(f"Starting MICE iterations (max {self.n_iterations})...")

        # MICE iterations
        previous_imputed = None
        self.convergence_history = []

        for iteration in range(self.n_iterations):
            if self.verbose:
                print(f"\nIteration {iteration + 1}/{self.n_iterations}")

            # Iterate through each column with missing values
            for col in self.imputation_order:
                # Get correlation-based predictors
                predictors = self.correlation_analyzer.get_predictors(
                    col,
                    max_predictors=self.max_predictors
                )

                # If no predictors found, use all other columns
                if not predictors:
                    predictors = [c for c in imputed.columns if c != col]

                # Filter out predictors that are all NaN or have too many NaNs
                valid_predictors = []
                for pred in predictors:
                    if imputed[pred].isnull().sum() < len(imputed) * 0.5:
                        valid_predictors.append(pred)

                if not valid_predictors:
                    if self.verbose:
                        print(f"  {col}: No valid predictors, skipping")
                    continue

                # Apply PMM imputation
                try:
                    # Create temporary dataset with only needed columns
                    temp_data = imputed[[col] + valid_predictors].copy()

                    # Impute the column
                    imputed_col = self.imputer.fit_transform(
                        temp_data,
                        col,
                        valid_predictors
                    )

                    # Update only the originally missing values
                    missing_mask = self.missing_indicators[col]
                    imputed.loc[missing_mask, col] = imputed_col[missing_mask]

                    if self.verbose:
                        print(f"  {col}: Imputed with {len(valid_predictors)} predictors")

                except Exception as e:
                    if self.verbose:
                        print(f"  {col}: Error during imputation - {str(e)}")
                    continue

            # Check convergence
            convergence = self._compute_convergence_metric(imputed, previous_imputed)
            self.convergence_history.append(convergence)

            if self.verbose:
                print(f"Convergence metric: {convergence:.6f}")

            if convergence < self.convergence_threshold:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

            previous_imputed = imputed.copy()

        if self.verbose:
            print("\nImputation complete!")
            print(f"Total iterations: {len(self.convergence_history)}")

        return imputed

    def fit(self, data: pd.DataFrame) -> 'HybridMICEImputer':
        """
        Fit the imputer (mainly for correlation analysis).

        Parameters:
        -----------
        data : pd.DataFrame
            Training data

        Returns:
        --------
        self : HybridMICEImputer
        """
        complete_data = data.dropna()
        if len(complete_data) > 0:
            self.correlation_analyzer.fit(complete_data)
        else:
            # If no complete cases, use all data
            self.correlation_analyzer.fit(data)

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted imputer.

        Parameters:
        -----------
        data : pd.DataFrame
            Data to impute

        Returns:
        --------
        imputed_data : pd.DataFrame
        """
        return self.fit_transform(data)

    def get_diagnostics(self) -> Dict[str, Union[List, pd.DataFrame]]:
        """
        Get diagnostic information about the imputation process.

        Returns:
        --------
        diagnostics : dict
            Dictionary containing diagnostic information
        """
        diagnostics = {
            'convergence_history': self.convergence_history,
            'imputation_order': self.imputation_order,
            'columns_with_missing': self.columns_with_missing,
            'correlation_matrix': self.correlation_analyzer.correlation_matrix,
            'predictor_sets': self.correlation_analyzer.predictor_sets
        }

        return diagnostics

    def plot_convergence(self):
        """
        Plot the convergence history.
        """
        try:
            import matplotlib.pyplot as plt

            if not self.convergence_history:
                print("No convergence history available. Run fit_transform first.")
                return

            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.convergence_history) + 1), self.convergence_history, marker='o')
            plt.axhline(y=self.convergence_threshold, color='r', linestyle='--', label='Convergence threshold')
            plt.xlabel('Iteration')
            plt.ylabel('Convergence Metric')
            plt.title('MICE Convergence History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Plotting requires matplotlib. Install with: pip install matplotlib")

    @staticmethod
    def load_data(
        file_path: str,
        sheet_name: Optional[Union[str, int]] = 0,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from CSV or Excel (.xlsx) file.

        Parameters:
        -----------
        file_path : str
            Path to the file (supports .csv, .xlsx, .xls)
        sheet_name : str or int, default=0
            Sheet name or index for Excel files (ignored for CSV)
        **kwargs : dict
            Additional arguments passed to pd.read_csv() or pd.read_excel()

        Returns:
        --------
        data : pd.DataFrame
            Loaded data

        Examples:
        ---------
        >>> # Load CSV file
        >>> data = HybridMICEImputer.load_data('data.csv')
        >>>
        >>> # Load Excel file (first sheet)
        >>> data = HybridMICEImputer.load_data('data.xlsx')
        >>>
        >>> # Load specific sheet from Excel
        >>> data = HybridMICEImputer.load_data('data.xlsx', sheet_name='Sheet2')
        >>> data = HybridMICEImputer.load_data('data.xlsx', sheet_name=1)
        """
        file_path = str(file_path)

        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, **kwargs)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        else:
            # Try to infer format
            try:
                return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            except:
                return pd.read_csv(file_path, **kwargs)

    @staticmethod
    def save_data(
        data: pd.DataFrame,
        file_path: str,
        sheet_name: str = 'Sheet1',
        index: bool = False,
        **kwargs
    ) -> None:
        """
        Save data to CSV or Excel (.xlsx) file.

        Parameters:
        -----------
        data : pd.DataFrame
            Data to save
        file_path : str
            Path to save the file (supports .csv, .xlsx)
        sheet_name : str, default='Sheet1'
            Sheet name for Excel files (ignored for CSV)
        index : bool, default=False
            Whether to include the index in the saved file
        **kwargs : dict
            Additional arguments passed to pd.to_csv() or pd.to_excel()

        Examples:
        ---------
        >>> # Save to CSV
        >>> HybridMICEImputer.save_data(imputed_data, 'output.csv')
        >>>
        >>> # Save to Excel
        >>> HybridMICEImputer.save_data(imputed_data, 'output.xlsx')
        >>>
        >>> # Save to Excel with custom sheet name
        >>> HybridMICEImputer.save_data(imputed_data, 'output.xlsx', sheet_name='Imputed Data')
        """
        file_path = str(file_path)

        if file_path.endswith('.csv'):
            data.to_csv(file_path, index=index, **kwargs)
        elif file_path.endswith('.xlsx'):
            data.to_excel(file_path, sheet_name=sheet_name, index=index, **kwargs)
        else:
            # Default to CSV if extension not recognized
            data.to_csv(file_path, index=index, **kwargs)
