"""
Example usage of the Hybrid Correlation + PMM + MICE Imputer
"""

import numpy as np
import pandas as pd
from hybrid_imputer import HybridMICEImputer


def create_sample_data_with_missing(n_samples=1000, missing_rate=0.2, random_state=42):
    """
    Create a sample dataset with missing values for demonstration.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    missing_rate : float
        Proportion of values to set as missing
    random_state : int
        Random seed

    Returns:
    --------
    data : pd.DataFrame
        Dataset with missing values
    data_complete : pd.DataFrame
        Original complete dataset (for comparison)
    """
    np.random.seed(random_state)

    # Generate correlated features
    # Feature 1: base random variable
    x1 = np.random.randn(n_samples)

    # Feature 2: strongly correlated with x1
    x2 = 0.8 * x1 + 0.2 * np.random.randn(n_samples)

    # Feature 3: moderately correlated with x1 and x2
    x3 = 0.5 * x1 + 0.3 * x2 + 0.4 * np.random.randn(n_samples)

    # Feature 4: weakly correlated
    x4 = 0.2 * x1 + 0.8 * np.random.randn(n_samples)

    # Feature 5: independent
    x5 = np.random.randn(n_samples)

    # Create DataFrame
    data_complete = pd.DataFrame({
        'feature1': x1,
        'feature2': x2,
        'feature3': x3,
        'feature4': x4,
        'feature5': x5
    })

    # Add a target variable
    data_complete['target'] = (
        2 * x1 + 1.5 * x2 - 0.5 * x3 + np.random.randn(n_samples) * 0.5
    )

    # Create missing values
    data_with_missing = data_complete.copy()

    for col in data_with_missing.columns:
        # Randomly select indices to set as missing
        n_missing = int(n_samples * missing_rate)
        missing_indices = np.random.choice(n_samples, n_missing, replace=False)
        data_with_missing.loc[missing_indices, col] = np.nan

    return data_with_missing, data_complete


def example_basic_usage():
    """Example 1: Basic usage of the hybrid imputer."""
    print("=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)

    # Create sample data
    data_missing, data_complete = create_sample_data_with_missing(
        n_samples=500,
        missing_rate=0.15
    )

    print("\nOriginal data with missing values:")
    print(data_missing.head(10))
    print("\nMissing value statistics:")
    print(data_missing.isnull().sum())

    # Initialize and fit the hybrid imputer
    imputer = HybridMICEImputer(
        n_iterations=10,
        n_neighbors=5,
        correlation_threshold=0.3,
        verbose=True,
        random_state=42
    )

    # Impute missing values
    data_imputed = imputer.fit_transform(data_missing)

    print("\n\nImputed data:")
    print(data_imputed.head(10))

    # Calculate imputation accuracy (RMSE)
    print("\n\nImputation Quality Assessment:")
    for col in data_missing.columns:
        missing_mask = data_missing[col].isnull()
        if missing_mask.any():
            true_values = data_complete.loc[missing_mask, col]
            imputed_values = data_imputed.loc[missing_mask, col]
            rmse = np.sqrt(np.mean((true_values - imputed_values) ** 2))
            print(f"{col}: RMSE = {rmse:.4f}")


def example_custom_configuration():
    """Example 2: Custom configuration with different settings."""
    print("\n\n" + "=" * 70)
    print("Example 2: Custom Configuration")
    print("=" * 70)

    # Create data with more missing values
    data_missing, data_complete = create_sample_data_with_missing(
        n_samples=300,
        missing_rate=0.30
    )

    print(f"\nDataset size: {len(data_missing)} samples")
    print(f"Missing rate: ~30%")

    # Try different model configurations
    configs = [
        {'name': 'Linear PMM', 'pmm_model_type': 'linear'},
        {'name': 'Bayesian PMM', 'pmm_model_type': 'bayesian'},
        {'name': 'Random Forest PMM', 'pmm_model_type': 'rf'}
    ]

    results = {}

    for config in configs:
        print(f"\n\nTesting: {config['name']}")
        print("-" * 40)

        imputer = HybridMICEImputer(
            n_iterations=15,
            n_neighbors=7,
            pmm_model_type=config['pmm_model_type'],
            correlation_threshold=0.25,
            verbose=False,
            random_state=42
        )

        data_imputed = imputer.fit_transform(data_missing)

        # Calculate overall RMSE
        total_rmse = 0
        n_cols = 0

        for col in data_missing.columns:
            missing_mask = data_missing[col].isnull()
            if missing_mask.any():
                true_values = data_complete.loc[missing_mask, col]
                imputed_values = data_imputed.loc[missing_mask, col]
                rmse = np.sqrt(np.mean((true_values - imputed_values) ** 2))
                total_rmse += rmse
                n_cols += 1

        avg_rmse = total_rmse / n_cols if n_cols > 0 else 0
        results[config['name']] = avg_rmse

        print(f"Average RMSE: {avg_rmse:.4f}")
        print(f"Iterations to convergence: {len(imputer.convergence_history)}")

    print("\n\nComparison Summary:")
    print("-" * 40)
    for name, rmse in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name}: {rmse:.4f}")


def example_diagnostics():
    """Example 3: Using diagnostic features."""
    print("\n\n" + "=" * 70)
    print("Example 3: Diagnostics and Analysis")
    print("=" * 70)

    # Create sample data
    data_missing, _ = create_sample_data_with_missing(
        n_samples=400,
        missing_rate=0.20
    )

    # Initialize imputer
    imputer = HybridMICEImputer(
        n_iterations=20,
        verbose=False,
        random_state=42
    )

    # Impute
    data_imputed = imputer.fit_transform(data_missing)

    # Get diagnostics
    diagnostics = imputer.get_diagnostics()

    print("\nImputation Order (based on correlations):")
    for i, col in enumerate(diagnostics['imputation_order'], 1):
        print(f"{i}. {col}")

    print("\n\nPredictor Sets (correlation-based):")
    for col, predictors in diagnostics['predictor_sets'].items():
        if col in diagnostics['columns_with_missing']:
            print(f"\n{col}:")
            print(f"  Top predictors: {predictors[:3]}")

    print("\n\nCorrelation Matrix:")
    print(diagnostics['correlation_matrix'])

    print("\n\nConvergence History:")
    for i, conv in enumerate(diagnostics['convergence_history'], 1):
        print(f"Iteration {i}: {conv:.6f}")

    # Plot convergence (if matplotlib available)
    try:
        imputer.plot_convergence()
    except:
        print("\nNote: Install matplotlib to visualize convergence")


def example_partial_imputation():
    """Example 4: Imputing only specific columns."""
    print("\n\n" + "=" * 70)
    print("Example 4: Partial Imputation (Specific Columns)")
    print("=" * 70)

    # Create sample data
    data_missing, data_complete = create_sample_data_with_missing(
        n_samples=500,
        missing_rate=0.15
    )

    print("\nImputing only 'feature1' and 'target' columns...")

    # Initialize imputer
    imputer = HybridMICEImputer(
        n_iterations=10,
        verbose=True,
        random_state=42
    )

    # Impute only specific columns
    data_imputed = imputer.fit_transform(
        data_missing,
        columns_to_impute=['feature1', 'target']
    )

    # Check which columns were imputed
    print("\n\nMissing values after imputation:")
    print(data_imputed.isnull().sum())


def example_xlsx_file_support():
    """Example 5: Loading and saving Excel (.xlsx) files."""
    print("\n\n" + "=" * 70)
    print("Example 5: Excel (.xlsx) File Support")
    print("=" * 70)

    # Create sample data
    data_missing, data_complete = create_sample_data_with_missing(
        n_samples=500,
        missing_rate=0.15
    )

    # Save data to Excel file
    print("\n1. Saving data with missing values to Excel file...")
    HybridMICEImputer.save_data(
        data_missing,
        'data_with_missing.xlsx',
        sheet_name='Original Data'
    )
    print("   Saved to: data_with_missing.xlsx")

    # Load data from Excel file
    print("\n2. Loading data from Excel file...")
    data_loaded = HybridMICEImputer.load_data('data_with_missing.xlsx')
    print(f"   Loaded {len(data_loaded)} rows, {len(data_loaded.columns)} columns")
    print(f"   Missing values: {data_loaded.isnull().sum().sum()}")

    # Perform imputation
    print("\n3. Imputing missing values...")
    imputer = HybridMICEImputer(
        n_iterations=10,
        verbose=False,
        random_state=42
    )
    data_imputed = imputer.fit_transform(data_loaded)
    print(f"   Imputation complete!")
    print(f"   Remaining missing values: {data_imputed.isnull().sum().sum()}")

    # Save imputed data to Excel file
    print("\n4. Saving imputed data to Excel file...")
    HybridMICEImputer.save_data(
        data_imputed,
        'data_imputed.xlsx',
        sheet_name='Imputed Data'
    )
    print("   Saved to: data_imputed.xlsx")

    # Also demonstrate CSV support
    print("\n5. The same methods work for CSV files...")
    HybridMICEImputer.save_data(data_imputed, 'data_imputed.csv')
    data_from_csv = HybridMICEImputer.load_data('data_imputed.csv')
    print(f"   CSV: {len(data_from_csv)} rows loaded")

    # Calculate imputation accuracy
    print("\n6. Imputation Quality Assessment:")
    total_rmse = 0
    n_cols = 0
    for col in data_missing.columns:
        missing_mask = data_missing[col].isnull()
        if missing_mask.any():
            true_values = data_complete.loc[missing_mask, col]
            imputed_values = data_imputed.loc[missing_mask, col]
            rmse = np.sqrt(np.mean((true_values - imputed_values) ** 2))
            total_rmse += rmse
            n_cols += 1
            print(f"   {col}: RMSE = {rmse:.4f}")

    avg_rmse = total_rmse / n_cols if n_cols > 0 else 0
    print(f"\n   Average RMSE: {avg_rmse:.4f}")

    print("\n✓ Files created: data_with_missing.xlsx, data_imputed.xlsx, data_imputed.csv")


if __name__ == "__main__":
    # Run all examples
    print("\n")
    print("*" * 70)
    print("Hybrid Correlation + PMM + MICE Imputer Examples")
    print("*" * 70)

    example_basic_usage()
    example_custom_configuration()
    example_diagnostics()
    example_partial_imputation()
    example_xlsx_file_support()

    print("\n\n" + "*" * 70)
    print("All examples completed!")
    print("*" * 70)
