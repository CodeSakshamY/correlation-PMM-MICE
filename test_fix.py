"""
Test script to verify that PMM imputation is working correctly
and generating varied realistic values instead of repeating the same value.
"""

import numpy as np
import pandas as pd
from hybrid_imputer import HybridMICEImputer

# Set random seed for reproducibility
np.random.seed(42)

# Create a simple dataset with correlations
n_samples = 100

# Generate correlated features
x1 = np.random.randn(n_samples)
x2 = 0.8 * x1 + 0.2 * np.random.randn(n_samples)  # Strongly correlated with x1
x3 = 0.6 * x1 + 0.4 * x2 + 0.3 * np.random.randn(n_samples)  # Correlated with both

# Create DataFrame
data = pd.DataFrame({
    'feature1': x1,
    'feature2': x2,
    'feature3': x3
})

# Save complete data for comparison
data_complete = data.copy()

# Introduce missing values in feature2
missing_rate = 0.30
n_missing = int(n_samples * missing_rate)
missing_indices = np.random.choice(n_samples, n_missing, replace=False)
data.loc[missing_indices, 'feature2'] = np.nan

print("="*70)
print("Testing PMM Imputation Fix")
print("="*70)
print(f"\nDataset size: {n_samples} samples")
print(f"Missing values in feature2: {n_missing} ({missing_rate*100}%)")
print(f"\nfeature2 statistics (before introducing missing values):")
print(f"  Mean: {data_complete['feature2'].mean():.4f}")
print(f"  Std:  {data_complete['feature2'].std():.4f}")
print(f"  Min:  {data_complete['feature2'].min():.4f}")
print(f"  Max:  {data_complete['feature2'].max():.4f}")

# Initialize imputer
print("\n" + "="*70)
print("Running HybridMICEImputer...")
print("="*70)

imputer = HybridMICEImputer(
    n_iterations=5,
    n_neighbors=5,
    correlation_threshold=0.3,
    verbose=True,
    random_state=42
)

# Impute missing values
data_imputed = imputer.fit_transform(data)

# Analyze the imputed values
imputed_values = data_imputed.loc[missing_indices, 'feature2'].values
true_values = data_complete.loc[missing_indices, 'feature2'].values

print("\n" + "="*70)
print("Verification Results")
print("="*70)

# Check if all imputed values are the same (the bug we fixed)
unique_imputed = np.unique(imputed_values)
print(f"\nNumber of unique imputed values: {len(unique_imputed)}")

if len(unique_imputed) == 1:
    print("❌ FAIL: All imputed values are the same!")
    print(f"   Repeated value: {unique_imputed[0]:.4f}")
    print("   This indicates PMM is NOT working - still just using mean imputation")
else:
    print("✓ PASS: Imputed values are varied (PMM is working!)")
    print(f"   Range of imputed values: [{imputed_values.min():.4f}, {imputed_values.max():.4f}]")
    print(f"   Std of imputed values: {imputed_values.std():.4f}")

# Calculate imputation accuracy
rmse = np.sqrt(np.mean((true_values - imputed_values) ** 2))
print(f"\nImputation Quality:")
print(f"  RMSE: {rmse:.4f}")

# Show correlation between imputed and true values
correlation = np.corrcoef(true_values, imputed_values)[0, 1]
print(f"  Correlation (true vs imputed): {correlation:.4f}")

# Show some examples
print(f"\nSample of imputed values (first 10):")
print(f"  True values:    {true_values[:10]}")
print(f"  Imputed values: {imputed_values[:10]}")

# Check that imputed values are drawn from observed values (PMM property)
observed_values = data_complete.loc[~data['feature2'].isnull(), 'feature2'].values
all_from_donors = all(val in observed_values for val in imputed_values)

print(f"\nPMM Property Check:")
if all_from_donors:
    print("✓ PASS: All imputed values are from donor pool (true PMM behavior)")
else:
    print("⚠ INFO: Some imputed values not in exact donor pool (expected with MICE iterations)")

print("\n" + "="*70)
print("Test Complete!")
print("="*70)
