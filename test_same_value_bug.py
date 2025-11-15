"""
Test to check if PMM is imputing the same value for all missing entries (bug check)
"""

import numpy as np
import pandas as pd
from hybrid_imputer import HybridMICEImputer

def test_varied_imputation():
    """Test that imputed values are varied, not all the same"""
    np.random.seed(42)

    # Create dataset with strong correlations
    n = 200
    x1 = np.random.randn(n)
    x2 = 0.9 * x1 + 0.1 * np.random.randn(n)
    x3 = 0.8 * x1 + 0.2 * x2 + 0.1 * np.random.randn(n)

    data = pd.DataFrame({
        'var1': x1,
        'var2': x2,
        'var3': x3
    })

    # Introduce 40% missing in var2
    missing_indices = np.random.choice(n, size=int(n * 0.4), replace=False)
    data.loc[missing_indices, 'var2'] = np.nan

    print("="*70)
    print("TEST: Checking for 'same value' bug in PMM imputation")
    print("="*70)
    print(f"\nDataset: {n} samples, 40% missing in var2")
    print(f"Missing indices: {len(missing_indices)}")

    # Run imputation
    imputer = HybridMICEImputer(
        n_iterations=10,
        n_neighbors=5,
        verbose=True,
        random_state=42
    )

    data_imputed = imputer.fit_transform(data)

    # Extract imputed values
    imputed_vals = data_imputed.loc[missing_indices, 'var2'].values

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    # Check uniqueness
    unique_vals = np.unique(imputed_vals)
    print(f"\nNumber of unique imputed values: {len(unique_vals)}")
    print(f"Total imputed values: {len(imputed_vals)}")
    print(f"Percentage unique: {100 * len(unique_vals) / len(imputed_vals):.1f}%")

    print(f"\nStatistics of imputed values:")
    print(f"  Mean: {imputed_vals.mean():.4f}")
    print(f"  Std:  {imputed_vals.std():.4f}")
    print(f"  Min:  {imputed_vals.min():.4f}")
    print(f"  Max:  {imputed_vals.max():.4f}")
    print(f"  Range: {imputed_vals.max() - imputed_vals.min():.4f}")

    print(f"\nFirst 20 imputed values:")
    print(imputed_vals[:20])

    # Test for the bug
    if len(unique_vals) == 1:
        print("\n" + "!"*70)
        print("❌ BUG DETECTED: All imputed values are identical!")
        print(f"   All values = {unique_vals[0]:.4f}")
        print("!"*70)
        return False
    elif len(unique_vals) < 5:
        print("\n" + "!"*70)
        print("⚠️  WARNING: Very few unique values!")
        print(f"   Only {len(unique_vals)} unique values out of {len(imputed_vals)}")
        print("!"*70)
        return False
    else:
        print("\n" + "="*70)
        print("✓ PASS: PMM is working correctly - values are varied!")
        print("="*70)
        return True

def test_multiple_scenarios():
    """Test multiple scenarios to ensure PMM works in all cases"""
    scenarios = [
        {"n": 100, "missing_rate": 0.2, "name": "Small dataset, 20% missing"},
        {"n": 500, "missing_rate": 0.5, "name": "Medium dataset, 50% missing"},
        {"n": 1000, "missing_rate": 0.1, "name": "Large dataset, 10% missing"},
    ]

    print("\n\n" + "="*70)
    print("TESTING MULTIPLE SCENARIOS")
    print("="*70)

    results = []

    for scenario in scenarios:
        print(f"\n\nScenario: {scenario['name']}")
        print("-"*70)

        np.random.seed(42)
        n = scenario['n']
        missing_rate = scenario['missing_rate']

        # Create data
        x1 = np.random.randn(n)
        x2 = 0.7 * x1 + 0.3 * np.random.randn(n)
        x3 = 0.6 * x1 + 0.4 * x2 + 0.2 * np.random.randn(n)

        data = pd.DataFrame({'v1': x1, 'v2': x2, 'v3': x3})

        # Introduce missing
        missing_idx = np.random.choice(n, size=int(n * missing_rate), replace=False)
        data.loc[missing_idx, 'v2'] = np.nan

        # Impute
        imputer = HybridMICEImputer(n_iterations=5, verbose=False, random_state=42)
        data_imp = imputer.fit_transform(data)

        # Check
        imputed_vals = data_imp.loc[missing_idx, 'v2'].values
        n_unique = len(np.unique(imputed_vals))

        passed = n_unique > 5
        status = "✓ PASS" if passed else "❌ FAIL"

        print(f"  Missing values: {len(missing_idx)}")
        print(f"  Unique imputed values: {n_unique}")
        print(f"  Std of imputed: {imputed_vals.std():.4f}")
        print(f"  Status: {status}")

        results.append(passed)

    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total scenarios tested: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")

    if all(results):
        print("\n✓ ALL TESTS PASSED - PMM is working correctly!")
    else:
        print("\n❌ SOME TESTS FAILED - There may be a bug!")

    return all(results)

if __name__ == "__main__":
    # Run basic test
    test1_passed = test_varied_imputation()

    # Run multiple scenarios
    test2_passed = test_multiple_scenarios()

    print("\n\n" + "="*70)
    print("OVERALL TEST RESULT")
    print("="*70)
    if test1_passed and test2_passed:
        print("✓ ALL TESTS PASSED")
        print("PMM+MICE+Correlation imputation is working correctly!")
    else:
        print("❌ TESTS FAILED")
        print("There are bugs in the imputation logic!")
