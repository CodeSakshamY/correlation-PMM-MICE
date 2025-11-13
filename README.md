# Hybrid Correlation + PMM + MICE Imputer

A sophisticated missing data imputation library that combines three powerful techniques:
- **Correlation Analysis**: Identifies optimal predictor sets based on feature correlations
- **PMM (Predictive Mean Matching)**: Preserves data distribution through semi-parametric imputation
- **MICE (Multivariate Imputation by Chained Equations)**: Iteratively refines imputations for better accuracy

## Features

- **Intelligent Predictor Selection**: Uses correlation analysis to automatically identify the most relevant features for predicting each missing variable
- **Distribution Preservation**: PMM ensures imputed values maintain the original data distribution
- **Iterative Refinement**: MICE framework allows imputed values to inform subsequent imputations
- **Convergence Monitoring**: Built-in convergence detection to avoid unnecessary iterations
- **Flexible Configuration**: Multiple model types (Linear, Bayesian, Random Forest) and customizable parameters
- **Comprehensive Diagnostics**: Detailed insights into the imputation process including convergence history and predictor sets
- **Adaptive Algorithms**: Automatically adjusts to data characteristics and availability

## How It Works

The hybrid approach combines the strengths of each technique:

1. **Correlation Analysis Phase**:
   - Computes pairwise correlations between all features
   - Identifies strongly correlated features for each variable
   - Determines optimal imputation order based on predictor availability

2. **Predictive Mean Matching Phase**:
   - Fits a prediction model (linear regression, Bayesian, or random forest) on observed data
   - Generates predictions for missing values
   - Finds observed values with similar predictions (donor pool)
   - Randomly selects from donor pool to preserve distribution

3. **MICE Iteration Phase**:
   - Iteratively imputes each variable using updated values from other variables
   - Monitors convergence by tracking changes in imputed values
   - Stops when convergence threshold is met or max iterations reached

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/correlation-PMM-MICE.git
cd correlation-PMM-MICE

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
import pandas as pd
from hybrid_imputer import HybridMICEImputer

# Load your data with missing values
data = pd.read_csv('your_data.csv')

# Initialize the imputer
imputer = HybridMICEImputer(
    n_iterations=10,        # Maximum MICE iterations
    n_neighbors=5,          # Number of donors for PMM
    correlation_threshold=0.3,  # Minimum correlation for predictor selection
    verbose=True,           # Show progress
    random_state=42         # For reproducibility
)

# Impute missing values
data_imputed = imputer.fit_transform(data)

# Check convergence
imputer.plot_convergence()
```

## Usage Examples

### Basic Imputation

```python
from hybrid_imputer import HybridMICEImputer
import pandas as pd
import numpy as np

# Create sample data with missing values
data = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [2, np.nan, 6, 8, 10],
    'C': [3, 6, 9, np.nan, 15]
})

# Impute
imputer = HybridMICEImputer(verbose=True)
data_complete = imputer.fit_transform(data)
print(data_complete)
```

### Custom Configuration

```python
# Use Bayesian Ridge regression for PMM
imputer = HybridMICEImputer(
    n_iterations=15,
    n_neighbors=7,
    pmm_model_type='bayesian',  # Options: 'linear', 'bayesian', 'rf'
    correlation_threshold=0.25,
    max_predictors=10,
    convergence_threshold=0.001,
    random_state=42,
    verbose=True
)

data_imputed = imputer.fit_transform(data)
```

### Impute Specific Columns Only

```python
# Impute only selected columns
imputer = HybridMICEImputer()
data_imputed = imputer.fit_transform(
    data,
    columns_to_impute=['column1', 'column2']
)
```

### Get Diagnostic Information

```python
# Perform imputation
imputer = HybridMICEImputer(verbose=True)
data_imputed = imputer.fit_transform(data)

# Get diagnostics
diagnostics = imputer.get_diagnostics()

print("Imputation order:", diagnostics['imputation_order'])
print("Convergence history:", diagnostics['convergence_history'])
print("Predictor sets:", diagnostics['predictor_sets'])

# Visualize convergence
imputer.plot_convergence()

# Visualize correlations
imputer.correlation_analyzer.visualize_correlations()
```

### Working with Excel (.xlsx) Files

The library provides convenient methods for loading and saving Excel files:

```python
from hybrid_imputer import HybridMICEImputer

# Load data from Excel file
data = HybridMICEImputer.load_data('data.xlsx')

# Or load from a specific sheet
data = HybridMICEImputer.load_data('data.xlsx', sheet_name='Sheet2')
data = HybridMICEImputer.load_data('data.xlsx', sheet_name=1)  # By index

# Perform imputation
imputer = HybridMICEImputer(verbose=True)
data_imputed = imputer.fit_transform(data)

# Save to Excel file
HybridMICEImputer.save_data(data_imputed, 'imputed_data.xlsx')

# Save with custom sheet name
HybridMICEImputer.save_data(
    data_imputed,
    'imputed_data.xlsx',
    sheet_name='Imputed Data'
)

# The same methods work for CSV files
data = HybridMICEImputer.load_data('data.csv')
HybridMICEImputer.save_data(data_imputed, 'imputed_data.csv')
```

**Note**: Excel file support requires the `openpyxl` package, which is included in `requirements.txt`.

## API Reference

### HybridMICEImputer

Main class for hybrid imputation.

**Parameters:**
- `n_iterations` (int, default=10): Maximum number of MICE iterations
- `n_neighbors` (int, default=5): Number of nearest neighbors for PMM donor pool
- `correlation_threshold` (float, default=0.3): Minimum correlation coefficient to consider a feature as predictor
- `max_predictors` (int, default=10): Maximum number of predictors to use per variable
- `pmm_model_type` (str, default='linear'): Model type for PMM ('linear', 'bayesian', or 'rf')
- `convergence_threshold` (float, default=0.001): Convergence detection threshold
- `random_state` (int, optional): Random seed for reproducibility
- `verbose` (bool, default=False): Whether to print progress information

**Methods:**
- `fit_transform(data, columns_to_impute=None)`: Impute missing values and return completed dataset
- `fit(data)`: Fit the imputer on training data
- `transform(data)`: Transform data using fitted imputer
- `get_diagnostics()`: Get detailed diagnostic information
- `plot_convergence()`: Visualize convergence history
- `load_data(file_path, sheet_name=0, **kwargs)`: Load data from CSV or Excel file (static method)
- `save_data(data, file_path, sheet_name='Sheet1', index=False, **kwargs)`: Save data to CSV or Excel file (static method)

### CorrelationAnalyzer

Analyzes correlations for predictor selection.

**Parameters:**
- `correlation_threshold` (float, default=0.3): Minimum absolute correlation coefficient

**Methods:**
- `fit(data)`: Compute correlation matrix
- `get_predictors(target_column, max_predictors=None)`: Get predictor list for a column
- `get_correlation_strength(col1, col2)`: Get correlation between two columns
- `get_imputation_order(columns_with_missing)`: Determine optimal imputation order
- `visualize_correlations(figsize=(12, 10))`: Create correlation heatmap

### PMMImputer / AdaptivePMMImputer

Predictive Mean Matching imputers.

**Parameters:**
- `n_neighbors` (int, default=5): Number of donors in PMM
- `model_type` (str, default='linear'): Prediction model type
- `random_state` (int, optional): Random seed

**Methods:**
- `fit_transform(data, target_column, predictor_columns)`: Impute a single column
- `impute_column(data, target_column, predictor_columns=None)`: Convenience method

## Advantages Over Traditional Methods

### Compared to Simple Imputation (Mean/Median)
- Preserves data distribution and relationships between variables
- Accounts for correlations between features
- More accurate for complex datasets

### Compared to Standard MICE
- Automatically selects optimal predictors using correlation analysis
- Uses PMM instead of simple regression, preserving distributions
- Adaptive neighbor selection for robust performance

### Compared to PMM Alone
- Iterative refinement through MICE improves accuracy
- Handles multiple missing patterns more effectively
- Better convergence properties

## Advanced Topics

### Understanding Convergence

The imputation process iterates until:
1. The convergence threshold is met (changes in imputed values become negligible)
2. Maximum iterations are reached

Monitor convergence with:
```python
imputer.plot_convergence()
diagnostics = imputer.get_diagnostics()
print(diagnostics['convergence_history'])
```

### Choosing Model Type

- **Linear** (`pmm_model_type='linear'`): Fast, works well for linear relationships
- **Bayesian** (`pmm_model_type='bayesian'`): More robust to outliers, provides uncertainty estimates
- **Random Forest** (`pmm_model_type='rf'`): Captures non-linear relationships, slower but more flexible

### Handling Different Missing Data Patterns

The hybrid approach handles:
- **MCAR** (Missing Completely at Random): Excellent performance
- **MAR** (Missing at Random): Good performance when predictors are available
- **MNAR** (Missing Not at Random): Limited performance (inherent limitation of all imputation methods)

## Performance Tips

1. **Large Datasets**: Reduce `max_predictors` to speed up computation
2. **High Missing Rates**: Increase `n_iterations` for better convergence
3. **Complex Patterns**: Use `pmm_model_type='rf'` for non-linear relationships
4. **Speed vs Quality**: Use `pmm_model_type='linear'` for faster imputation

## Examples

Run the comprehensive examples:

```bash
python example.py
```

This includes:
- Basic usage demonstration
- Comparison of different model types
- Diagnostic features
- Partial imputation examples
- Excel (.xlsx) file support

## Requirements

- Python >= 3.7
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0 (optional, for visualization)
- seaborn >= 0.11.0 (optional, for correlation heatmaps)
- openpyxl >= 3.0.0 (for Excel .xlsx file support)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{hybrid_mice_imputer,
  title={Hybrid Correlation + PMM + MICE Imputer},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/correlation-PMM-MICE}
}
```

## References

- van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. Journal of Statistical Software, 45(3), 1-67.
- Little, R. J. A. (1988). Missing-data adjustments in large surveys. Journal of Business & Economic Statistics, 6(3), 287-296.
- Schenker, N., & Taylor, J. M. (1996). Partially parametric techniques for multiple imputation. Computational Statistics & Data Analysis, 22(4), 425-446.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.
