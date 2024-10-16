# Support Vector Machine (SVM) Implementation

This directory contains a Python implementation of the Support Vector Machine (SVM) algorithm for binary classification, specifically applied to employee data. The implementation includes data preprocessing, model training, evaluation metrics, and visualization.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Code Structure](#code-structure)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)

## Introduction

Support Vector Machines are supervised learning models used for classification tasks. This implementation demonstrates how to use SVM to predict whether an employee will leave the company based on various features.

## Dataset

The dataset used for this implementation is `Employee.csv`. It contains various attributes related to employees, including education level, gender, and whether they have ever been benched. The target variable is `LeaveOrNot`, which indicates whether the employee left the company.

### Data Preprocessing
- Categorical variables are encoded (e.g., `Education`, `Gender`, `EverBenched`).
- The dataset is split into features and the target variable.
- Data is split into training and testing sets.

## Code Structure

```
svm/
├── svm.ipynb                  # Main implementation of the SVM model
└── README.md                  # Documentation for the SVM implementation
```

## Requirements

To run the code in this directory, you need the following Python libraries:

- pandas
- scikit-learn

You can install the required libraries using pip:

```bash
pip install pandas scikit-learn
```

## Getting Started

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/rprakashdass/svm.git
   cd ml-concepts/svm
   ```

2. **Prepare your dataset**: Ensure you have the `Employee.csv` dataset in the specified path within the code.

3. **Run the SVM implementation**:
   ```bash
   python svm.py
   ```

## Usage

The main script (`svm.py`) contains the complete implementation of the SVM model, including:
- Loading and preprocessing the dataset
- Training the SVM classifier
- Evaluating the model with various metrics

### Example Output
Upon running the code, you will see printed outputs for:
- Confusion Matrix
- Accuracy
- Precision
- Recall
- Specificity
- F-score

## Evaluation Metrics

The following metrics are calculated to assess model performance:
- **Accuracy**: The proportion of correct predictions among the total predictions.
- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were identified correctly.
- **Specificity**: The proportion of actual negatives that were identified correctly.
- **F-score**: The harmonic mean of precision and recall.

## Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, please open an issue or submit a pull request.
