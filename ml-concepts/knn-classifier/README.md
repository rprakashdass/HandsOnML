# K-Nearest Neighbors (KNN) Implementation

This directory contains a Python implementation of the K-Nearest Neighbors (KNN) algorithm for binary classification using employee data. The implementation includes data preprocessing, model training, and evaluation.

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

K-Nearest Neighbors is a simple, yet powerful supervised learning algorithm used for classification tasks. This implementation demonstrates how to use KNN to predict whether an employee will leave the company based on various features.

## Dataset

The dataset used for this implementation is assumed to be in the format of `Employee.csv`, containing various attributes related to employees, including education level, gender, and whether they have ever been benched. The target variable is `LeaveOrNot`, indicating if the employee left the company.

### Data Preprocessing
- Categorical variables are encoded (e.g., `Education`, `Gender`, `EverBenched`).
- The dataset is split into features and the target variable.
- Data is split into training and testing sets.
- Features are scaled using Min-Max scaling.

## Code Structure

```
knn/
├── knn.py                    # Main implementation of the KNN model
├── datasets
└── README.md                  # Documentation for the KNN implementation
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
   git clone https://github.com/rprakashdass/HandsOnML.git
   cd ml-concepts/knn
   ```

2. **Prepare your dataset**: Ensure you have the `Employee.csv` dataset in the specified path within the code.

3. **Run the KNN implementation**:
   ```bash
   python knn.py
   ```

## Usage

The main script (`knn.py`) contains the complete implementation of the KNN model, including:
- Loading and preprocessing the dataset
- Training the KNN classifier
- Evaluating the model's accuracy

### Example Output
Upon running the code, you will see printed output for the accuracy score of the model.

## Evaluation Metrics

The primary metric used to evaluate the model's performance is:
- **Accuracy**: The proportion of correct predictions among the total predictions.

## Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, please open an issue or submit a pull request.
