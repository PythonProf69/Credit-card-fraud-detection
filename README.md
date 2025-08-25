# Credit Card Fraud Detection 💳

## Project Overview

This project builds a machine learning model to detect fraudulent credit card transactions. The primary challenge with this type of problem is the highly imbalanced dataset, where fraudulent transactions are extremely rare compared to legitimate ones.

This project addresses the class imbalance issue by implementing an **undersampling** technique. It trains a **`RandomForestClassifier`** on a balanced subset of the data to effectively learn the patterns of fraudulent activities. The final trained model is saved as `credit_card_fraud_model.pkl` for future use.

## Methodology

The project follows a structured approach to handle the data and train the model:

1.  **Data Loading**: The `creditcard.csv` dataset is loaded into a Pandas DataFrame.
2.  **Class Imbalance Handling**: To create a balanced training environment, the majority class (legitimate transactions) is undersampled to match the number of minority class instances (fraudulent transactions).
3.  **Data Splitting**: The newly created balanced dataset is split into training and testing sets (80% for training, 20% for testing).
4.  **Model Training**: A `RandomForestClassifier` is trained on the training data. This model is well-suited for this task due to its robustness and ability to handle complex datasets.
5.  **Evaluation**: The model's performance is evaluated on the test set using the **accuracy score**.
6.  **Model Saving**: The trained model is saved to a file using `joblib`, allowing it to be easily loaded for making predictions later without retraining.

## Technologies Used

  * **Python 3**
  * **Pandas**: For data manipulation and analysis.
  * **NumPy**: For numerical operations.
  * **Scikit-learn**: For machine learning, including data splitting, model training, and evaluation.
  * **Joblib**: For saving the final trained model.

## How to Run This Project

### 1\. Prerequisites

First, make sure you have Python installed. Then, install the required libraries by creating a `requirements.txt` file with the content below.

**`requirements.txt`**

```
pandas
numpy
scikit-learn
```

Install these packages using pip:

```bash
pip install -r requirements.txt
```

### 2\. Dataset

Download the `creditcard.csv` dataset and place it in the same directory as your Python script.

### 3\. Execute the Script

Run the script from your terminal. Replace `your_script_name.py` with the actual name of your file.

```bash
python your_script_name.py
```

## Output

The script will first print the mean values for each feature grouped by class. After training and evaluation, it will print the model's accuracy on the test set and create a file named **`credit_card_fraud_model.pkl`** in the same directory.