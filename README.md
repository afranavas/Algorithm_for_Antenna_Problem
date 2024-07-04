# Predicting S-Parameter Responses Using Hybrid ANN-SVM Model

## Introduction

This assignment aims to predict S-parameter responses, specifically the S11 parameter, using a hybrid Artificial Neural Network (ANN) and Support Vector Machine (SVM) model. S-parameters, or scattering parameters, describe how RF signals behave in multi-port networks and are fundamental in antenna design and analysis. Accurate prediction of these parameters is essential for optimizing the performance of antennas.

## Dataset

The dataset used in this project is provided in two `.mat` files:
- `Training_Data.mat`
- `Real_Test_Data.mat`

Each file contains the following:
- `candidates`: The input features for training the model.
- `responses`: The S-parameter responses corresponding to the training candidates.
- `real_test_candidates`: The input features for testing the model.
- `real_test_responses`: The actual S-parameter responses for the test candidates.

## Data Preprocessing

Data preprocessing is a crucial step that ensures the input data is in the correct format and scale for the machine learning models. This involves several steps:

1. **Loading the Data:**  
   The data is loaded from the `.mat` files using the `scipy.io.loadmat` function, which reads MATLAB files and returns a dictionary containing all variables.

2. **Extracting Relevant Data:**  
   The relevant data arrays, such as `candidates`, `responses`, `real_test_candidates`, and `real_test_responses`, are extracted from the loaded data.

3. **Normalization:**  
   The input features are normalized to have a mean of 0 and a standard deviation of 1. This step is critical because it ensures that the features contribute equally to the model's predictions and improves the convergence of the training algorithm.

4. **Reshaping Data:**  
   The data arrays are reshaped to fit the requirements of the ANN and SVM models. This includes ensuring that the input features and responses are in the correct format for the models to process.

## Model Architecture

The hybrid model comprises two main components: the Artificial Neural Network (ANN) and the Support Vector Machine (SVM).

1. **Artificial Neural Network (ANN):**
   - The ANN is designed to extract meaningful features from the input data.
   - The network consists of an input layer, several hidden layers with ReLU (Rectified Linear Unit) activation functions, and an output layer.
   - The ANN is trained to minimize the Mean Squared Error (MSE) between the predicted and actual S-parameter responses.

2. **Support Vector Machine (SVM):**
   - The SVM takes the features extracted by the ANN and uses them to predict the S-parameter responses.
   - An RBF (Radial Basis Function) kernel is used to transform the input data into a higher-dimensional space, allowing for better separation and prediction of the responses.

## Training

The training process involves two stages:

1. **Training the ANN:**
   - The ANN is trained using the training data. The goal is to minimize the MSE between the predicted and actual responses.
   - The Adam optimizer is used to update the network weights during training.

2. **Training the SVM:**
   - After training the ANN, the extracted features from the training data are used to train the SVM.
   - The SVM model is trained to further refine the predictions based on the features provided by the ANN.

## Evaluation

The model's performance is evaluated using the testing data. The key steps in the evaluation process include:

1. **Predicting Test Responses:**
   - The trained ANN extracts features from the test data, which are then fed into the SVM to predict the S-parameter responses.

2. **Calculating Mean Squared Error:**
   - The Mean Squared Error (MSE) between the predicted and actual test responses is calculated to quantify the model's performance.

3. **Visualization:**
   - The predicted and actual S-parameter responses are plotted to visually assess the model's accuracy. The goal is for the predicted responses to closely match the actual responses.

## Results

The results demonstrate the effectiveness of the hybrid ANN-SVM model in predicting S-parameter responses. Key findings include:

- The MSE value indicates the average squared difference between the predicted and actual responses.
- Visual comparisons show how well the predicted responses align with the actual test responses.
- Discussion of the model's performance, including any observed patterns or discrepancies, helps identify areas for improvement.

## Conclusion

In conclusion, this assignment successfully implements a hybrid ANN-SVM model to predict S-parameter responses. The model combines the feature extraction capabilities of ANNs with the predictive power of SVMs to achieve accurate results. Future work may involve exploring different model architectures, tuning hyperparameters, and incorporating additional data for further improvements.
