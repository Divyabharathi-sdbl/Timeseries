**Advanced Time Series Forecasting Using LSTM and Seq2Seq Attention Models**

## 1. Introduction

Time series forecasting is a critical component of modern predictive analytics, with applications in finance, energy systems, industrial monitoring, and demand estimation. Traditional forecasting approaches often struggle with complex multivariate temporal patterns, non-linear dependencies, and long-range relationships. To address these challenges, deep learning models have become increasingly relevant.

This project focuses on building and comparing two neural network architectures for multivariate time series forecasting: a baseline Long Short-Term Memory (LSTM) model and an advanced Sequence-to-Sequence (Seq2Seq) model enhanced with Bahdanau Attention. The objective is to evaluate how attention mechanisms improve multi-step forecasting accuracy and interpretability.

## 2. Problem Statement

The forecasting task involves predicting the next 24 time steps of a multivariate sequence using the past 96 time steps. The dataset contains five correlated features that exhibit seasonality, noise, random spikes, and inter-feature relationships. The challenge is to design and train models capable of accurately learning these temporal dependencies and producing reliable multi-step predictions.

## 3. Dataset Description

A synthetic multivariate time series dataset was generated to emulate real-world sensor or financial signals. The dataset consists of:

* 6000 time steps
* 5 continuous features
* Seasonal patterns
* Linear trends
* Gaussian noise
* Random anomalies
* Lag-based feature coupling

The dataset provides sufficient variability and complexity to evaluate advanced forecasting architectures.

## 4. Methodology

### 4.1 Data Preparation

The dataset was standardized using the StandardScaler. A sliding-window method was applied to generate supervised learning samples: each input window consisted of 96 time steps, and the corresponding target output window consisted of 24 time steps. The data was split into training, validation, and testing subsets to ensure reliable evaluation.

### 4.2 Baseline Model: LSTM

The baseline model uses a multi-layer LSTM architecture for many-to-many forecasting. The final hidden representation of the LSTM is passed through a linear layer to generate a 24-step multivariate output. This model provides a benchmark to assess the enhancement offered by attention mechanisms.

### 4.3 Proposed Model: Seq2Seq with Bahdanau Attention

The advanced model follows an encoder-decoder framework. The encoder processes the input sequence using an LSTM and produces hidden representations. The decoder generates the forecasted sequence step-by-step using an LSTMCell. Bahdanau Attention is applied at each decoding step to compute context vectors, enabling the model to focus selectively on relevant input time steps. This architecture is designed to address long-term dependencies and improve prediction accuracy.

### 4.4 Training Setup

Both models were implemented in PyTorch and trained using the Adam optimizer with a learning rate of 0.001. The mean squared error (MSE) loss function was used. Training was conducted on GPU (CUDA), and early stopping was applied to prevent overfitting.

* LSTM: trained for 5 epochs
* Seq2Seq Attention: trained for 5 epochs

Although the training duration was short due to scheduling constraints, the models converged reasonably well.

## 5. Results and Evaluation

### 5.1 Test Set Metrics

The models were evaluated on the test dataset using Mean Absolute Error (MAE) and Root Mean Square Error (RMSE). The results are as follows:

Baseline LSTM Model

* MAE: 0.2077
* RMSE: 0.2658

Seq2Seq Attention Model

* MAE: 0.1762
* RMSE: 0.2235

The attention-based model demonstrated a clear improvement over the LSTM baseline for both metrics.

### 5.2 Visualization

Forecast versus actual plots showed that the Seq2Seq Attention model better captured short-term and trend-based patterns. The attention heatmap provided interpretability, highlighting that the decoder consistently focused on the most informative recent input time steps.

### 5.3 Rolling-Origin Evaluation

A rolling-origin evaluation was performed to simulate real-world forecasting scenarios where the model is repeatedly retrained or updated. Both models performed reasonably well across multiple folds, with the attention model showing competitive performance despite limited training epochs.

## 6. Discussion

The results confirm that attention mechanisms enhance temporal learning by directing the decoder toward the most relevant encoder states. Even with a synthetic dataset and limited training epochs, the Seq2Seq Attention model demonstrated superior forecasting ability compared to the traditional LSTM model. This validates the effectiveness of attention-based architectures for multivariate forecasting tasks.

## 7. Conclusion

This project successfully implements and compares LSTM and Seq2Seq Attention models for multivariate time series forecasting. The attention-based architecture outperformed the baseline in accuracy and interpretability. The generated heatmaps and forecast plots provided valuable insights into the decision-making process of the model.
Overall, this project demonstrates that incorporating attention mechanisms leads to more reliable and interpretable multi-step forecasts.

## 8. Future Work

* Train models for longer epochs and apply hyperparameter optimization.
* Evaluate Transformer-based architectures such as Informer or TFT.
* Apply the model to real-world datasets in finance, weather forecasting, or energy consumption.
* Experiment with feature-level attention and multi-head attention mechanisms.

