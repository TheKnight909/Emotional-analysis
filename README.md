# Emotion Prediction in Arabic Text

This repository contains the implementation of an emotion analysis model that predicts emotions from Arabic text. It leverages advanced machine learning techniques and NLP models like Arabert, Marabert, Gemini, and ChatGPT.

## Features

- Preprocessing of Arabic text data.
- Use of TF-IDF, BM-25, and Word2Vec for text vectorization.
- Integration of transfer learning models Arabert and Marabert.
- Emotion prediction using machine learning algorithms including KNN, SVM, and Random Forest.
- Utilization of generative AI models, such as Gemini and ChatGPT, to enhance emotion prediction capabilities.

## Getting Started

### Prerequisites

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk
- regex
- pyarabic
- emoji
- arabicstopwords

### Installation

Clone the repository to your local machine:

git clone https://github.com/your-repo/emotion-prediction.git
cd emotion-prediction


Install the required packages:
```
pip install -r requirements.txt
```

### Usage

Load the data and run the main script:

```python
import pandas as pd
```
# Load your data
df = pd.read_csv('your-data.csv')

# Assuming you have a function setup to preprocess and predict
from your_module import preprocess_data, predict_emotions

# Preprocess data
preprocessed_data = preprocess_data(df)

# Predict emotions
predicted_emotions = predict_emotions(preprocessed_data)

## Models and Performance

### Word Representation and Algorithms Performance

| Word Representation | Algorithm     | Accuracy | Precision (weighted avg) | Recall (weighted avg) | f1-score (weighted avg) |
|---------------------|---------------|----------|--------------------------|-----------------------|-------------------------|
| **TF-IDF**          | KNN           | 0.88     | 0.89                     | 0.88                  | 0.88                    |
|                     | Random Forest | 0.88     | 0.89                     | 0.88                  | 0.88                    |
|                     | SVM           | 0.89     | 0.90                     | 0.89                  | 0.89                    |
| **BM-25**           | KNN           | 0.81     | 0.84                     | 0.81                  | 0.82                    |
|                     | Random Forest | 0.82     | 0.83                     | 0.82                  | 0.82                    |
|                     | SVM           | 0.70     | 0.77                     | 0.70                  | 0.70                    |
| **Word2Vec**        | KNN           | 0.69     | 0.73                     | 0.69                  | 0.71                    |
|                     | Random Forest | 0.77     | 0.79                     | 0.77                  | 0.78                    |
|                     | SVM           | 0.50     | 0.57                     | 0.50                  | 0.49                    |

### Transfer Learning Model Performance

| Model                                       | Training Loss | Validation Loss | Accuracy | Roc Auc | f1-score (weighted avg) |
|---------------------------------------------|---------------|-----------------|----------|---------|-------------------------|
| UBC-NLP/MARBERTv2                           | 0.08          | 0.25            | 0.66     | 0.90    | 0.83                    |
| CAMeL-Lab/bert-base-arabic-camelbert-mix    | 0.09          | 0.24            | 0.64     | 0.89    | 0.82                    |
| aubmindlab/bert-base-arabertv2              | 0.14          | 0.25            | 0.61     | 0.87    | 0.80                    |
| Bert After Translation                      | 0.09          | 0.23            | 0.59     | 0.87    | 0.79                    |

## Contributing

Contributions to this project are welcome! Please fork the repository and submit pull requests with your proposed changes.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- Arabert and Marabert teams for providing pre-trained models.
- OpenAI for the ChatGPT model.

