# Sentiment Analysis of Tweets

This project aims to build a sentiment analysis model that classifies tweets as either positive or negative.

## Project Structure

The project consists of the following files:

- **index.html:** Front-end HTML template for the sentiment analysis web application.
- **model_training.ipynb:** Jupyter Notebook containing code for data loading, preprocessing, model training, and evaluation.
- **utils.py:** Python module with utility functions for tweet processing and model definition.
- **app.py:** Flask application for hosting the sentiment analysis API.

## Libraries and Technologies

### Libraries:

- **NLTK (Natural Language Toolkit):** For text processing tasks like tokenization, stemming, and stop word removal.
- **TensorFlow/Keras:** For building and training the neural network model.
- **Scikit-learn:** For evaluation metrics and model performance analysis.
- **Flask:** For building the web application and API.
- **Axios:** For making API requests from the front-end.

### Technologies:

- **Python:** Programming language used for the project.
- **HTML, CSS, JavaScript:** Front-end technologies for the web application.

## Model Training

1. **Data Collection:** The model is trained on a dataset of positive and negative tweets from the NLTK Twitter corpus.
2. **Data Preprocessing:** Tweets are preprocessed by removing special characters, punctuation, stop words, and applying stemming.
3. **Model Architecture:** A simple neural network with an embedding layer, a hidden layer, and an output layer.
4. **Training:** The model is trained using the Adam optimizer and binary cross-entropy loss function.
5. **Evaluation:** The model is evaluated on a validation dataset using metrics like accuracy, precision, recall, and F1-score.
6. **Model Saving:** The trained model is saved for future use.

## Results

The sentiment analysis model achieved an accuracy of **99%** on the validation dataset. Though this model indicates high accuracy but 
the model fails to capture the actual meaning of the tweets and inclines heavily towards emojis and count of words.

## Usage
1. Enter a tweet in the text area provided on the webpage.
2. Click the "Submit" button.
3. The model will analyze the sentiment of the tweet and display the result as either "Positive Statement" or "Negative Statement."

## Future Improvements
* **Advanced Model Architecture**: Explore more complex neural network architectures for better performance.
* **Fine-tuning**: Fine-tune the model on specific domains to improve accuracy for relevant topics.
* **Real-time Sentiment Analysis**: Integrate the model with a real-time streaming platform to analyze live tweets.