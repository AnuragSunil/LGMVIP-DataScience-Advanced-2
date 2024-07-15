# Next Word Prediction with LSTM Neural Network

This project focuses on predicting the next word in a sequence using an LSTM (Long Short-Term Memory) neural network. As an advanced task, I extended the project to generate text using the trained model. The dataset used is the text from a public domain book.

## Project Structure

- `1661-0.txt`: The text file containing the training data.
- `mymodel.keras`: The saved Keras model after training.

## Dependencies

- Python 3.x
- TensorFlow
- NumPy
- NLTK
- scikit-learn

Install the required libraries using:

```bash
pip install tensorflow numpy nltk scikit-learn
```

## Data Preprocessing

The text data is preprocessed by:

- Removing non-alphabetic characters and converting to lowercase.
- Tokenizing the text into words.
- Creating a mapping of unique tokens to indices.
- Generating input-output pairs for the model where input is a sequence of 10 words and output is the next word.

## Model Architecture

The model used is a Sequential LSTM neural network with the following architecture:

- Two LSTM layers with 128 units each.
- Dense layer with the number of units equal to the number of unique tokens.
- Softmax activation for the output layer.

## Training

The model is trained using the categorical crossentropy loss function and RMSprop optimizer. The data is split into training and validation sets to evaluate validation accuracy.

## Next Word Prediction

The primary task of this project is to predict the next word in a sequence given the previous 10 words. The model is designed to take an input sequence and predict the most likely next word.

## Advanced Task: Text Generation

As a more advanced task, I extended the project to generate text. The model can generate text given a seed input by predicting the next word based on the previous 10 words and appending it to the input sequence iteratively.

## Acknowledgements

This project was completed as part of the Let's Grow More Virtual Internship Program. Special thanks to the Let's Grow More team for their guidance and support.

