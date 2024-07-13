# Text Generation with LSTM Neural Network

This project demonstrates how to generate text using an LSTM (Long Short-Term Memory) neural network. The dataset used is the text from a public domain book, and the model is trained to predict the next word in a sequence given a context of previous words.

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

## Text Generation

The model can generate text given a seed input. It predicts the next word based on the previous 10 words and appends it to the input sequence.

## Acknowledgements

This project is inspired by various tutorials and examples of text generation using LSTM networks. Special thanks to the authors of these resources.
