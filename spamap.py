import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def get_sequences(texts, tokenizer, max_seq_length=14804):
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq_length, padding='post')
    return sequences

model =load_model('models/email_classification.h5')
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=30000)
# Replace 'new_email_text' with the actual text of the new email you want to classify
new_email_text = "you won a million dollars send us your bank account for transfer the money"
# Tokenize and preprocess the new email text
new_sequences = get_sequences([new_email_text], tokenizer)
# Make predictions using the loaded model
prediction = model.predict(new_sequences)
print(prediction)
# Since this is a binary classification, you can consider a threshold for deciding the class
threshold = 0.5
predicted_class = 1 if prediction > threshold else 0
# Print the prediction result
if predicted_class == 1:
    print("The email is predicted as spam.")
else:
    print("The email is predicted as not spam.")
