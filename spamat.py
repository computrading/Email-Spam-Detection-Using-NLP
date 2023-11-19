import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
data = pd.read_csv('./dataset/spamails.csv')
def get_sequences(texts, tokenizer, train=True, max_seq_length=None):
    sequences = tokenizer.texts_to_sequences(texts)
    if train == True:
        max_seq_length = np.max(list(map(lambda x: len(x), sequences)))
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq_length, padding='post')
    return sequences

def preprocess_inputs(df):
    df = df.copy()
    # Drop FILE_NAME column
    df = df.drop('FILE_NAME', axis=1)
    # Split df into X and y
    y = df['CATEGORY']
    X = df['MESSAGE']
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    # Create tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=30000)
    # Fit the tokenizer
    tokenizer.fit_on_texts(X_train)
    # Convert texts to sequences
    X_train = get_sequences(X_train, tokenizer, train=True)
    X_test = get_sequences(X_test, tokenizer, train=False, max_seq_length=X_train.shape[1])
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_inputs(data)
inputs = tf.keras.Input(shape=(14804,))
embedding = tf.keras.layers.Embedding(
    input_dim=30000,
    output_dim=64
)(inputs)
flatten = tf.keras.layers.Flatten()(embedding)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(flatten)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc')
    ]
)
print(model.summary())
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)
model.save("models/email_classification.h5")

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=30000)
# Replace 'new_email_text' with the actual text of the new email you want to classify
new_email_text = "homeowner best rate help . contact us!"
# Tokenize and preprocess the new email text
print("X_train",X_train)
new_sequences = get_sequences([new_email_text], tokenizer, train=False, max_seq_length=X_train.shape[1])
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
