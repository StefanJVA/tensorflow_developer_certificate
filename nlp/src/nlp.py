import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow as tf

train_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train', 'test'),
    as_supervised=True)

train_sentences = []
train_labels = []
for sent, label in train_data.take(5000):
    train_sentences.append(sent.numpy().decode("utf-8"))
    train_labels.append(label.numpy())

TEST_SIZE = 512
test_sentences = []
test_labels = []
for sent, label in test_data.take(TEST_SIZE):
    test_sentences.append(sent.numpy().decode("utf-8"))
    test_labels.append(label.numpy())

OOV_TOKEN = '<OOV>'
tokenizier = keras.preprocessing.text.Tokenizer(oov_token=OOV_TOKEN)
tokenizier.fit_on_texts(train_sentences)
word_index = tokenizier.word_index
train_sequnces = tokenizier.texts_to_sequences(train_sentences)
test_sequnces = tokenizier.texts_to_sequences(test_sentences)

# maxlen = max([len(s) for s in train_sequnces])
# print('max len is', maxlen)
maxlen = 1024

train_sequnces_padded = keras.preprocessing.sequence.pad_sequences(train_sequnces, maxlen=maxlen, truncating="post")
test_sequnces_padded = keras.preprocessing.sequence.pad_sequences(test_sequnces, maxlen=maxlen, truncating="post")

train_ds = tf.data.Dataset.from_tensor_slices((train_sequnces_padded, train_labels)).batch(64).prefetch(1)
other_ds = tf.data.Dataset.from_tensor_slices((test_sequnces_padded, test_labels))
test_ds = other_ds.take(TEST_SIZE // 2).batch(64).prefetch(1)
val_ds = other_ds.skip(TEST_SIZE // 2).batch(64).prefetch(1)

# Model 1 : LSTM
lstm_model = keras.Sequential([
    keras.layers.Embedding(len(word_index), 16, input_length=maxlen),
    keras.layers.Bidirectional(keras.layers.LSTM(32, )),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

LSCB = keras.callbacks.TensorBoard('./files/lstm_logs')
MC = tf.keras.callbacks.ModelCheckpoint(
    './files/lstm_model/lstm-{epoch:02d}-{val_loss:.3f}.h5',
    monitor='val_loss',
    save_best_only=False,
    verbose=1
)
LR = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10 ** (epoch / 2), verbose=1)
lstm_model.fit(train_ds, epochs=10, validation_data=test_ds, callbacks=[LSCB, MC, LR])
print('eval lstm', lstm_model.evaluate(test_ds))

# Model 2: CNN
cnn_model = keras.Sequential([
    keras.layers.Embedding(len(word_index), 16, input_length=maxlen),
    keras.layers.Conv1D(128, 5, activation='relu'),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

CNCB = keras.callbacks.TensorBoard('./files/imdb_cnn')
cnn_model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[CNCB])
print('eval cnn', cnn_model.evaluate(test_ds))

# Model 3: GRU
gru_model = keras.Sequential([
    keras.layers.Embedding(len(word_index), 16, input_length=maxlen),
    keras.layers.Bidirectional(keras.layers.GRU(32)),
    keras.layers.Flatten(),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
gru_model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

GRUCB = keras.callbacks.TensorBoard('./files/imdb_gru')
gru_model.fit(train_ds, epochs=10, validation_data=test_ds, callbacks=[GRUCB])
print('eval gru', gru_model.evaluate(test_ds))

print('done')
