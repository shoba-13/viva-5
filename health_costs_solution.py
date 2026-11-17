# ===== Paste into the matching Colab cells =====
# Cells 1-2 (imports + data download) are given. The dataframe is called `dataset`.
# Columns: age, sex, bmi, children, smoker, region, expenses

# ---------------------------------------------------------------- cell 3: prepare data
# Convert categorical columns to numbers (one-hot encoding)
dataset = pd.get_dummies(dataset, columns=['sex', 'smoker', 'region'],
                         drop_first=True, dtype=float)

# 80% train / 20% test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Pop the label column off both datasets
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

# ---------------------------------------------------------------- cell 4: build model
# Normalization layer lives inside the model, so raw test data works in the final cell
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_dataset))

model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mae',
              metrics=['mae', 'mse'])

model.summary()

# ---------------------------------------------------------------- cell 5: train
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=25, restore_best_weights=True)

history = model.fit(
    train_dataset,
    train_labels,
    epochs=300,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0)

# ---------------------------------------------------------------- cell 6: given
# (leave the final cell unchanged - it runs model.evaluate and plots predictions)
