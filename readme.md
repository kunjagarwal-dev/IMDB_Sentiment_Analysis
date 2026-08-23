# 🎬 IMDB Sentiment Analysis

A deep learning project comparing SimpleRNN, LSTM, and GRU architectures for binary sentiment classification on IMDB movie reviews, deployed as an interactive Streamlit app for live sentiment prediction.

## 📊 Results

| Model                     | Best Val Accuracy | Notes                                                                     |
| ------------------------- | ----------------- | ------------------------------------------------------------------------- |
| SimpleRNN                 | ~51%              | Severe overfitting (83.6% train vs ~51% val) — vanishing gradient problem |
| LSTM (+ EarlyStopping)    | ~85–86%           | Well-generalized, stopped early at epoch 7                                |
| **GRU (+ EarlyStopping)** | **~87.6%**        | Best result, stopped early at epoch 10                                    |

## 🗂️ Project Structure

```
imdb-sentiment-analysis/
├── notebooks/
│   └── IMDB_analysis.ipynb   # Data loading + all 3 models + comparison
├── models/
│   ├── imdb_gru_model.h5
│   └── imdb_word_index.json
├── app/
│   └── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧠 Models

All three models share the same structure — `Embedding(10000, 32) → [Recurrent Layer](32) → Dense(1, sigmoid)` — differing only in the recurrent layer used.

**SimpleRNN:** Trained for 10 fixed epochs. Showed a dramatic overfitting gap — training accuracy climbed to 83.6% while validation accuracy stayed flat around 51% (essentially random guessing for this binary task), with validation loss actively worsening over time. This is a textbook demonstration of the **vanishing gradient problem**: at 200 timesteps, gradients from early words in a review shrink too much during backpropagation for the RNN to learn generalizable long-range patterns, so it ends up memorizing training noise instead.

**LSTM:** Same architecture, swapping in an LSTM cell. Its gating mechanism (input, forget, and output gates) preserves gradient flow across long sequences far better than a plain RNN. Immediately closed the overfitting gap — 88.7% train / 84.8% val accuracy in a fixed 10-epoch run. Adding `EarlyStopping` (patience 4, restoring best weights) and allowing up to 30 epochs improved this further, with training automatically stopping once validation loss stopped improving.

**GRU:** A simpler, faster gating mechanism than LSTM (fewer gates, fewer parameters). Achieved the best result of the three (~87.6% val accuracy), showing that GRU can match or exceed LSTM performance on this task while training faster per epoch.

## 🚀 Running the App

```
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Paste any movie review into the text box and get a live sentiment prediction (positive/negative) with a confidence score, powered by the trained GRU model.

## 🛠️ Tech Stack

TensorFlow / Keras, Streamlit, NumPy, Matplotlib, Seaborn, scikit-learn

## 📈 Skills Demonstrated

- Text preprocessing for deep learning: tokenization, padding, word embeddings
- Recurrent neural network architectures: SimpleRNN, LSTM, GRU
- Diagnosing the vanishing gradient problem through direct empirical comparison
- Regularization for sequence models via early stopping
- Building a text-to-integer encoding pipeline for real-world deployment (handling Keras's IMDB index-offset quirk)
- Interactive NLP model deployment with Streamlit
