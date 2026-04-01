# SMS Spam Detection — NLP & Deep Learning

End-to-end text classification pipeline to detect spam in SMS messages, using **TF-IDF**, a **Deep Autoencoder**, and **Logistic Regression** — with a BERT-based model implemented as an advanced extension.

---

## What This Project Does

This project builds a complete NLP pipeline that:
1. Preprocesses raw SMS text using NLTK (stopword removal, Porter stemming)
2. Vectorizes messages using TF-IDF (5,000 features)
3. Compresses features with a Deep Autoencoder (5000 → 128 dimensions)
4. Trains a Logistic Regression classifier with cost-sensitive cross-validation
5. Implements a BERT-based classifier (HuggingFace + PyTorch) as an advanced NLP extension

**Key results:**
- TF-IDF + Logistic Regression: **97.8% accuracy, 91.8% F1-score**
- Autoencoder-compressed features: 86.76% accuracy, 62.86% F1-score
- Custom cost scorer: false positives penalized 5× to minimize false alarm rate

---

## Dataset

| | |
|---|---|
| **Name** | SMS Spam Collection v.1 |
| **Source** | UCI Machine Learning Repository |
| **Link** | https://archive.ics.uci.edu/dataset/228/sms+spam+collection |
| **Size** | 5,572 messages |
| **Balance** | 86.6% ham (legitimate) · 13.4% spam |
| **Format** | Tab-separated (label + raw SMS text) |

> The dataset is not included in this repo. Download it from the UCI link above.

---

## Technologies Used

- Python 3
- scikit-learn (TF-IDF, Logistic Regression, GridSearchCV, StratifiedKFold)
- TensorFlow / Keras (Deep Autoencoder)
- NLTK (stopwords, PorterStemmer)
- HuggingFace Transformers + PyTorch (BERT)
- Pandas, NumPy
- Jupyter Notebook / Google Colab

---

## Files

| File | Description |
|---|---|
| `sms_preprocessing.ipynb` | **Step 1** — Text preprocessing: cleaning, NLTK stopword removal, Porter stemming, label encoding |
| `sms_spam_detection.ipynb` | **Step 2** — Full pipeline: TF-IDF vectorization, Deep Autoencoder (Keras), Logistic Regression with GridSearchCV, custom cost scorer, BERT extension |

---

## Pipeline Overview

```
Raw SMS messages (5,572)
        ↓
sms_preprocessing.ipynb   →  NLTK cleaning (stopwords + stemming)
        ↓
sms_spam_detection.ipynb
   ├── TF-IDF (5000 features)
   ├── Deep Autoencoder: 5000 → 512 → 256 → 128 → 256 → 512 → 5000
   ├── Logistic Regression (5-fold StratifiedKFold + GridSearchCV)
   │        └── Custom scorer: FP penalized 5×
   └── BERT classifier (HuggingFace + PyTorch) — advanced extension
```

---

## Model Architecture — Deep Autoencoder

```
Encoder: 5000 → 512 → 256 → 128
Bottleneck: 128-dimensional dense representation
Decoder: 128 → 256 → 512 → 5000
```

The 128-d bottleneck features are extracted and passed to the Logistic Regression classifier.

---

## How to Run

**1. Install dependencies**
```bash
pip install scikit-learn tensorflow keras nltk pandas numpy transformers torch
```

**2. Download the dataset** from the UCI link above
Rename the file to: `SMSSpamCollection`

**3. Run the preprocessing notebook**
```bash
jupyter notebook sms_preprocessing.ipynb
```

**4. Run the full detection pipeline**
```bash
jupyter notebook sms_spam_detection.ipynb
```


---

## Author

**Fares Sghaier** — Applied Data Science, La Cité College, Ottawa, Canada
