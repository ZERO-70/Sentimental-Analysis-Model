# 🧠 Transformer-based (BERT) Sentiment Analysis

This repository contains an implementation of a Transformer-based sentiment classification model using PyTorch. The model is trained on the Yelp Review Full dataset to classify user reviews into 5 sentiment categories.

## 📚 Features

* Custom Transformer encoder from scratch (no pre-trained models)
* Tokenization and vocabulary building
* Positional encoding and multi-head attention
* Training and evaluation on the Yelp Review Full dataset
* Accuracy and loss plots
* Model checkpointing for resuming training

## 🗂️ File Structure

```
📦 Transformer-Sentiment
├── New_Sentiment.ipynb       # Main notebook with data processing, model training, and evaluation
├── model_checkpoint.pt       # (Optional) Saved model state for resuming training
└── README.md                 # Project overview
```

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Transformer-Sentiment.git
cd Transformer-Sentiment
```

2. Open `New_Sentiment.ipynb` in Jupyter Notebook or Google Colab.

3. Run the notebook cells to:

   * Load the dataset
   * Build the vocabulary
   * Define the model
   * Train and evaluate

> 💡 Tip: To run on Google Colab, make sure to mount your Google Drive if using model checkpointing.

## 🧪 Dataset

* **Yelp Review Full**: 5-class sentiment classification (1 to 5 stars)
* Loaded using `torchtext.datasets.YelpReviewFull`
* Preprocessed with tokenization and vocabulary building

## 🏗️ Model Overview

* Embedding Layer
* Positional Encoding
* N layers of Transformer Encoder blocks
* Global average pooling
* Fully connected classification head

## 📊 Results

The model achieves competitive accuracy on the Yelp Review Full dataset, showing the potential of custom Transformers without pretraining.

| Metric   | Value (Example) |
| -------- | --------------- |
| Accuracy | \~63%           |
| Loss     | \~1.2           |

> Note: Actual results may vary based on hyperparameters and training duration.

## 💾 Checkpointing

Model checkpoints are saved using `torch.save()` and can be resumed by loading the `.pt` file:

```python
checkpoint = torch.load("model_checkpoint.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## 🧰 Requirements

* Python 3.8+
* PyTorch
* Torchtext
* Matplotlib
* NumPy
* tqdm

You can install dependencies via:

```bash
pip install torch torchtext matplotlib numpy tqdm
```

## ✍️ Author

* Your Name ([@ZERO-70](https://github.com/ZERO-70))

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.
