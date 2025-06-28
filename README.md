# 🐦 Tweet Sentiment Analysis

A complete NLP pipeline that trains a logistic regression model on tweets, and uses the Twitter API to analyze real-time tweet sentiments.

## 💡 Features
- Dataset: Preprocessed tweets from open-source Kaggle dataset
- Text cleaning using `re`, `nltk`, `string`
- Model training using TF-IDF + Logistic Regression
- Real-time tweet fetching using Twitter API v2 (`tweepy`)
- Sentiment prediction and visualization using `seaborn` and `matplotlib`

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/tweet-sentiment-analysis.git
cd tweet-sentiment-analysis 
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
## Usage

1.Replace BEARER_TOKEN with your Twitter developer token.
2.Run the project:

```bash
python sentiment_analysis.py
```
## Example Output

             precision    recall  f1-score   support
   Positive     0.94       0.96      0.95       815
   Negative     0.96       0.94      0.95       857
