from sentiment-analysis/sentimentanalysis import SentimentAnalysisBidirectionalLSTMTemperature
import torch
import re
import numpy as np
import torchtext.vocab as vocab
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
glove = vocab.GloVe(name='6B', dim=100)

embedding_matrix = torch.load('sentiment-analysis/glove_embeddings.pt')

sentiment_analyzer = SentimentAnalysisBidirectionalLSTMTemperature(
    embedding_dim=100,
    hidden_dim=256,
    n_layers=2,
    dropout=0.5,
    pretrained_embedding=embedding_matrix,
    init_temp=7.0
)

sentiment_analyzer.to(device)
sentiment_analyzer.load_state_dict(torch.load('sentiment-analysis/combined_model_weights.pth', map_location=device))
sentiment_analyzer.eval()


def predict_text(text, model, max_length):
    def preprocess(s):
        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", '', s)
        # Replace all runs of whitespaces with one space
        s = re.sub(r"\s+", ' ', s)
        # replace digits with no space
        s = re.sub(r"\d", '', s)

        return s
    words = [preprocess(word) for word in text.lower().split()[:max_length]]
    word_indices = [glove.stoi[word] if word in glove.stoi else 0 for word in words]

    if len(word_indices) < max_length:
        word_indices.extend([0] * (max_length - len(word_indices)))

    inputs = torch.tensor(word_indices).unsqueeze(0).to(device)

    batch_size = inputs.size(0)
    h = model.init_hidden(batch_size, device)
    h = tuple([each.data for each in h])

    model.eval()
    with torch.no_grad():
        output, h = model(inputs, h)
        prediction = torch.softmax(output, dim=1).cpu().numpy()

    label_mapping = {2: 'Positive', 1: 'Neutral', 0: 'Negative'}
    predicted_class = label_mapping[np.argmax(prediction)]
    predicted_probabilities = prediction[0][np.argmax(prediction)]

    return predicted_class, predicted_probabilities

with open ('random_forest_model.pkl', 'rb') as f:
    rf = pickle.load(f)

def fetch_inputs(ticker, api_key='0f1ad038aa3c47c180c5d9070c0eb6ca'):
    def fetch_indicators(ticker):
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo", interval='5m')
        hist['RSI'] = ta.rsi(hist['Close'], length=14)

        # Using pandas_ta to calculate MACD
        macd = hist.ta.macd(close='Close', fast=12, slow=26, signal=9)
        hist = pd.concat([hist, macd], axis=1)

        hist = hist.reset_index()
        hist.rename(columns={'Datetime': 'datetime'}, inplace=True)

        # Drop the last row as it has a NaN value for 'y'
        hist = hist.dropna()

        return hist[['datetime', 'Close', 'RSI', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']]

    def fetch_news_api(ticker, api_key):
        # Ensure the date is formatted correctly
        date_str = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
        url_base = f'https://newsapi.org/v2/everything?q={ticker}&from={date_str}&sortBy=publishedAt&language=en&apiKey={api_key}'

        news_data = []
        page = 1
        total_pages = 1  # Assume there's at least one page

        while page <= total_pages:
            url = f"{url_base}&page={page}"
            response = requests.get(url)
            response_json = response.json()

            # Check if this is the first request to determine the total pages available
            if page == 1:
                total_results = response_json.get('totalResults', 0)
                total_pages = (total_results // 20) + 1  # Assuming default pageSize is 20, adjust as per actual pageSize

            articles = response_json.get('articles', [])

            for article in articles:
                # Filter out headlines with non-ASCII characters
                if all(ord(char) < 128 for char in article['title']):
                    news_data.append({
                        'datetime': pd.to_datetime(article['publishedAt']),
                        'headline': article['title']
                    })

            page += 1  # Increment to fetch the next page

        df = pd.DataFrame(news_data)
        return df

    def combine_with_api_news(ticker, api_key):
        api_news = fetch_news_api(ticker, api_key)  # Fetch news using the API
        indicators = fetch_indicators(ticker) # Fetch indicators using the function defined above

        api_news['datetime'] = api_news['datetime'].dt.tz_localize(None)
        indicators['datetime'] = indicators['datetime'].dt.tz_localize(None)

        combined_df = pd.merge_asof(api_news.sort_values('datetime'), indicators.sort_values('datetime'), on='datetime', direction='backward')
        return combined_df

    combine = combine_with_api_news(ticker, api_key)
    label_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}

    combine['headline'] = combine['headline'].astype(str).apply(predict_text, args=(sentiment_analyzer, 15))
    combine['headline'] = combine['headline'].apply(lambda x: x[0])
    combine['headline'] = combine['headline'].map(label_mapping)

    return combine.drop('datetime', axis=1)

ticker = input('Enter a ticker: ')
prediction = 'Increase' if rf.predict(fetch_inputs(ticker))[0] else 'Decrease'
print(prediction)
