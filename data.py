from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd

def get_news_dataframe(ticker):
    """
    Retrieves news headlines for the given ticker symbol from Finviz and returns a pandas DataFrame.
    """
    finviz_url = 'https://finviz.com/quote.ashx?t='
    url = finviz_url + ticker

    request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        response = urlopen(request)
    except Exception as e:
        print(f"Error accessing URL: {e}")
        return None

    soup = BeautifulSoup(response, 'html.parser')
    news_table = soup.find(id='news-table')

    if news_table:
        date_times = []
        headlines = []
        for row in news_table.findAll('tr'):
            columns = row.findAll('td')
            if len(columns) > 1:
                date_time = columns[0].get_text(strip=True)
                headline = columns[1].get_text(strip=True)
                date_times.append(date_time)
                headlines.append(headline)

        df = pd.DataFrame({
            'Date/Time': date_times,
            'Headline': headlines
        })

        return df
    else:
        print(f"No news table found for ticker symbol: {ticker}.")
        return None

while True:
    ticker = input("Enter the ticker symbol (e.g., AAPL): ").strip().upper()
    news_df = get_news_dataframe(ticker)

    if news_df is not None:
        print("\nDataFrame:")
        print(news_df)
        break 
    else:
        print("Please try again.")
