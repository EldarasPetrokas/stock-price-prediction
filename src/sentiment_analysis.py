import os
import requests
import pandas as pd
import time
import nltk
from datetime import datetime
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

# Ensure required NLTK components are downloaded
nltk.download("vader_lexicon")

# Set up directories
DATA_DIR = "data"
LOG_DIR = "logs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load FinBERT Model for financial sentiment analysis
finbert = pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")

# Tech stock tickers and their SEC CIK numbers
SEC_CIKS = {
    "AAPL": "0000320193",
    "GOOGL": "0001652044",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
    "META": "0001326801"
}

# SEC EDGAR API Endpoint for filings
SEC_BASE_URL = "https://data.sec.gov/submissions/CIK{}.json"
SEC_FILINGS_URL = "https://www.sec.gov/Archives/"

# Headers to avoid getting blocked
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Keywords to identify important sections
IMPORTANT_SECTIONS = [
    "Management’s Discussion and Analysis",
    "Risk Factors",
    "Business Overview",
    "Results of Operations"
]

# Filings Weight for Sentiment Calculation
FILING_WEIGHTS = {
    "10-K": 1.5,  # More impact (annual)
    "10-Q": 1.2,  # Quarterly impact
    "8-K": 1.0  # Short-term events
}


def fetch_sec_filings(ticker, start_year=2010):
    """Fetch SEC filings (10-K, 10-Q, 8-K) for a stock from EDGAR API"""
    cik = SEC_CIKS.get(ticker)
    if not cik:
        print(f"❌ No CIK found for {ticker}")
        return []

    url = SEC_BASE_URL.format(cik.zfill(10))  # SEC requires 10-digit CIK
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"⚠️ Failed to fetch filings for {ticker}")
        return []

    data = response.json()
    filings = data.get("filings", {}).get("recent", {})

    sec_data = []
    for i in range(len(filings["accessionNumber"])):
        filing_type = filings["form"][i]
        filing_date = filings["filingDate"][i]
        filing_year = int(filing_date.split("-")[0])

        if filing_type in ["10-K", "10-Q", "8-K"] and filing_year >= start_year:
            accession_number = filings["accessionNumber"][i].replace("-", "")
            primary_doc = filings["primaryDocument"][i]

            document_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{primary_doc}"
            sec_data.append((filing_date, filing_type, document_url))

    print(f"📄 {len(sec_data)} filings found for {ticker} from {start_year} onwards.")
    return sec_data


def extract_filing_text(filing_url):
    """Extract key sections from SEC filings"""
    print(f"🔍 Fetching SEC filing from: {filing_url}")
    response = requests.get(filing_url, headers=HEADERS)

    if response.status_code != 200:
        print(f"⚠️ Failed to fetch document from {filing_url}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract key sections
    extracted_text = []
    for tag in soup.find_all(["p", "div"]):
        text = tag.get_text()
        if any(section in text for section in IMPORTANT_SECTIONS) and len(text) > 500:
            extracted_text.append(text)

    # Debugging: Save raw extracted text
    log_file = os.path.join(LOG_DIR, f"extracted_{filing_url.split('/')[-1]}.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(extracted_text))

    if extracted_text:
        print(f"✅ Extracted {len(extracted_text)} relevant sections.")
        return "\n".join(extracted_text)

    return None


def preprocess_text(text):
    """Remove boilerplate, metadata, and clean extracted text"""
    if not text:
        return ""

    # Remove SEC form labels and repetitive structures
    text = text.replace("\n", " ").replace("  ", " ")
    text = text[:3000]  # Limit text size for processing efficiency

    return text


def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)["compound"]
    return round(score, 3)


def analyze_sentiment_finbert(text):
    """Analyze sentiment using FinBERT"""
    if not text:
        return 0

    # Split into multiple calls if text is too long
    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    scores = [finbert([chunk])[0]["label"] for chunk in chunks]

    sentiment_map = {"positive": 1, "neutral": -0.2, "negative": -1}
    sentiment_scores = [sentiment_map[score] for score in scores]

    return round(sum(sentiment_scores) / len(sentiment_scores), 3) if sentiment_scores else 0


def get_sec_sentiment(start_year=2010):
    """Retrieve historical sentiment scores from SEC filings"""
    sentiment_data = []

    for ticker in SEC_CIKS.keys():
        print(f"\n📡 Fetching SEC filings for {ticker} from {start_year} onwards...")
        filings = fetch_sec_filings(ticker, start_year)

        for filing_date, filing_type, filing_url in filings:
            print(f"\n📄 Processing {filing_type} for {ticker} on {filing_date}...")

            filing_text = extract_filing_text(filing_url)
            cleaned_text = preprocess_text(filing_text)
            if not cleaned_text:
                print(f"⚠️ No usable text extracted from {filing_url}")
                continue

            vader_score = analyze_sentiment_vader(cleaned_text)
            finbert_score = analyze_sentiment_finbert(cleaned_text)

            weighted_sentiment = round((vader_score + finbert_score) / 2 * FILING_WEIGHTS.get(filing_type, 1.0), 3)

            sentiment_data.append({
                "Date": filing_date,
                "Ticker": ticker,
                "Filing_Type": filing_type,
                "Weighted_Sentiment": weighted_sentiment,
                "Filing_URL": filing_url
            })

            time.sleep(0.5)

    df = pd.DataFrame(sentiment_data)
    df.to_csv(os.path.join(DATA_DIR, "sec_sentiment.csv"), index=False)
    return df


if __name__ == "__main__":
    sec_sentiment_df = get_sec_sentiment(start_year=2024)
    print(sec_sentiment_df)
