from sec_edgar_downloader import Downloader
import os
DATA_DIR = "E:/ai-projects/rag_finance/data"
TICKERS = ["AAPL","MSFT","NVDA"]

dl = Downloader("YourName","[EMAIL_ADDRESS]",DATA_DIR)

for ticker in TICKERS:
    print(f"Downloading latest 10-k for {ticker}....")
    dl.get("10-K", ticker, limit=1, download_details=True)

print("\n--- Verification ---")
base = os.path.join(DATA_DIR, "sec-edgar-filings")
for ticker in os.listdir(base):
    path = os.path.join(base, ticker, "10-K")
    filings = os.listdir(path)
    print(f"{ticker} → {len(filings)} filing(s): {filings}")
print("Done.")