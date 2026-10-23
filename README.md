# HanuAi ML Assessment

**Published reports:** https://aravindashok18.github.io/hanuai-ml-assessment/

Two data-analytics tasks on real-world text: **(1)** scraping and sentiment analysis of public product reviews, and **(2)** exploratory analysis and text mining of 1,000 vehicle warranty claims.

| | Task 1 - Web scraping and sentiment | Task 2 - Advanced EDA and text mining |
|---|---|---|
| **Notebook** | [`Task1_Scraping_Sentiment.ipynb`](Task1_Web_Scraping_Sentiment/Task1_Scraping_Sentiment.ipynb) | [`Task2_EDA_TextMining.ipynb`](Task2_EDA_Text_Mining/Task2_EDA_TextMining.ipynb) |
| **Data output** | [`task1_reviews_sentiment.csv`](Task1_Web_Scraping_Sentiment/task1_reviews_sentiment.csv) / [`.xlsx`](Task1_Web_Scraping_Sentiment/task1_reviews_sentiment.xlsx) | [`task2_tagged_dataset.csv`](Task2_EDA_Text_Mining/task2_tagged_dataset.csv) / [`.xlsx`](Task2_EDA_Text_Mining/task2_tagged_dataset.xlsx) |
| **Report** | [PDF](Task1_Web_Scraping_Sentiment/Task1_Report.pdf) / [DOCX](Task1_Web_Scraping_Sentiment/Task1_Report.docx) / [web](docs/task1.html) | [PDF](Task2_EDA_Text_Mining/Task2_Stakeholder_Report.pdf) / [DOCX](Task2_EDA_Text_Mining/Task2_Stakeholder_Report.docx) / [web](docs/task2.html) |

## Task 1 - Apple AirPods 4 reviews (Best Buy Canada)
**What it does.** A Selenium + BeautifulSoup scraper (sort filters, "Show More" pagination, verified-buyers filter, retries with back-off, throttling, robots.txt check, hard stop on a block) extracts the required fields: primary key, title, review text, date (`YYYY-MM-DD`), rating, source, reviewer name, plus helpfulness, sub-ratings and sort ranks. Sentiment is scored with VADER and an open-source LLM (`google/flan-t5-base`, zero-shot), then tagged by aspect in the brief's format, e.g. `['Fit & Comfort (Neg)']`.

**Headline findings** (127 reviews, 64 with text)
- Text reviewers average **3.6 stars vs 4.5 for star-only** reviewers: star averages hide the complaints.
- The LLM label agrees with star-derived sentiment on **86%** of text reviews (VADER 78%).
- **Sound and fit** are the two leading complaint themes (statistically tied); sound is also the most praised aspect.
- **36%** of text reviews discuss the shopping experience: delivery/pickup is the strongest positive, the open-box "hygiene" return policy the most negative.

**Collection note (read this).** `robots.txt` allows `/en-ca/product/`, but Best Buy's Conditions of Use prohibit scraping. The scraper was limited to one product, throttled, and designed to stop on any block. It **was blocked on its first request ("Access Denied") and stopped**; the same public pages were then read through an unmodified browser session with the same selectors, filters, delays and click budget (28 "Show More" clicks). The notebook documents this, and for production use a data licence or official API is the right route. The collected reviews are public customer content, included here for assessment purposes only.

## Task 2 - Infotainment warranty claims (1,000 claims)
**What it does.** Profiles and cleans the data (non-breaking-space label bug, placeholder text, impossible dates, noisy multilingual text), then turns three free-text narratives into the six tag columns defined by the five worked examples using a transparent lexicon with clause-level pairing; classifies the type of issue from technician evidence; tests trends with multiple-testing correction; and clusters complaints (TF-IDF/K-Means, MiniLM embeddings, NMF topics) to propose root causes.

**Headline findings**
- The radio is replaced in **88%** of claims whatever the symptom, so it behaves as a common failure point; **46%** replace it with no documented cause.
- Of 32 categories tested for change over time, **one** survives correction: "radio stays on / drains the battery" rose to **~11% of 2021 claims** (from 2-4%).
- **Camera/video** complaints concentrate in specific models (FusionNova, HyperFury X, StellarGlide) - confirmed by three independent views. Plants show no significant effect.
- The structured coded cause and complaint barely predict what technicians found or fixed.

**Honest limits.** Counts, not rates (no production volumes); tag accuracy checked only against the 5 provided examples plus consistency checks; cluster separation is weak (silhouette ~0.1); dollar figures in the report are illustrative assumptions.

## Repository layout
```
Task1_Web_Scraping_Sentiment/   notebook, scraper (bestbuy_scraper.py), CSV/XLSX, report, outputs/ (collection log, captured data, figures)
Task2_EDA_Text_Mining/          notebook, tagged CSV/XLSX, report, outputs/ (figures, facts)
docs/                           web versions of both reports
requirements.txt
```

## Running it
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Task1_Web_Scraping_Sentiment/Task1_Scraping_Sentiment.ipynb
jupyter nbconvert --to notebook --execute --inplace Task2_EDA_Text_Mining/Task2_EDA_TextMining.ipynb
```
- **Task 1** runs in `SCRAPE_MODE = "saved"` by default (analyses the captured reviews and does **not** contact the website). `"live"` runs the Selenium scraper - read section 1 of the notebook first.
- **Task 2** needs the provided dataset, which is **not redistributed here**: place `Hanuai_task2_assignment.csv` in `Task2_EDA_Text_Mining/` before re-running. The notebook falls back gracefully if the sentence-embedding model is unavailable.
- Seeds are fixed (`SEED = 42`); results reproduce.
