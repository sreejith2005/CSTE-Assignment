# Amazon Soft Toys Scraper & Analyzer

Complete data pipeline for scraping, cleaning, and analyzing Amazon India soft toy products.

## Repository Contents

### Python Scripts
- **`amazon_scraper.py`** - Enhanced web scraper with anti-detection features for Amazon sponsored products
- **`data_cleaner.py`** - Data cleaning and preprocessing pipeline with feature engineering
- **`data_analyzer.py`** - Comprehensive analysis engine with statistical insights and visualizations

### Data Files
- **`amazon_soft_toys_enhanced.csv`** - Raw scraped data (9 sponsored products)
- **`amazon_soft_toys_cleaned.csv`** - Cleaned dataset with engineered features
- **`brand_summary_analysis.csv`** - Aggregated brand performance metrics

### Visualizations
- **`brand_performance_analysis.png`** - Brand frequency, market share, ratings, and value scores
- **`price_rating_analysis.png`** - Price-rating correlation, distribution, and value analysis
- **`review_rating_distribution.png`** - Review volume and rating distribution patterns

## Features

### Scraping Capabilities
- Bypasses basic anti-bot detection with rotating headers
- Targets sponsored products specifically
- Extracts: title, brand, rating, reviews, price, image URL, product URL
- Rate limiting and error handling

### Data Processing
- Removes duplicates and similar products
- Standardizes price, rating, and review formats
- Creates price categories (Budget/Mid-Range/Premium/Luxury)
- Generates value scores and additional metrics

### Analysis Output
- **Brand Performance**: Market share, quality metrics, value positioning
- **Price-Rating Correlation**: Relationship analysis with statistical significance
- **Review Distribution**: Customer engagement and trust indicators
- **Market Insights**: Strategic recommendations and competitive gaps

## Requirements

pip install requests beautifulsoup4 pandas matplotlib seaborn numpy fake-useragent


## Usage

python amazon_scraper.py # Scrape data
python data_cleaner.py # Clean and process
python data_analyzer.py # Generate analysis and charts


## Key Findings

- **9 sponsored products** analyzed across 6 brands
- **Hug** dominates with 50% market share (4 products)
- **Mirada** leads quality with 4.5/5 rating
- **Storio** offers best value (12.8 value score)
- Weak price-rating correlation (r=0.445) suggests pricing opportunities
