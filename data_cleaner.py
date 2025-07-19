import pandas as pd
import numpy as np
import re

def clean_amazon_data(csv_file='amazon_soft_toys_enhanced.csv'):
    """Comprehensive data cleaning pipeline"""
    
    print("Starting data cleaning process...")
    
    # Load the data
    df = pd.read_csv(csv_file)
    print(f"Initial dataset: {len(df)} products")
    
    # Display current data state
    print("\nData Overview:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
    # 1. Remove Duplicates
    print("\n=== REMOVING DUPLICATES ===")
    initial_count = len(df)
    
    # Remove exact duplicates based on title and brand
    df = df.drop_duplicates(subset=['title', 'brand'], keep='first')
    
    # Remove similar titles (fuzzy matching)
    df = remove_similar_products(df)
    
    duplicates_removed = initial_count - len(df)
    print(f"Removed {duplicates_removed} duplicates. Remaining: {len(df)} products")
    
    # 2. Clean Price Data
    print("\n=== CLEANING PRICE DATA ===")
    df['selling_price_original'] = df['selling_price'].copy()  # Backup
    df['selling_price'] = df['selling_price'].apply(clean_price)
    
    # Handle zero or invalid prices
    invalid_prices = df[df['selling_price'] <= 0]
    print(f"Found {len(invalid_prices)} products with invalid prices")
    
    # Remove products with no valid price data
    df = df[df['selling_price'] > 0]
    print(f"After price cleaning: {len(df)} products")
    
    # 3. Clean Rating Data
    print("\n=== CLEANING RATING DATA ===")
    df['rating_original'] = df['rating'].copy()  # Backup
    df['rating'] = df['rating'].apply(clean_rating)
    
    # Validate rating range (0-5)
    df['rating'] = df['rating'].clip(0, 5)
    print(f"Rating range: {df['rating'].min()} - {df['rating'].max()}")
    
    # 4. Clean Reviews Data
    print("\n=== CLEANING REVIEWS DATA ===")
    df['reviews_original'] = df['reviews'].copy()  # Backup
    df['reviews'] = df['reviews'].apply(clean_reviews)
    
    # 5. Clean Brand Data
    print("\n=== CLEANING BRAND DATA ===")
    df['brand_original'] = df['brand'].copy()  # Backup
    df['brand'] = df['brand'].apply(clean_brand_name)
    
    # 6. Clean Title Data
    print("\n=== CLEANING TITLE DATA ===")
    df['title_original'] = df['title'].copy()  # Backup
    df['title'] = df['title'].apply(clean_title)
    
    # 7. Validate URLs
    print("\n=== VALIDATING URLS ===")
    df['image_url'] = df['image_url'].apply(clean_url)
    df['product_url'] = df['product_url'].apply(clean_url)
    
    # 8. Data Type Conversion
    print("\n=== CONVERTING DATA TYPES ===")
    df = convert_data_types(df)
    
    # 9. Handle Missing Values
    print("\n=== HANDLING MISSING VALUES ===")
    df = handle_missing_values(df)
    
    # 10. Create Additional Features
    print("\n=== CREATING ADDITIONAL FEATURES ===")
    df = create_additional_features(df)
    
    # Final validation
    print("\n=== FINAL VALIDATION ===")
    print(f"Final dataset: {len(df)} products")
    print("\nData types after cleaning:")
    print(df.dtypes)
    
    print("\nCleaned data summary:")
    print(df.describe())
    
    # Save cleaned data
    df.to_csv('amazon_soft_toys_cleaned.csv', index=False)
    print("\nCleaned data saved to 'amazon_soft_toys_cleaned.csv'")
    
    return df

def clean_price(price):
    """Clean price data - remove symbols, convert to numeric"""
    if pd.isna(price) or price == 0:
        return 0
    
    # Convert to string if not already
    price_str = str(price)
    
    # Remove currency symbols, commas, and spaces
    price_cleaned = re.sub(r'[₹$,\s]', '', price_str)
    
    # Extract numbers only
    numbers = re.findall(r'\d+', price_cleaned)
    
    if numbers:
        return int(''.join(numbers))
    else:
        return 0

def clean_rating(rating):
    """Clean rating data - ensure numeric and valid range"""
    if pd.isna(rating):
        return 0.0
    
    # Convert to string to handle various formats
    rating_str = str(rating)
    
    # Extract numeric rating
    match = re.search(r'(\d+\.?\d*)', rating_str)
    if match:
        rating_val = float(match.group(1))
        # Ensure rating is between 0 and 5
        return min(5.0, max(0.0, rating_val))
    
    return 0.0

def clean_reviews(reviews):
    """Clean review count - remove commas and convert to numeric"""
    if pd.isna(reviews):
        return 0
    
    # Convert to string
    reviews_str = str(reviews)
    
    # Remove commas and extract numbers
    numbers = re.sub(r'[,\s]', '', reviews_str)
    match = re.search(r'\d+', numbers)
    
    return int(match.group()) if match else 0

def clean_brand_name(brand):
    """Standardize brand names"""
    if pd.isna(brand) or brand == "":
        return "Unknown"
    
    brand_str = str(brand).strip()
    
    # Common brand name standardizations for soft toys
    brand_mapping = {
        'dimpy': 'Dimpy Stuff',
        'gund': 'GUND',
        'ty': 'Ty',
        'melissa': 'Melissa & Doug',
        'vtech': 'VTech',
        'fisher': 'Fisher-Price',
        'hasbro': 'Hasbro',
        'mattel': 'Mattel',
        'disney': 'Disney'
    }
    
    brand_lower = brand_str.lower()
    for key, value in brand_mapping.items():
        if key in brand_lower:
            return value
    
    # Capitalize first letter of each word
    return ' '.join(word.capitalize() for word in brand_str.split())

def clean_title(title):
    """Clean product titles"""
    if pd.isna(title):
        return "Unknown Product"
    
    title_str = str(title).strip()
    
    # Remove excessive whitespace
    title_cleaned = re.sub(r'\s+', ' ', title_str)
    
    # Remove special characters at the end
    title_cleaned = re.sub(r'[^\w\s\-&().,]$', '', title_cleaned)
    
    return title_cleaned

def clean_url(url):
    """Clean and validate URLs"""
    if pd.isna(url) or url == "":
        return ""
    
    url_str = str(url).strip()
    
    # Ensure Amazon URLs are complete
    if url_str.startswith('/'):
        return f"https://www.amazon.in{url_str}"
    elif not url_str.startswith('http'):
        return f"https://{url_str}"
    
    return url_str

def remove_similar_products(df):
    """Remove products with very similar titles"""
    from difflib import SequenceMatcher
    
    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    indices_to_remove = set()
    
    for i in range(len(df)):
        if i in indices_to_remove:
            continue
            
        for j in range(i + 1, len(df)):
            if j in indices_to_remove:
                continue
                
            if similarity(df.iloc[i]['title'], df.iloc[j]['title']) > 0.85:
                # Keep the one with more reviews
                if df.iloc[i]['reviews'] >= df.iloc[j]['reviews']:
                    indices_to_remove.add(j)
                else:
                    indices_to_remove.add(i)
                    break
    
    print(f"Removed {len(indices_to_remove)} similar products")
    return df.drop(list(indices_to_remove)).reset_index(drop=True)

def convert_data_types(df):
    """Convert columns to appropriate data types"""
    
    # Numeric columns
    numeric_columns = ['selling_price', 'rating', 'reviews']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # String columns
    string_columns = ['title', 'brand', 'image_url', 'product_url']
    for col in string_columns:
        df[col] = df[col].astype(str)
    
    return df

def handle_missing_values(df):
    """Handle remaining missing values"""
    
    # Fill missing ratings with 0
    df['rating'] = df['rating'].fillna(0)
    
    # Fill missing reviews with 0
    df['reviews'] = df['reviews'].fillna(0)
    
    # Fill missing prices with median price
    df['selling_price'] = df['selling_price'].fillna(df['selling_price'].median())
    
    # Fill missing brands
    df['brand'] = df['brand'].fillna('Unknown')
    
    return df

def create_additional_features(df):
    """Create additional features for analysis"""
    
    # Price categories
    df['price_category'] = pd.cut(df['selling_price'], 
                                 bins=[0, 500, 1000, 2000, float('inf')], 
                                 labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
    
    # Rating categories
    df['rating_category'] = pd.cut(df['rating'], 
                                  bins=[0, 3, 4, 4.5, 5], 
                                  labels=['Poor', 'Average', 'Good', 'Excellent'])
    
    # Review volume categories
    df['review_volume'] = pd.cut(df['reviews'], 
                                bins=[0, 50, 200, 1000, float('inf')], 
                                labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Calculate value score (rating/price ratio)
    df['value_score'] = df['rating'] / (df['selling_price'] / 1000)
    df['value_score'] = df['value_score'].replace([np.inf, -np.inf], 0)
    
    # Title length
    df['title_length'] = df['title'].str.len()
    
    return df

if __name__ == "__main__":
    # Run the cleaning process
    cleaned_data = clean_amazon_data('amazon_soft_toys_enhanced.csv')
    
    print("\n=== CLEANING COMPLETE ===")
    print("Next: Run the analysis code (Part 3)")
