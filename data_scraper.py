import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from fake_useragent import UserAgent
import json

class AmazonScraper:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        
    def get_headers(self):
        """Rotate headers to avoid detection"""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
    
    def scrape_sponsored_products(self, keyword="soft toys", max_pages=3):
        products = []
        
        for page in range(1, max_pages + 1):
            print(f"Scraping page {page}...")
            
            # Use multiple URL patterns Amazon uses
            urls_to_try = [
                f"https://www.amazon.in/s?k={keyword.replace(' ', '+')}&page={page}",
                f"https://www.amazon.in/s?k={keyword.replace(' ', '%20')}&ref=sr_pg_{page}",
                f"https://www.amazon.in/s?i=toys&k={keyword.replace(' ', '+')}&page={page}"
            ]
            
            for url in urls_to_try:
                try:
                    response = self.session.get(url, headers=self.get_headers())
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        page_products = self.extract_products(soup)
                        
                        if page_products:
                            products.extend(page_products)
                            print(f"Found {len(page_products)} products on page {page}")
                            break
                            
                except Exception as e:
                    print(f"Error with URL {url}: {e}")
                    continue
            
            # Random delay between pages
            time.sleep(random.uniform(5, 10))
        
        return products
    
    def extract_products(self, soup):
        """Extract sponsored products with multiple selectors"""
        products = []
        
        # Multiple selectors for different Amazon layouts
        selectors = [
            '[data-component-type="s-search-result"]',
            '[data-cel-widget*="search_result"]',
            '.s-result-item[data-component-type="s-search-result"]'
        ]
        
        for selector in selectors:
            containers = soup.select(selector)
            if containers:
                break
        
        print(f"Found {len(containers)} product containers")
        
        for container in containers:
            try:
                # Check for sponsored indicators
                sponsored_indicators = [
                    'Sponsored',
                    'sponsoredLabel',
                    '[data-sponsor]',
                    '.a-color-secondary:contains("Sponsored")'
                ]
                
                is_sponsored = any(
                    container.find(string=lambda text: text and indicator.lower() in text.lower()) 
                    if isinstance(indicator, str) else container.select(indicator)
                    for indicator in sponsored_indicators
                )
                
                if not is_sponsored:
                    continue
                
                product_data = self.extract_product_data(container)
                if product_data:
                    products.append(product_data)
                    
            except Exception as e:
                continue
        
        return products
    
    def extract_product_data(self, container):
        """Enhanced product data extraction"""
        try:
            # Title - multiple selectors
            title_selectors = [
                'h2 a span',
                '[data-cy="title-recipe-label"]',
                '.a-size-base-plus',
                '.a-size-medium'
            ]
            title = self.get_text_from_selectors(container, title_selectors)
            
            # Brand extraction
            brand = self.extract_brand_name(title, container)
            
            # Rating
            rating_selectors = [
                '[aria-label*="out of 5 stars"]',
                '.a-icon-alt'
            ]
            rating = self.extract_rating(container, rating_selectors)
            
            # Reviews count
            review_selectors = [
                'a[href*="#customerReviews"] span',
                '.a-size-base'
            ]
            reviews = self.extract_reviews(container, review_selectors)
            
            # Price
            price_selectors = [
                '.a-price-whole',
                '.a-offscreen',
                '[data-a-color="price"]'
            ]
            price = self.extract_price(container, price_selectors)
            
            # Image URL
            img = container.select_one('img')
            image_url = img.get('src', '') if img else ''
            
            # Product URL
            link = container.select_one('h2 a, [data-cy="title-recipe-label"] a')
            product_url = f"https://www.amazon.in{link.get('href', '')}" if link else ''
            
            return {
                'title': title,
                'brand': brand,
                'rating': rating,
                'reviews': reviews,
                'selling_price': price,
                'image_url': image_url,
                'product_url': product_url
            }
            
        except Exception as e:
            return None
    
    def get_text_from_selectors(self, container, selectors):
        """Try multiple selectors and return first match"""
        for selector in selectors:
            element = container.select_one(selector)
            if element:
                return element.get_text(strip=True)
        return "N/A"
    
    def extract_brand_name(self, title, container):
        """Enhanced brand extraction"""
        common_toy_brands = [
            'Dimpy', 'GUND', 'Ty', 'Melissa & Doug', 'VTech', 
            'Fisher-Price', 'Hasbro', 'Mattel', 'Disney'
        ]
        
        title_lower = title.lower()
        for brand in common_toy_brands:
            if brand.lower() in title_lower:
                return brand
        
        # Extract first word as potential brand
        words = title.split()
        return words[0] if words else "Unknown"
    
    def extract_rating(self, container, selectors):
        """Extract rating with multiple patterns"""
        for selector in selectors:
            element = container.select_one(selector)
            if element:
                aria_label = element.get('aria-label', '')
                if aria_label:
                    import re
                    match = re.search(r'(\d+\.?\d*)', aria_label)
                    if match:
                        return float(match.group(1))
        return 0.0
    
    def extract_reviews(self, container, selectors):
        """Extract review count"""
        for selector in selectors:
            element = container.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                import re
                match = re.search(r'([\d,]+)', text)
                if match:
                    return int(match.group(1).replace(',', ''))
        return 0
    
    def extract_price(self, container, selectors):
        """Extract price with currency handling"""
        for selector in selectors:
            element = container.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                import re
                match = re.search(r'([\d,]+)', text)
                if match:
                    return int(match.group(1).replace(',', ''))
        return 0

# Usage
if __name__ == "__main__":
    scraper = AmazonScraper()
    products = scraper.scrape_sponsored_products("soft toys", max_pages=2)
    
    if products:
        df = pd.DataFrame(products)
        df.to_csv('amazon_soft_toys_enhanced.csv', index=False)
        print(f"\nSuccessfully scraped {len(products)} sponsored products!")
        print(df.head())
    else:
        print("No products found. Try Solution 2 or 3.")
