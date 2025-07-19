import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

class AmazonSoftToysAnalyzer:
    def __init__(self, csv_file='amazon_soft_toys_cleaned.csv'):
        """Initialize analyzer with cleaned data"""
        self.df = pd.read_csv(csv_file)
        self.setup_plotting_style()
        print(f"Loaded {len(self.df)} products for analysis")
        
    def setup_plotting_style(self):
        """Configure matplotlib and seaborn for better visuals"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def run_complete_analysis(self):
        """Execute all three required analyses"""
        print("=== STARTING COMPREHENSIVE ANALYSIS ===\n")
        
        # Analysis 1: Brand Performance Analysis
        self.brand_performance_analysis()
        
        # Analysis 2: Price vs Rating Analysis  
        self.price_rating_analysis()
        
        # Analysis 3: Review & Rating Distribution
        self.review_rating_distribution()
        
        # Bonus: Market Insights
        self.market_insights()
        
        print("\n=== ANALYSIS COMPLETE ===")
        
    def brand_performance_analysis(self):
        """Analysis 1: Brand Performance Analysis"""
        print("*** ANALYSIS 1: BRAND PERFORMANCE ***")
        print("=" * 50)
        
        # Brand frequency analysis
        brand_counts = self.df['brand'].value_counts()
        print(f"\n[CHART] Brand Frequency:")
        print(brand_counts.head())
        
        # Average rating by brand
        brand_ratings = self.df.groupby('brand').agg({
            'rating': 'mean',
            'reviews': 'sum',
            'selling_price': 'mean',
            'value_score': 'mean'
        }).round(2)
        
        brand_ratings = brand_ratings.sort_values('rating', ascending=False)
        print(f"\n[STAR] Brand Performance Metrics:")
        print(brand_ratings)
        
        # Visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Brand frequency bar chart
        top_brands = brand_counts.head(5)
        bars1 = ax1.bar(top_brands.index, top_brands.values, color='skyblue', edgecolor='navy')
        ax1.set_title('Top 5 Brands by Product Count', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Brand')
        ax1.set_ylabel('Number of Products')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 2. Brand market share pie chart
        ax2.pie(top_brands.values, labels=top_brands.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Market Share by Brand', fontsize=14, fontweight='bold')
        
        # 3. Average rating by brand
        top_rated = brand_ratings.head(5)['rating']
        bars3 = ax3.bar(top_rated.index, top_rated.values, color='lightgreen', edgecolor='darkgreen')
        ax3.set_title('Top 5 Brands by Average Rating', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Brand')
        ax3.set_ylabel('Average Rating')
        ax3.set_ylim(0, 5)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 4. Brand value score comparison
        top_value = brand_ratings.head(5)['value_score']
        bars4 = ax4.bar(top_value.index, top_value.values, color='orange', edgecolor='red')
        ax4.set_title('Top 5 Brands by Value Score', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Brand')
        ax4.set_ylabel('Value Score (Rating/Price)')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('brand_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Actionable insights
        print(f"\n[BULB] BRAND PERFORMANCE INSIGHTS:")
        dominant_brand = brand_counts.index[0]
        highest_rated = brand_ratings.index[0]
        best_value = brand_ratings.sort_values('value_score', ascending=False).index[0]
        
        print(f"• **Dominant Brand**: {dominant_brand} leads with {brand_counts[dominant_brand]} products")
        print(f"• **Quality Leader**: {highest_rated} has highest rating ({brand_ratings.loc[highest_rated, 'rating']:.1f}/5)")
        print(f"• **Best Value**: {best_value} offers best value score ({brand_ratings.loc[best_value, 'value_score']:.1f})")
        
    def price_rating_analysis(self):
        """Analysis 2: Price vs Rating Analysis"""
        print(f"\n*** ANALYSIS 2: PRICE VS RATING ***")
        print("=" * 50)
        
        # Price vs rating correlation
        correlation = self.df['selling_price'].corr(self.df['rating'])
        print(f"[CHART] Price-Rating Correlation: {correlation:.3f}")
        
        # Price by rating category
        price_by_rating = self.df.groupby('rating_category')['selling_price'].agg(['mean', 'median', 'count'])
        print(f"\n[MONEY] Average Price by Rating Category:")
        print(price_by_rating)
        
        # Value products (high rating, low price)
        median_price = self.df['selling_price'].median()
        value_products = self.df[
            (self.df['rating'] >= 4.0) & 
            (self.df['selling_price'] <= median_price)
        ].sort_values('value_score', ascending=False)
        
        print(f"\n[TROPHY] HIGH-VALUE PRODUCTS (Rating >=4.0, Price <=Rs.{median_price}):")
        print(value_products[['title', 'brand', 'rating', 'selling_price', 'value_score']].head())
        
        # Visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Scatter plot: Price vs Rating
        scatter = ax1.scatter(self.df['selling_price'], self.df['rating'], 
                             c=self.df['reviews'], s=60, alpha=0.7, cmap='viridis')
        ax1.set_xlabel('Selling Price (Rs)')
        ax1.set_ylabel('Rating')
        ax1.set_title('Price vs Rating (Color = Review Count)', fontsize=14, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(self.df['selling_price'], self.df['rating'], 1)
        p = np.poly1d(z)
        ax1.plot(self.df['selling_price'].sort_values(), p(self.df['selling_price'].sort_values()), 
                "r--", alpha=0.8, label=f'Trend (r={correlation:.3f})')
        ax1.legend()
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax1, label='Review Count')
        
        # 2. Average price by rating range
        rating_ranges = ['Poor', 'Average', 'Good', 'Excellent']
        price_means = []
        for cat in rating_ranges:
            cat_data = self.df[self.df['rating_category'] == cat]
            if len(cat_data) > 0:
                price_means.append(cat_data['selling_price'].mean())
            else:
                price_means.append(0)
        
        bars2 = ax2.bar(rating_ranges, price_means, color=['red', 'orange', 'lightblue', 'green'])
        ax2.set_title('Average Price by Rating Category', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Rating Category')
        ax2.set_ylabel('Average Price (Rs)')
        
        # Add value labels
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                        f'Rs{int(height)}', ha='center', va='bottom')
        
        # 3. Value score distribution
        ax3.hist(self.df['value_score'], bins=10, color='purple', alpha=0.7, edgecolor='black')
        ax3.set_title('Value Score Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Value Score (Rating/Price)')
        ax3.set_ylabel('Number of Products')
        ax3.axvline(self.df['value_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["value_score"].mean():.1f}')
        ax3.legend()
        
        # 4. Price categories breakdown
        price_category_counts = self.df['price_category'].value_counts()
        ax4.pie(price_category_counts.values, labels=price_category_counts.index, 
                autopct='%1.1f%%', startangle=90)
        ax4.set_title('Price Category Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('price_rating_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Insights
        print(f"\n[BULB] PRICE-RATING INSIGHTS:")
        if correlation > 0.3:
            print(f"• **Strong Positive Correlation**: Higher prices generally mean better ratings")
        elif correlation < -0.3:
            print(f"• **Negative Correlation**: Higher prices don't guarantee better ratings")
        else:
            print(f"• **Weak Correlation**: Price and rating are not strongly related")
        
        best_value_product = value_products.iloc[0] if len(value_products) > 0 else None
        if best_value_product is not None:
            print(f"• **Best Value Product**: {best_value_product['title'][:50]}...")
            print(f"  - Rating: {best_value_product['rating']}/5, Price: Rs{best_value_product['selling_price']}")
        
    def review_rating_distribution(self):
        """Analysis 3: Review & Rating Distribution"""
        print(f"\n*** ANALYSIS 3: REVIEW & RATING DISTRIBUTION ***")
        print("=" * 50)
        
        # Top products by reviews
        top_reviewed = self.df.nlargest(5, 'reviews')[['title', 'brand', 'reviews', 'rating']]
        print(f"[NOTE] TOP 5 MOST REVIEWED PRODUCTS:")
        for idx, product in top_reviewed.iterrows():
            print(f"• {product['brand']} - {product['reviews']} reviews (Rating: {product['rating']}/5)")
        
        # Top products by rating
        top_rated = self.df.nlargest(5, 'rating')[['title', 'brand', 'rating', 'reviews']]
        print(f"\n[STAR] TOP 5 HIGHEST RATED PRODUCTS:")
        for idx, product in top_rated.iterrows():
            print(f"• {product['brand']} - {product['rating']}/5 rating ({product['reviews']} reviews)")
        
        # Review volume analysis
        review_stats = self.df['reviews'].describe()
        print(f"\n[CHART] REVIEW STATISTICS:")
        print(f"• Average reviews per product: {review_stats['mean']:.0f}")
        print(f"• Median reviews: {review_stats['50%']:.0f}")
        print(f"• Most reviewed product: {review_stats['max']:.0f} reviews")
        
        # Visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Top 5 by reviews
        top_5_reviews = self.df.nlargest(5, 'reviews')
        bars1 = ax1.barh(range(len(top_5_reviews)), top_5_reviews['reviews'], color='lightcoral')
        ax1.set_yticks(range(len(top_5_reviews)))
        ax1.set_yticklabels([f"{row['brand']}" for _, row in top_5_reviews.iterrows()])
        ax1.set_xlabel('Number of Reviews')
        ax1.set_title('Top 5 Products by Review Count', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + 50, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}', ha='left', va='center')
        
        # 2. Top 5 by rating
        top_5_rating = self.df.nlargest(5, 'rating')
        bars2 = ax2.barh(range(len(top_5_rating)), top_5_rating['rating'], color='lightgreen')
        ax2.set_yticks(range(len(top_5_rating)))
        ax2.set_yticklabels([f"{row['brand']}" for _, row in top_5_rating.iterrows()])
        ax2.set_xlabel('Rating')
        ax2.set_xlim(0, 5)
        ax2.set_title('Top 5 Products by Rating', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}', ha='left', va='center')
        
        # 3. Rating distribution
        ax3.hist(self.df['rating'], bins=np.arange(0, 5.5, 0.5), color='skyblue', 
                alpha=0.7, edgecolor='navy')
        ax3.set_xlabel('Rating')
        ax3.set_ylabel('Number of Products')
        ax3.set_title('Rating Distribution', fontsize=14, fontweight='bold')
        ax3.axvline(self.df['rating'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["rating"].mean():.1f}')
        ax3.legend()
        
        # 4. Review volume categories
        review_volume_counts = self.df['review_volume'].value_counts()
        colors = ['lightblue', 'orange', 'lightgreen', 'red']
        ax4.pie(review_volume_counts.values, labels=review_volume_counts.index, 
                autopct='%1.1f%%', startangle=90, colors=colors)
        ax4.set_title('Review Volume Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('review_rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Insights
        print(f"\n[BULB] REVIEW & RATING INSIGHTS:")
        most_reviewed = top_reviewed.iloc[0]
        highest_rated = top_rated.iloc[0]
        
        print(f"• **Most Popular**: {most_reviewed['brand']} dominates with {most_reviewed['reviews']} reviews")
        print(f"• **Quality Champion**: {highest_rated['brand']} leads with {highest_rated['rating']}/5 rating")
        print(f"• **Market Trust**: Average rating is {self.df['rating'].mean():.1f}/5 across all products")
        
        # Find hidden gems (high rating, low reviews)
        hidden_gems = self.df[
            (self.df['rating'] >= 4.0) & 
            (self.df['reviews'] < self.df['reviews'].median())
        ].sort_values('rating', ascending=False)
        
        if len(hidden_gems) > 0:
            gem = hidden_gems.iloc[0]
            print(f"• **Hidden Gem**: {gem['brand']} has {gem['rating']}/5 rating but only {gem['reviews']} reviews")
    
    def market_insights(self):
        """Bonus: Market Insights and Recommendations"""
        print(f"\n*** BONUS: MARKET INSIGHTS & RECOMMENDATIONS ***")
        print("=" * 50)
        
        # Market summary
        total_products = len(self.df)
        avg_price = self.df['selling_price'].mean()
        avg_rating = self.df['rating'].mean()
        total_reviews = self.df['reviews'].sum()
        
        print(f"[CHART] MARKET OVERVIEW:")
        print(f"• Total sponsored products analyzed: {total_products}")
        print(f"• Average price: Rs{avg_price:.0f}")
        print(f"• Average rating: {avg_rating:.1f}/5")
        print(f"• Total customer reviews: {total_reviews:,}")
        
        # Price segments
        print(f"\n[MONEY] PRICE SEGMENT ANALYSIS:")
        price_segments = self.df['price_category'].value_counts()
        for segment, count in price_segments.items():
            pct = (count/total_products)*100
            print(f"• {segment}: {count} products ({pct:.1f}%)")
        
        # Competitive gaps
        print(f"\n[TARGET] STRATEGIC RECOMMENDATIONS:")
        
        # Find underrepresented segments
        low_competition_brands = self.df['brand'].value_counts().tail(3)
        print(f"• **Opportunity Brands**: {', '.join(low_competition_brands.index)} have low market presence")
        
        # Price-quality gaps
        premium_products = self.df[self.df['price_category'] == 'Premium']
        if len(premium_products) > 0:
            premium_avg_rating = premium_products['rating'].mean()
            print(f"• **Premium Segment**: Average rating is {premium_avg_rating:.1f}/5 - quality justifies price")
        
        # Create summary table
        print(f"\n[CLIPBOARD] COMPLETE PRODUCT SUMMARY:")
        summary_table = self.df.groupby('brand').agg({
            'selling_price': ['mean', 'min', 'max'],
            'rating': 'mean',
            'reviews': 'sum',
            'value_score': 'mean'
        }).round(2)
        
        print(summary_table)
        
        # Save summary
        summary_table.to_csv('brand_summary_analysis.csv')
        print(f"\n[CHECK] Analysis complete! Summary saved to 'brand_summary_analysis.csv'")

if __name__ == "__main__":
    # Run complete analysis
    analyzer = AmazonSoftToysAnalyzer()
    analyzer.run_complete_analysis()
