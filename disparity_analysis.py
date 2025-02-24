import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Read the CSV files
cbow_df = pd.read_csv('top_100_disparities_gensim_cbow.csv')
gensim_df = pd.read_csv('top_100_disparities_gensim_sg.csv')

def analyze_disparities(df1, df2, name1="gensim_CBOW", name2="SG"):
    """Analyze and compare disparities between two methods"""
    
    # Basic statistics
    stats = {
        f"{name1}_mean": df1['Disparity'].mean(),
        f"{name1}_std": df1['Disparity'].std(),
        f"{name1}_max": df1['Disparity'].max(),
        f"{name1}_min": df1['Disparity'].min(),
        f"{name2}_mean": df2['Disparity'].mean(),
        f"{name2}_std": df2['Disparity'].std(),
        f"{name2}_max": df2['Disparity'].max(),
        f"{name2}_min": df2['Disparity'].min()
    }
    
    # Create comparative visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Disparity Distributions using histogram
    plt.subplot(2, 1, 1)
    plt.hist(df1['Disparity'], bins=20, alpha=0.5, label=name1, density=True)
    plt.hist(df2['Disparity'], bins=20, alpha=0.5, label=name2, density=True)
    
    # Add smoothed line using gaussian_kde
    for data, label in [(df1['Disparity'], name1), (df2['Disparity'], name2)]:
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        plt.plot(x_range, kde(x_range), label=f'{label} KDE')
    
    plt.title('Distribution of Disparities')
    plt.xlabel('Disparity')
    plt.ylabel('Density')
    plt.legend()
    
    # Plot 2: Disparity Rankings
    plt.subplot(2, 1, 2)
    plt.plot(range(len(df1)), df1['Disparity'], label=name1, marker='o', markersize=4)
    plt.plot(range(len(df2)), df2['Disparity'], label=name2, marker='o', markersize=4)
    plt.title('Disparity Rankings (Top 100 pairs)')
    plt.xlabel('Rank')
    plt.ylabel('Disparity')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('disparity_comparison_gensim_cbow_sg.png')
    plt.close()
    
    # Compare similarities
    common_pairs = set(zip(df1['Word1'], df1['Word2'])) & set(zip(df2['Word1'], df2['Word2']))
    
    print("\nDisparity Statistics:")
    print(f"\n{name1}:")
    print(f"Mean Disparity: {stats[f'{name1}_mean']:.4f}")
    print(f"Std Deviation: {stats[f'{name1}_std']:.4f}")
    print(f"Max Disparity: {stats[f'{name1}_max']:.4f}")
    print(f"Min Disparity: {stats[f'{name1}_min']:.4f}")
    
    print(f"\n{name2}:")
    print(f"Mean Disparity: {stats[f'{name2}_mean']:.4f}")
    print(f"Std Deviation: {stats[f'{name2}_std']:.4f}")
    print(f"Max Disparity: {stats[f'{name2}_max']:.4f}")
    print(f"Min Disparity: {stats[f'{name2}_min']:.4f}")
    
    print(f"\nNumber of common word pairs: {len(common_pairs)}")
    
    if common_pairs:
        print("\nSample of common pairs and their disparities:")
        for (word1, word2) in list(common_pairs)[:5]:
            cbow_disp = df1[(df1['Word1'] == word1) & (df1['Word2'] == word2)]['Disparity'].values[0]
            gensim_disp = df2[(df2['Word1'] == word1) & (df2['Word2'] == word2)]['Disparity'].values[0]
            print(f"\nPair: '{word1}' and '{word2}'")
            print(f"{name1} Disparity: {cbow_disp:.4f}")
            print(f"{name2} Disparity: {gensim_disp:.4f}")
            print(f"Difference: {abs(cbow_disp - gensim_disp):.4f}")
    
    # Additional Analysis: Print top 5 pairs from each method
    print(f"\nTop 5 most disparate pairs from {name1}:")
    for _, row in df1.head().iterrows():
        print(f"'{row['Word1']}' and '{row['Word2']}': {row['Disparity']:.4f}")
        
    print(f"\nTop 5 most disparate pairs from {name2}:")
    for _, row in df2.head().iterrows():
        print(f"'{row['Word1']}' and '{row['Word2']}': {row['Disparity']:.4f}")
    
    # Create correlation plot for common pairs
    if common_pairs:
        cbow_vals = []
        gensim_vals = []
        labels = []
        for word1, word2 in common_pairs:
            cbow_disp = df1[(df1['Word1'] == word1) & (df1['Word2'] == word2)]['Disparity'].values[0]
            gensim_disp = df2[(df2['Word1'] == word1) & (df2['Word2'] == word2)]['Disparity'].values[0]
            cbow_vals.append(cbow_disp)
            gensim_vals.append(gensim_disp)
            labels.append(f"{word1}-{word2}")
        
        plt.figure(figsize=(8, 8))
        plt.scatter(cbow_vals, gensim_vals, alpha=0.5)
        
        # Add diagonal line
        max_val = max(max(cbow_vals), max(gensim_vals))
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        
        plt.xlabel(f'{name1} Disparities')
        plt.ylabel(f'{name2} Disparities')
        plt.title('Correlation of Disparities for Common Pairs')
        plt.legend()
        
        # Calculate and display correlation coefficient
        correlation = np.corrcoef(cbow_vals, gensim_vals)[0,1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.savefig('disparity_correlation.png')
        plt.close()

# Run the analysis
analyze_disparities(cbow_df, gensim_df)