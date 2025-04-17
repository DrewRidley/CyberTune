#!/usr/bin/env python3
"""
This script visualizes the TTP categorization results from the fine-tuned model.
"""

import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from pathlib import Path

def load_category_mapping(result_file):
    """Load category mapping results from JSON file"""
    with open(result_file, 'r') as f:
        data = json.load(f)
    return data["results"]

def plot_category_distribution(results, output_dir):
    """Plot the distribution of TTP categories"""
    # Count categories
    categories = [result["category"] for result in results]
    category_counts = Counter(categories)
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        'Category': list(category_counts.keys()),
        'Count': list(category_counts.values())
    }).sort_values('Count', ascending=False)
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='Count', y='Category', palette='viridis')
    plt.title('Distribution of TTP Categories')
    plt.xlabel('Count')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'category_distribution.png')
    plt.close()
    
    return df

def plot_similarity_scores(results, output_dir):
    """Plot the similarity scores for each category"""
    # Create DataFrame with categories and similarity scores
    df = pd.DataFrame({
        'Category': [result["category"] for result in results],
        'Similarity': [result["similarity_score"] for result in results],
        'Text': [result["text"] for result in results]
    })
    
    # Plot boxplot of similarity scores by category
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='Similarity', y='Category', palette='viridis')
    plt.title('Similarity Scores by TTP Category')
    plt.xlabel('Similarity Score')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'similarity_by_category.png')
    plt.close()
    
    # Plot scatter plot of similarity scores
    plt.figure(figsize=(12, 8))
    categories = sorted(df['Category'].unique())
    colors = sns.color_palette('viridis', len(categories))
    category_color = {cat: color for cat, color in zip(categories, colors)}
    
    for i, row in df.iterrows():
        plt.scatter(i, row['Similarity'], color=category_color[row['Category']], s=100)
        plt.text(i, row['Similarity'] - 0.02, str(i), ha='center', va='top', fontsize=9)
    
    plt.ylim(0.6, 1.0)
    plt.title('Similarity Scores for Each Text')
    plt.xlabel('Text Index')
    plt.ylabel('Similarity Score')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color, markersize=10, label=cat)
                       for cat, color in category_color.items()]
    plt.legend(handles=legend_elements, title='Categories')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'similarity_scores.png')
    plt.close()
    
    return df

def create_text_index(results, output_dir):
    """Create a text index with categories and scores"""
    # Build HTML table
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TTP Categorization Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            th { padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: #4CAF50; color: white; }
            .high { background-color: #a8e6cf; }
            .medium { background-color: #dcedc1; }
            .low { background-color: #ffd3b6; }
        </style>
    </head>
    <body>
        <h1>TTP Categorization Results</h1>
        <table>
            <tr>
                <th>#</th>
                <th>Text</th>
                <th>Mapped Category</th>
                <th>Similarity Score</th>
            </tr>
    """
    
    # Add rows
    for i, result in enumerate(results):
        # Determine cell color based on similarity score
        if result["similarity_score"] >= 0.85:
            score_class = "high"
        elif result["similarity_score"] >= 0.75:
            score_class = "medium"
        else:
            score_class = "low"
            
        html += f"""
            <tr>
                <td>{i+1}</td>
                <td>{result["text"]}</td>
                <td>{result["category"]}</td>
                <td class="{score_class}">{result["similarity_score"]:.4f}</td>
            </tr>
        """
    
    html += """
        </table>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(Path(output_dir) / 'text_index.html', 'w') as f:
        f.write(html)

def main(args):
    """Main function to visualize TTP categorization results"""
    # Load results
    results = load_category_mapping(args.result_file)
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    # Plot category distribution
    category_df = plot_category_distribution(results, args.output_dir)
    print("Category Distribution:")
    print(category_df)
    
    # Plot similarity scores
    similarity_df = plot_similarity_scores(results, args.output_dir)
    
    # Create text index
    create_text_index(results, args.output_dir)
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize TTP categorization results")
    
    parser.add_argument("--result_file", type=str, default="results/category_mapping.json",
                        help="Path to category mapping JSON file (default: results/category_mapping.json)")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Directory to save visualizations (default: visualizations)")
    
    args = parser.parse_args()
    main(args) 