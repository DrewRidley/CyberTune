#!/usr/bin/env python3
"""
This script enhances the visualization by creating a combined HTML page with images and table.
"""

import argparse
import json
import base64
from pathlib import Path
import os

def read_image_as_base64(image_path):
    """Read image file and encode as base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_enhanced_visualization(vis_dir, output_file='enhanced_visualization.html'):
    """Generate an enhanced HTML visualization that includes all charts and the table"""
    
    # Define paths
    vis_dir = Path(vis_dir)
    category_dist_path = vis_dir / 'category_distribution.png'
    similarity_by_cat_path = vis_dir / 'similarity_by_category.png'
    similarity_scores_path = vis_dir / 'similarity_scores.png'
    text_index_path = vis_dir / 'text_index.html'
    
    # Check if all required files exist
    required_files = [category_dist_path, similarity_by_cat_path, similarity_scores_path, text_index_path]
    missing_files = [f for f in required_files if not f.exists()]
    
    if missing_files:
        print(f"Error: The following required files are missing: {', '.join(str(f) for f in missing_files)}")
        return False
    
    # Read and extract table content from text_index.html
    with open(text_index_path, 'r') as f:
        text_index_content = f.read()
    
    # Extract just the table part (between <table> and </table>)
    table_start = text_index_content.find('<table')
    table_end = text_index_content.find('</table>', table_start) + 8
    if table_start == -1 or table_end == 7:
        print("Error: Could not extract table from text_index.html")
        return False
    
    table_content = text_index_content[table_start:table_end]
    
    # Encode images as base64
    try:
        category_dist_b64 = read_image_as_base64(category_dist_path)
        similarity_by_cat_b64 = read_image_as_base64(similarity_by_cat_path)
        similarity_scores_b64 = read_image_as_base64(similarity_scores_path)
    except Exception as e:
        print(f"Error encoding images: {str(e)}")
        return False
    
    # Create enhanced HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced TTP Categorization Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            color: #333;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .chart {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        th {{
            padding-top: 12px;
            padding-bottom: 12px;
            text-align: left;
            background-color: #4CAF50;
            color: white;
        }}
        .high {{
            background-color: #a8e6cf;
        }}
        .medium {{
            background-color: #dcedc1;
        }}
        .low {{
            background-color: #ffd3b6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Cybersecurity TTP Categorization Results</h1>
        
        <div class="chart-container">
            <h2>Distribution of TTP Categories</h2>
            <img class="chart" src="data:image/png;base64,{category_dist_b64}" alt="Category Distribution">
        </div>
        
        <div class="chart-container">
            <h2>Similarity Scores by TTP Category</h2>
            <img class="chart" src="data:image/png;base64,{similarity_by_cat_b64}" alt="Similarity by Category">
        </div>
        
        <div class="chart-container">
            <h2>Similarity Scores for Each Text</h2>
            <img class="chart" src="data:image/png;base64,{similarity_scores_b64}" alt="Similarity Scores">
        </div>
        
        <h2>TTP Categorization Details</h2>
        {table_content}
    </div>
</body>
</html>
"""
    
    # Write the enhanced HTML file
    output_path = vis_dir / output_file
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Enhanced visualization created at: {output_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create enhanced visualization HTML with embedded images")
    parser.add_argument("--vis_dir", type=str, default="visualizations",
                        help="Directory containing visualization files (default: visualizations)")
    parser.add_argument("--output", type=str, default="enhanced_visualization.html",
                        help="Output HTML file name (default: enhanced_visualization.html)")
    
    args = parser.parse_args()
    generate_enhanced_visualization(args.vis_dir, args.output) 