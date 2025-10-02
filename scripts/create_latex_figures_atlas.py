import os
from pathlib import Path
from collections import defaultdict

def generate_atlas_latex():
    """Generate LaTeX code for all images in the atlas subfolder"""
    
    atlas_path = Path("output/atlas")
    
    if not atlas_path.exists():
        print(f"Atlas directory not found: {atlas_path}")
        return
    
    latex_code = []
    latex_code.append("% Auto-generated LaTeX code for atlas images")
    latex_code.append("% Add \\usepackage{placeins} to your preamble")
    latex_code.append("% Add \\usepackage[section]{placeins} for automatic barriers")
    latex_code.append("")
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.eps'}
    
    # Collect all images grouped by country
    country_images = defaultdict(list)
    
    # Walk through all subdirectories in atlas
    for subdir in sorted(atlas_path.iterdir()):
        if subdir.is_dir():
            # Find all images in this subdirectory
            for file_path in subdir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    # Extract country from filename
                    filename = file_path.stem
                    country = filename.split("_")[1] if "_" in filename else "unknown"
                    country_images[country].append((file_path, subdir))
    
    # Generate LaTeX code grouped by country
    figure_count = 0
    for country in sorted(country_images.keys()):
        # Add section header for each country
        country_display = country.replace('-', ' ').title()
        # latex_code.append(f"\\subsection{{{country_display}}}")
        latex_code.append("")
        
        for img_path, subdir in sorted(country_images[country], key=lambda x: x[1].name):
            # Use the path as is (already relative to project root)
            rel_path = img_path
            
            # Generate caption from filename
            filename = img_path.stem
            
            # Determine color note based on subdirectory type
            if "random" in subdir.name.lower():
                color_note = f"Random removal of edges in {country_display}. Retained edges and stations are displayed in dark gray, removed ones in orange."
            elif "knn" in subdir.name.lower():
                color_note = f"KNN based removal of edges in {country_display}. Retained edges and stations are displayed in blue, removed ones in orange."
            else:
                color_note = f"x is displayed in color y, z is displayed in color a."

            # Add figure LaTeX code with better float placement
            latex_code.append("\\begin{figure}[!htbp]")
            latex_code.append("    \\centering")
            latex_code.append(f"    \\includegraphics[width=0.8\\textwidth]{{{rel_path}}}")
            latex_code.append(f"    \\caption{{{color_note}}}")
            latex_code.append(f"    \\label{{fig:{filename}_{subdir.name}}}")
            latex_code.append("\\end{figure}")
            latex_code.append("")
            
            figure_count += 1
            
            # Add float barrier every 4-6 figures to prevent buildup
            if figure_count % 5 == 0:
                latex_code.append("\\FloatBarrier")
                latex_code.append("")
    
    # Add final float barrier
    latex_code.append("\\FloatBarrier")
    latex_code.append("")
    
    # Write to file
    output_file = "output/atlas/atlas_figures.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_code))
    
    print(f"LaTeX code generated and saved to: {output_file}")
    print(f"Found {figure_count} images")
    print("\nTo use this file:")
    print("1. Add \\usepackage{placeins} to your LaTeX preamble")
    print("2. Include the generated file with \\input{output/atlas/atlas_figures.tex}")
    print("3. Consider using \\clearpage before including if you have many figures")

if __name__ == "__main__":
    generate_atlas_latex()