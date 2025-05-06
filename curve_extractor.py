import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import io
import base64
from PIL import Image
import anthropic
import json
import re

def remove_text(img):
    """
    Simple function to remove text by thresholding.
    """
    # Convert to grayscale if necessary
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Binary thresholding to highlight text
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Create a mask (text is white, background is black)
    mask = thresh.copy()
    
    # Use mask to remove text from original image
    result = img.copy()
    if len(img.shape) == 3:
        # For color images, apply mask to each channel
        for i in range(3):
            result[:, :, i] = cv2.bitwise_and(result[:, :, i], result[:, :, i], mask=mask)
    else:
        result = cv2.bitwise_and(result, result, mask=mask)
        
    return result

def extract_survival_curves(img, colors=None, x_range=(0, 50), y_range=(0, 100), show_plots=False, api_key=None):
    """
    Extract survival curve data from a plot image.
    
    Args:
        img: OpenCV image or PIL Image object
        colors: List of colors to extract (default: 15 common colors used in KM curves).
        x_range: Tuple with (min, max) values for x-axis (default: (0, 50) years).
        y_range: Tuple with (min, max) values for y-axis (default: (0, 100) percent).
        show_plots: Whether to display matplotlib plots during processing (default: False)
        
    Returns:
        DataFrame with the digitized data.
    """
    # Convert PIL Image to OpenCV format if needed
    if isinstance(img, Image.Image):
        img_array = np.array(img)
        # Convert RGB to BGR (OpenCV format)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Initialize metadata
    metadata = {
        "x_range": [0, 50],
        "y_range": [0, 100],
        "curves": [],
        "colors": [],
        "color_to_name": {}
    }
    
    # Try to extract metadata using Claude if API key is provided
    if api_key:
        try:
            metadata = extract_curve_metadata(img, api_key)
            if metadata:
                # Use extracted x_range and y_range if available
                if 'x_range' in metadata and len(metadata['x_range']) == 2:
                    x_range = tuple(metadata['x_range'])
                    print(f"Using extracted x_range: {x_range}")
                    
                if 'y_range' in metadata and len(metadata['y_range']) == 2:
                    y_range = tuple(metadata['y_range'])
                    print(f"Using extracted y_range: {y_range}")
                    
                # Use extracted colors if available
                if 'colors' in metadata and metadata['colors']:
                    colors = metadata['colors']
                    print(f"Using extracted colors: {colors}")
        except Exception as e:
            print(f"Error extracting metadata: {str(e)}")
    
    # Define 15 common color schemes for KM curves with HSV thresholds.
    # These ranges can be further tuned to your plotting style.
    color_hsv_params = {
        'red':     { 'ranges': [ ([0, 70, 50], [10, 255, 255]), ([170, 70, 50], [180, 255, 255]) ] },
        'green':   { 'lower': [40, 50, 50],  'upper': [80, 255, 255] },
        'blue':    { 'lower': [100, 50, 50], 'upper': [140, 255, 255] },
        'purple':  { 'lower': [130, 50, 50], 'upper': [150, 255, 255] },
        'black':   { 'lower': [0, 0, 0],     'upper': [180, 255, 50] },
        'orange':  { 'lower': [10, 100, 100],'upper': [25, 255, 255] },
        'brown':   { 'lower': [10, 100, 20], 'upper': [20, 255, 150] },
        'pink':    { 'lower': [160, 40, 80], 'upper': [170, 255, 255] },
        'teal':    { 'lower': [80, 50, 50],  'upper': [100, 255, 255] },
        'gray':    { 'lower': [0, 0, 50],    'upper': [180, 50, 200] },
        'yellow':  { 'lower': [20, 100, 100],'upper': [30, 255, 255] },
        'magenta': { 'lower': [150, 50, 50], 'upper': [170, 255, 255] },
        'cyan':    { 'lower': [90, 50, 50],  'upper': [110, 255, 255] },
        'olive':   { 'lower': [25, 40, 40],  'upper': [35, 255, 200] },
        'gold':    { 'lower': [15, 100, 150],'upper': [25, 255, 255] }
    }
    
    if colors is None:
        # Default: extract all 15 common colors.
        colors = list(color_hsv_params.keys())
    
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if show_plots:
        plt.figure(figsize=(10, 6))
        plt.imshow(img_rgb)
        plt.title("Original Image")
        plt.show()
    
    # Detect the plot area via edge detection.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours and select the largest rectangular area.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    plot_rect = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area and w > img.shape[1] / 4 and h > img.shape[0] / 4:
            max_area = area
            plot_rect = (x, y, w, h)
    
    if plot_rect is None:
        plot_rect = (0, 0, img.shape[1], img.shape[0])
    
    x, y, w, h = plot_rect
    
    if show_plots:
        # Show the detected plot area.
        img_with_rect = img_rgb.copy()
        cv2.rectangle(img_with_rect, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.figure(figsize=(10, 6))
        plt.imshow(img_with_rect)
        plt.title("Detected Plot Area")
        plt.show()
    
    # Crop and display the plot area.
    plot_img = img[y:y + h, x:x + w]
    plot_rgb = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
    
    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.imshow(plot_rgb)
        plt.title("Extracted Plot Area")
        plt.show()
    
    # Create a mask to exclude the legend (right 20% of the plot).
    plot_height, plot_width = plot_img.shape[:2]
    legend_mask = np.ones_like(plot_img[:, :, 0], dtype=bool)
    legend_x_start = int(plot_width * 0.8)
    legend_mask[:, legend_x_start:] = False
    
    if show_plots:
        legend_vis = plot_rgb.copy()
        legend_vis[~legend_mask] = [255, 0, 0]  # Red overlay on the legend area.
        plt.figure(figsize=(8, 6))
        plt.imshow(legend_vis)
        plt.title("Legend Region Mask")
        plt.show()
    
    curves_data = {}
    extracted_points_img = plot_rgb.copy()
    
    # Get color-to-name mapping if available
    color_to_name = {}
    if 'color_to_name' in metadata and metadata['color_to_name']:
        color_to_name = metadata['color_to_name']
        print(f"Using color-to-name mapping: {color_to_name}")
    
    # Process each specified curve.
    for color_name in colors:
        print(f"Processing {color_name} curve...")
        if color_name in color_hsv_params:
            # Convert to HSV and apply the defined thresholds.
            plot_hsv = cv2.cvtColor(plot_img, cv2.COLOR_BGR2HSV)
            params = color_hsv_params[color_name]
            if 'ranges' in params:
                mask_total = None
                for lower_bound, upper_bound in params['ranges']:
                    lower_bound = np.array(lower_bound)
                    upper_bound = np.array(upper_bound)
                    mask = cv2.inRange(plot_hsv, lower_bound, upper_bound)
                    if mask_total is None:
                        mask_total = mask
                    else:
                        mask_total = cv2.bitwise_or(mask_total, mask)
                mask = mask_total
            else:
                lower = np.array(params['lower'])
                upper = np.array(params['upper'])
                mask = cv2.inRange(plot_hsv, lower, upper)
            print(f"Using HSV-based detection for {color_name}.")
        else:
            # Fallback on a BGR-based approach (if needed).
            print(f"Falling back on BGR detection for {color_name}.")
            target_color = np.array([0, 0, 0])  # Placeholder; adjust as needed.
            color_distance = np.sqrt(np.sum((plot_img.astype(np.float32) - target_color.astype(np.float32)) ** 2, axis=2))
            threshold = 100
            mask = (color_distance < threshold).astype(np.uint8) * 255

        # Apply the legend mask.
        mask = cv2.bitwise_and(mask, mask, mask=legend_mask.astype(np.uint8))
        
        if show_plots:
            # Show the mask for debugging.
            plt.figure(figsize=(8, 6))
            plt.imshow(mask, cmap='gray')
            plt.title(f"{color_name.capitalize()} Mask")
            plt.show()
        
        # Clean up the mask with morphological operations.
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        if np.sum(mask > 0) < 100:
            print(f"Not enough points detected for {color_name}.")
            continue
        
        # Extract nonzero coordinates from the mask.
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) == 0:
            print(f"No points detected for {color_name}.")
            continue
        
        # For each unique x-coordinate, choose the lowest y-value.
        unique_x = np.unique(x_coords)
        min_y = np.zeros_like(unique_x, dtype=int)
        for i, x_val in enumerate(unique_x):
            y_at_x = y_coords[x_coords == x_val]
            if len(y_at_x) > 0:
                min_y[i] = np.min(y_at_x)
        
        # Mark the extracted points on the visualization image.
        for x_val, y_val in zip(unique_x, min_y):
            cv2.circle(extracted_points_img, (int(x_val), int(y_val)), 2, (255, 0, 255), -1)
        
        curves_data[color_name] = (unique_x, min_y)
    
    if show_plots:
        # Display the aggregated extracted points.
        plt.figure(figsize=(8, 6))
        plt.imshow(extracted_points_img)
        plt.title("Extracted Curve Points")
        plt.show()
    
    # Convert pixel coordinates to data values.
    processed_curves = {}
    for color_name, (x_pixels, y_pixels) in curves_data.items():
        if len(x_pixels) == 0:
            continue
        
        # Convert x_pixels to data values.
        x_data = x_range[0] + (x_pixels / plot_width) * (x_range[1] - x_range[0])
        
        # ---- Iterative adjustment for y_data conversion ----
        # We start with the initial y_max (user-defined maximum) and a y_scale that represents the pixel-to-data ratio.
        y_max = y_range[1]
        y_scale = plot_height / (y_range[1] - y_range[0])
        
        current_y_max = y_max
        found_better_ymax = False
        for delta in np.linspace(0.01, 0.1, 10):
            temp_y_max = y_max + delta
            temp_y_data = temp_y_max - (y_pixels / y_scale)
            sort_idx = np.argsort(x_data)
            sorted_temp_y = temp_y_data[sort_idx]
            if sorted_temp_y[0] >= 0.999:
                current_y_max = temp_y_max
                found_better_ymax = True
                print(f"Adjusted y_max to {current_y_max:.3f} for better normalization for {color_name}.")
                break
        
        # Calculate the final y_data using the adjusted y_max.
        y_data = current_y_max - (y_pixels / y_scale)
        
        # Sort the points by x.
        sort_indices = np.argsort(x_data)
        x_sorted = x_data[sort_indices]
        y_sorted = y_data[sort_indices]
        
        # Force the curve to begin at the expected starting point.
        if x_sorted[0] > x_range[0] + 0.05 * (x_range[1] - x_range[0]):
            x_sorted = np.insert(x_sorted, 0, x_range[0])
            y_sorted = np.insert(y_sorted, 0, y_range[1])
        
        # Interpolate to smooth the curve.
        x_smooth = np.linspace(x_sorted[0], x_sorted[-1], 1000)
        y_smooth = interp1d(x_sorted, y_sorted, bounds_error=False, fill_value='extrapolate')(x_smooth)
        
        # Ensure monotonicity (survival should not increase).
        for i in range(1, len(y_smooth)):
            if y_smooth[i] > y_smooth[i - 1]:
                y_smooth[i] = y_smooth[i - 1]
        
        processed_curves[color_name] = (x_smooth, y_smooth)
    
    if show_plots:
        # Plot the final extracted and smoothed curves.
        plt.figure(figsize=(10, 6))
        plot_color_map = {
            'red': 'red',
            'green': 'green',
            'blue': 'blue',
            'purple': 'purple',
            'black': 'black',
            'orange': 'orange',
            'brown': 'saddlebrown',
            'pink': 'pink',
            'teal': 'teal',
            'gray': 'gray',
            'yellow': 'yellow',
            'magenta': 'magenta',
            'cyan': 'cyan',
            'olive': 'olive',
            'gold': 'gold'
        }
        
        for color_name, (x_data, y_data) in processed_curves.items():
            if len(x_data) <= 1:
                continue
            label = color_name.capitalize()
            plot_col = plot_color_map.get(color_name, color_name)
            plt.plot(x_data, y_data, color=plot_col, linewidth=2, label=label)
        
        plt.xlabel("Years from onset")
        plt.ylabel("Survival (%)")
        plt.title("Digitized Survival Curves")
        plt.grid(True)
        plt.legend()
        plt.xlim(x_range[0], x_range[1])
        plt.ylim(y_range[0], y_range[1])
        plt.show()
    
    # Create the DataFrame
    data_dict = {'time_months': np.linspace(x_range[0], x_range[1], 1000) * 12}  # Convert years to months
    
    for color_name, (x_data, y_data) in processed_curves.items():
        if len(x_data) <= 1:
            continue
            
        # Use the actual curve name from metadata if available, otherwise use the color name
        if color_name in color_to_name:
            curve_label = color_to_name[color_name]
            print(f"Using curve name '{curve_label}' for color '{color_name}'")
        else:
            curve_label = color_name.capitalize()
            
        interp_func = interp1d(x_data, y_data, bounds_error=False, fill_value='extrapolate')
        y_interp = interp_func(data_dict['time_months'] / 12)  # Convert back to years for interpolation
        y_interp = np.clip(y_interp, y_range[0], y_range[1])
        # Don't store curve_name in data_dict as it's a string and np.full_like expects numeric values
        data_dict[f"survival_prob_{color_name}"] = y_interp
    
    # Restructure into a long-format DataFrame for compatibility with existing code
    df_list = []
    for color_name in processed_curves.keys():
        if len(processed_curves[color_name][0]) <= 1:
            continue
            
        # Use the actual curve name from metadata if available
        if color_name in color_to_name:
            curve_label = color_to_name[color_name]
        else:
            curve_label = color_name.capitalize()
            
        df_color = pd.DataFrame({
            'time_months': data_dict['time_months'],
            'curve_name': curve_label,  # Use the actual curve name
            'survival_prob': data_dict[f"survival_prob_{color_name}"]
        })
        df_list.append(df_color)
    
    if not df_list:
        return pd.DataFrame(columns=['time_months', 'curve_name', 'survival_prob'])
    
    df = pd.concat(df_list, ignore_index=True)
    return df

def create_plot_from_df(df):
    """
    Create a matplotlib plot from the extracted curve dataframe
    
    Args:
        df: DataFrame with time_months, curve_name, and survival_prob columns
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique curve names
    curves = df['curve_name'].unique()
    
    # Plot each curve
    for curve in curves:
        curve_data = df[df['curve_name'] == curve]
        ax.plot(curve_data['time_months']/12, curve_data['survival_prob'], 
                label=curve, linewidth=2)
    
    ax.set_xlabel('Years')
    ax.set_ylabel('Survival Probability (%)')
    ax.set_title('Extracted Survival Curves')
    ax.grid(True)
    ax.legend()
    
    return fig

def extract_curve_metadata(img, api_key):
    """
    Use Claude 3.7 to extract metadata from a survival curve image
    
    Args:
        img: PIL Image object of the survival curve
        api_key: Anthropic API key
        
    Returns:
        Dictionary with x_range, y_range, and curves information
    """
    # Convert image to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Create Claude client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Prepare the prompt
    prompt = """
    Analyze this survival curve image and extract the following information in JSON format:
    1. X-axis range (min and max values)
    2. Y-axis range (min and max values)
    3. List of curves with their colors and names (if visible in legend)
    
    Return ONLY a valid JSON object with this structure:
    {
        "x_range": [min, max],
        "y_range": [min, max],
        "curves": [
            {"name": "curve1", "color": "red"},
            {"name": "curve2", "color": "blue"}
        ]
    }
    
    If you can't determine exact values, make your best estimate based on the visible tick marks and labels.
    """
    
    try:
        # Call Claude API
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219", #claude-3-7-sonnet-20250219
            max_tokens=1024,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}}
                ]}
            ]
        )
        
        # Extract JSON from response
        text_response = response.content[0].text
        
        # Find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', text_response)
        if json_match:
            json_str = json_match.group(0)
            try:
                metadata = json.loads(json_str)
                
                # Set default values if missing
                if 'x_range' not in metadata or not isinstance(metadata['x_range'], list) or len(metadata['x_range']) != 2:
                    metadata['x_range'] = [0, 50]  # Default: 0-50 months/years
                    
                if 'y_range' not in metadata or not isinstance(metadata['y_range'], list) or len(metadata['y_range']) != 2:
                    metadata['y_range'] = [0, 100]  # Default: 0-100%
                    
                if 'curves' not in metadata or not isinstance(metadata['curves'], list):
                    metadata['curves'] = []  # Default: empty list
                    
                # Extract colors and names from curves
                colors = [curve.get('color', '').lower() for curve in metadata['curves'] if 'color' in curve]
                if colors:
                    metadata['colors'] = colors
                
                # Create a color-to-name mapping
                color_to_name = {}
                for curve in metadata['curves']:
                    if 'color' in curve and 'name' in curve:
                        color_to_name[curve['color'].lower()] = curve['name']
                
                metadata['color_to_name'] = color_to_name
                
                return metadata
            except json.JSONDecodeError:
                print("Could not parse JSON from Claude response")
        
    except Exception as e:
        print(f"Error calling Claude API: {str(e)}")
    
    # Return default values if extraction fails
    return {
        "x_range": [0, 50],
        "y_range": [0, 100],
        "curves": [],
        "colors": [],
        "color_to_name": {}
    }

def lookup_survival_at_time(df, curve_name, time_months):
    """
    Look up the survival probability at a specific time point for a specific curve
    
    Args:
        df: DataFrame with time_months, curve_name, and survival_prob columns
        curve_name: Name of the curve to look up
        time_months: Time point in months
        
    Returns:
        Survival probability value or None if not found
    """
    # Make sure we have the required columns
    required_cols = ['time_months', 'survival_prob', 'curve_name']
    for col in required_cols:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return None
    
    # Clean up the dataframe
    # Convert to numeric and drop NaN values
    df = df.copy()
    df['time_months'] = pd.to_numeric(df['time_months'], errors='coerce')
    df['survival_prob'] = pd.to_numeric(df['survival_prob'], errors='coerce')
    df = df.dropna(subset=['time_months', 'survival_prob'])
    
    # Make sure curve_name column is string type
    df['curve_name'] = df['curve_name'].astype(str)
    
    # Try different matching approaches
    # 1. Exact match (case insensitive)
    curve_df = df[df['curve_name'].str.lower() == curve_name.lower()]
    
    # 2. If no match, try partial match
    if curve_df.empty:
        for idx, row_curve in enumerate(df['curve_name'].unique()):
            if curve_name.lower() in row_curve.lower() or row_curve.lower() in curve_name.lower():
                curve_df = df[df['curve_name'] == row_curve]
                print(f"Found partial match: {row_curve}")
                break
    
    # 3. If still no match, use all data
    if curve_df.empty:
        print("No match found, using all data")
        curve_df = df
    
    if curve_df.empty or len(curve_df) == 0:
        print("No data available after filtering")
        return None
    
    # Find the closest time point
    try:
        # Check if we have an exact match for the time point
        exact_match = curve_df[curve_df['time_months'] == time_months]
        if not exact_match.empty:
            return exact_match.iloc[0]['survival_prob']
        
        # Find the closest time point
        closest_idx = (curve_df['time_months'] - time_months).abs().idxmin()
        return curve_df.loc[closest_idx, 'survival_prob']
    except Exception as e:
        print(f"Error finding closest time point: {e}")
        return None
