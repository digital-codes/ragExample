import base64
import requests
import json
import ast

import io
from PIL import Image, ImageOps
import pandas as pd
import matplotlib.pyplot as plt


def encode_image(image: Image.Image, format: str = "png", max_size: int = 800) -> str:
    """
    Resize and encode a PIL image to base64 data URI for use with vision models.
    Keeps aspect ratio and limits max width or height to `max_size` to manage token usage.
    """
    # Apply EXIF orientation and ensure RGB
    image = ImageOps.exif_transpose(image) or image
    image = image.convert("RGB")

    # Resize if necessary
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Encode to base64 data URI
    buffer = io.BytesIO()
    image.save(buffer, format)
    encoding = base64.b64encode(buffer.getvalue()).decode("utf-8")
    uri = f"data:image/{format};base64,{encoding}"
    return uri


#with open("chart.png", "rb") as f:
#    image_b64 = base64.b64encode(f.read()).decode("utf-8")


# Load and encode image
input_filename = "chart.png"
image = Image.open(input_filename)
image_b64 = encode_image(image)


payload = {
    "model": "granite3.2-vision:2b",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_b64}},
                # {"type": "text", "text": "What does the image contain?"} # works
                {"type": "text", "text": "Extract all data points from this chart and return them as structured JSON.Make sure to handle all rows. Empty columns might be used as separators. Extract exactly one y value per x value. Provide compact and complete output."}
            ]
        }
    ],
	"temperature":0.0,
    "max_tokens": 2048,
    "stream": False,
    "top_p": 1
}

def toDf1(data):
    try:
        df = pd.DataFrame(parsed)
        return pd.DataFrame(data)
    except Exception as e:
        print("Error converting to DataFrame:", e)
        return None

def toDf2(data):
    try:
        years = parsed['categories']
        labels = parsed['labels']
        values = parsed['values']

        data = []
        for label_idx, label in enumerate(labels):
            for year_idx, year in enumerate(years):
                value = values[label_idx][year_idx]
                data.append({
                    'year': year,
                    'label': label,
                    'value': int(value.replace(',', ''))  # Ensure numeric format
                })

        return pd.DataFrame(data)
    except Exception as e:
        print("Error converting to DataFrame:", e)
        return None


parsers = [toDf1,toDf2]

#response = requests.post("http://localhost:11434/api/generate", json=payload)
response = requests.post("http://localhost:8080/v1/chat/completions", json=payload)

print("Response status code:", response.status_code)
# Step 3: Check for errors in the response
if response.status_code != 200:
    print("❌ Error in response:", response.status_code)
    print("🔍 Response content:", response.text)
    exit(1)

result = response.json()
#print(result)

content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
print("Response content:", content)
try:
    parsed = json.loads(content)
    print("Parsed JSON:", json.dumps(parsed, indent=2))
    # Save the parsed JSON as a file using the input filename as a template
    output_filename = input_filename.rsplit('.', 1)[0] + "_output.json"
    with open(output_filename, "w") as json_file:
        json.dump(parsed, json_file, indent=2)
    print(f"Parsed JSON saved to {output_filename}")
    df = None 
    for parser in parsers:
        df = parser(parsed)
        if df is not None:
            print("DataFrame:")
            print(df)
            break
    if df is None:
        print("No valid parser found for the JSON data.")
        raise ValueError("No valid parser found for the JSON data.")

    # Save the DataFrame to a CSV file
    output_csv_filename = input_filename.rsplit('.', 1)[0] + "_output.csv"
    df.to_csv(output_csv_filename, index=False)
    print(f"DataFrame saved to {output_csv_filename}")

    # Display the table using matplotlib
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.5))  # Adjust height based on number of rows
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    plt.show()
    # Plot a line chart
    try:
        plt.figure(figsize=(10, 6))
        for label in df['label'].unique():
            subset = df[df['label'] == label]
            plt.plot(subset['year'], subset['value'], marker='o', label=label)

        plt.title("Line Chart of Extracted Data")
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.legend(title="Labels")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("Error plotting line chart:", e)

    # Plot a stacked bar chart
    try:
        plt.figure(figsize=(10, 6))
        pivot_df = df.pivot(index='year', columns='label', values='value').fillna(0)
        pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6))

        plt.title("Stacked Bar Chart of Extracted Data")
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.legend(title="Labels")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Error plotting stacked bar chart:", e)
    
except json.JSONDecodeError as e:
    print("❌ Failed to parse JSON:", e)
    print("🔍 Raw content:", content)
    
    
