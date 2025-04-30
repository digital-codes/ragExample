import pandas as pd
import requests
import sys
import json

import private_remote as pr 
api_key = pr.openAi["apiKey"]

# Step 1: Get file path and read CSV
csv_path = input("Enter the path to the CSV file: ")

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("Error: File not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

# Step 2: Generate detailed summary
summary_parts = []

# Column names and types
summary_parts.append("🧾 Columns and Types:")
summary_parts.append(str(df.dtypes))

# Missing values
missing = df.isnull().sum()
summary_parts.append("\n🧩 Missing Values:")
summary_parts.append(str(missing))

# Describe numerical and categorical data
summary_parts.append("\n📊 Statistical Summary:")
try:
    desc = df.describe(include='all', datetime_is_numeric=True).transpose()
    summary_parts.append(desc.to_string())
except Exception as e:
    summary_parts.append(f"Could not compute describe(): {e}")

# Unique values for small object columns
summary_parts.append("\n🔢 Sample Unique Values:")
for col in df.columns:
    if df[col].dtype == 'object' and df[col].nunique() < 10:
        sample = df[col].unique()
        summary_parts.append(f"{col}: {sample}")

# Optional: year/date range
for col in df.columns:
    if "year" in col.lower() or "date" in col.lower():
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            min_date = df[col].min()
            max_date = df[col].max()
            summary_parts.append(f"\n📅 Range of '{col}': {min_date} to {max_date}")
        except:
            pass

# add full data
summary_parts.append("\n🔢 Full Dataset as stringified json:")
summary_parts.append(json.dumps(df.to_json()))


# Combine summary
summary_text = "\n".join(summary_parts)

print("\n✅ Data Summary Generated:\n")
print(summary_text)

# Step 3: Get user question
question = input("\n🧠 Enter your question about the data: ")

# Step 5: Prepare prompt
prompt = f"""
You are a data analysis assistant. Use the dataset summary below to answer the user's question.

--- Dataset Summary ---
{summary_text}

--- User Question ---
{question}
"""

# Step 6: Send to OpenAI API (GPT-4o)
url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
data = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful data analyst."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.3
}

try:
    response = requests.post(url, headers=headers, json=data)
except requests.exceptions.RequestException as e:
    print(f"Error connecting to OpenAI API: {e}")
    sys.exit(1)

if response.status_code != 200:
    try:
        error_msg = response.json().get("error", {}).get("message", "Unknown error")
    except:
        error_msg = response.text
    print(f"OpenAI API error ({response.status_code}): {error_msg}")
    sys.exit(1)

# Step 7: Extract and print response
try:
    answer = response.json()['choices'][0]['message']['content']
    print("\n💡 GPT Answer:\n")
    print(answer.strip())
except Exception as e:
    print(f"Unexpected response format: {e}")
    print(response.text)
