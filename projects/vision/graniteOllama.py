import base64
import requests
import json
import ast


with open("chart.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "granite3.2-vision:2b",
    "prompt": "Extract all data points from this chart and return them as structured JSON.Make sure to handle all rows. Empty columns might be used as separators. Provide compact and complete output. make sure to have a terminating curly bracket at the end of json.",
    "images": [image_b64],
"options": {
        "num_predict": 10240,  # increase this from the default (default is 128)
	"temperature":0.0
    }
}

response = requests.post("http://localhost:11434/api/generate", json=payload)


# Step 4: Accumulate full output from streaming lines
raw_text = ""
for line in response.iter_lines():
    if line:
        try:
            data = json.loads(line)
            raw_text += data.get("response", "")
        except json.JSONDecodeError:
            pass  # tolerate malformed intermediate lines

print(raw_text)

# Step 5: Fix single-quote Python dict format → proper JSON
try:
    python_dict = ast.literal_eval(raw_text.strip())
    with open("extracted_data.json", "w") as f:
        json.dump(python_dict, f, indent=2)
    print("✅ Converted and saved Python-style output as JSON.")
except Exception as e:
    print("❌ Still failed to convert output.")


# Step 5: Try parsing the full string
try:
    parsed = json.loads(raw_text)
    with open("extracted_data.json", "w") as f:
        json.dump(parsed, f, indent=2)
    print("✅ Saved parsed JSON to extracted_data.json")
except json.JSONDecodeError as e:
    print("❌ Failed to parse final output as JSON")
    print("🔍 Raw output preview:\n", raw_text[:500])






# Final result
#print("\n🧾 Full Model Response:\n")
#print(output_text)


