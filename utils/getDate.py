from bs4 import BeautifulSoup
import re
import datetime
import sys

# Sample HTML content
html_content = """
<!DOCTYPE html>
<html lang="de">

<head>
    <title>Karlsruhe: Ratsinformation - Gemeinsame Sitzung Jugendhilfe- und Sozialausschuss (nicht öffentlich) </title>
    <meta name="description" content="Aktuelle Ratsinformationen aus Karlsruhe, Sitzungen, Anträge, Mandate">
 </head>
 <body>
 <table class="contenttable">
                    <tbody>
                        <tr>
                            <td>Datum</td>
                            <td class="">9. Februar 2017,
                                16 Uhr</td>
                        </tr>
                        <tr>
                            <td>Ort</td>
                            <td class=""></td>
                        </tr>
                    </tbody>
                </table>
</body>
</html>
                """


if len(sys.argv) > 1:
    fn = sys.argv[1]
    with open(fn) as f:
        html_content = f.read()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

head = soup.find("head")
title = head.find("title").contents[0]

# Find the first occurrence of a table with class 'contenttable'
first_table = soup.find('table', class_='contenttable')

# Initialize variables to store date and location
extracted_date = None
extracted_location = None

if first_table:
    # Find all rows in the table
    rows = first_table.find_all('tr')
    
    for row in rows:
        # Get all cells in the row
        cells = row.find_all('td')
        
        # Check if the row contains 'Datum' and extract its corresponding value
        if len(cells) >= 2 and cells[0].get_text(strip=True) == 'Datum':
            extracted_date = cells[1].get_text(strip=True)
        
        # Check if the row contains 'Ort' and extract its corresponding value
        if len(cells) >= 2 and cells[0].get_text(strip=True) == 'Ort':
            extracted_location = cells[1].get_text(strip=True)
            


# Handle different month formats (German months and numbers)
month_map = {
    'Jan': '1', 'Feb': '2', 'Mär': '3','Maer': '3', 'Apr': '4', 'Mai': '5', 'Jun': '6',
    'Jul': '7', 'Aug': '8', 'Sep': '9', 'Okt': '10', 'Nov': '11', 'Dez': '12'
}

# Convert the extracted date to a datetime object if it exists
if extracted_date:
    try:
        extracted_date = re.sub(r'\n', '', extracted_date)  # Remove newlines
        extracted_date = re.sub(r',', ' ', extracted_date)  # Replace comma with space
        extracted_date = re.sub(r'\.', ' ', extracted_date)  # Replace . with space
        extracted_date = re.sub(r'\s+', ' ', extracted_date)  # Normalize spaces and newlines
        extracted_date = re.sub(r'Uhr', '', extracted_date).strip()  # Remove 'Uhr' if it exists
        dateitems = extracted_date.split(" ")
        month_code = dateitems[1][:3] # first 3 characters
        month = month_map.get(month_code,dateitems[1])
        dateitems[1] = month
        if len(dateitems) < 4:
            dateitems.append("00")
        time = dateitems[3]
        #print("time",time,dateitems)
        if len(time.split(":")) == 1:
            try:
                time += ":" + str(int(dateitems[4]))
            except:
                time += ":00"
        else:
                time += ":00"
            
        dateitems[3] = time
        extracted_date = datetime.datetime.strptime(" ".join(dateitems[:4]), '%d %m %Y %H:%M')  # Convert to datetime object
    except ValueError as e:
        print("Error parsing date:", e)
        extracted_date = None
        
# Print the extracted date and location
if title and extracted_date:
    print("Title:", title)
    print("Extracted Date:", extracted_date)
    if extracted_location != "":
        print("Extracted Location:", extracted_location)
else:
    print("Missing infos")


