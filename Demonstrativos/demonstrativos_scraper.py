from bs4 import BeautifulSoup
import pandas as pd
import os
import sys

"""COLOQUE O NOME DA ACAO COMO PARAMETRO"""
acao = sys.argv[1]

# Path to the local HTML file
html_file_path = f'HTML-Tables\{acao}.html'  # Replace with the actual file path

# Read the HTML file
with open(html_file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Find the table element (assuming it's the first table in the file)
table = soup.find('table')

# Extract headers
headers = []
for th in table.find_all('th'):
    headers.append(th.text.strip())

# Extract rows
rows = []
for tr in table.find_all('tr')[1:]:  # Skip the header row
    cells = tr.find_all('td')
    row = [cell.text.strip() for cell in cells]
    rows.append(row)

# Create a DataFrame using pandas
df = pd.DataFrame(rows, columns=headers)

# Display the DataFrame
print(df)

# Optionally, save the DataFrame to a CSV file
csv_file_path = os.path.join("Tables",f'{acao}.csv')  # Replace with your desired output path
df.to_csv(csv_file_path, index=False)

print(f"Table successfully extracted and saved to {csv_file_path}")
