import requests
from bs4 import BeautifulSoup

# Fetching WordPress content for testing purposes
url = 'https://websitenearme.online'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extracting the main content of the article
wp_content = soup.find('div', class_='post-content').prettify()

# Cleaning the fetched content
cleaned_content = clean_wordpress_content(wp_content)

cleaned_content[:1000]  # Displaying the first 1000 characters for brevity
