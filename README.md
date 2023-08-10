# websitenearme
chatbots using langchain, pinecone, to answer business questions


Step 1. 
Set up venv:
mkdir websitenearme
cd websitenearme
source ./webnearme-venv/bin/activate
git clone https://github.com/data-science-nerds/websitenearme.git
pip install -r requirements.txt


Step 2. 
Get the data from wordpress site using Tools > Export

Step 3. 
Convert the raw xml data to text using scrape_website.py 

Step 4. 
# clean the xml data using data_cleansing.py
Read the XML file.
This script will first remove the sections surrounded by comments containing any of the keywords in WP_TERMS. It will then filter out any line that contains any of the stop words in STOP_WORDS. Finally, it will save the cleaned and filtered content to a new text file with the prefix clean_.