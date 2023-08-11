# websitenearme
chatbots using langchain, pinecone, to answer business questions
## To run:
### Run locally without flask app
# Step 1 - add keys to dotenv
- Create a dotenv and include openAI api key, pinecone api key, and pinecone env:
OPENAI_API_KEY, PINECONE_ENV, PINECONE_API_KEY

# Step 2 - fully automated so only need to run 1 file
- from a terminal enter:
  ```
  python3 websitenearme_chatbot.py
  ```
### Run flask app


## The Workflow:
# Step 1. 
Set up venv:
```
mkdir websitenearme
cd websitenearme
source ./webnearme-venv/bin/activate
git clone https://github.com/data-science-nerds/websitenearme.git
pip install -r requirements.txt
```


# Step 2. 
Get the data from wordpress site using Tools > Export

# Step 3. 
Convert the raw xml data to text using scrape_website.py 

# Step 4. 
clean the xml data using data_cleansing.py
Read the XML file.
This script will first remove the sections surrounded by comments containing any of the keywords in WP_TERMS. It will then filter out any line that contains any of the stop words in STOP_WORDS. Finally, it will save the cleaned and filtered content to a new text file with the prefix clean_.

### Step 5. 
Upsert data to pinecone.

### Step 6.
Test that results make sense.
