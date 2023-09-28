"""
Runs the full end-to-end generation of content to 
Pinecone DB vector insertion and produces sample outputs.
"""

PREFIX_CLEAN = 'clean_'
WP_TERMS = ["/wp:heading", "/wp:paragraph", "/wp:table", "/wp:image", "wp:generateblocks/container"]
STOP_WORDS = ["OIL", "TIRES", "FOODIES", "BREAKFAST", "LUNCH", "DINNER", "DESERT", "MECHANIC", "Drinks", "drinks", "Ensalada", "Mariscos"]
INDEX_NAME = "websitenearme-fast-api"

def main_ada(url):
    """
    Entry point for the script.
    Orchestrates the scraping of a website, cleansing of the data, 
    and upserting the data into Pinecone.
    """
    from scripts.xml_content_parser import make_content_file

    from scripts.data_cleanser import create_cleansed_file

    from scripts.upserts.upsert_pinecone_ada_engine_script import upsert_to_pinecone
    
    # Extract and save content from the website
    name_space, content_file_path = make_content_file(url)
    
    # Cleanse the scraped content
    cleaned_file_path = create_cleansed_file(content_file_path, PREFIX_CLEAN, WP_TERMS, STOP_WORDS, name_space)

    # Upsert the cleansed data to Pinecone
    upsert_to_pinecone(cleaned_file_path, name_space, INDEX_NAME)


def main_hybrid(url):
    """
    Entry point for the script.
    Orchestrates the scraping of a website, cleansing of the data, 
    and upserting the data into Pinecone.
    """
    from scripts.xml_content_parser import make_content_file
    from scripts.data_cleanser import create_cleansed_file
    from scripts.upserts.upsert_pinecone_hybrid_script import upsert_to_pinecone
    
    # Extract and save content from the website
    name_space, content_file_path = make_content_file(url)
    
    # Cleanse the scraped content
    cleaned_file_path = create_cleansed_file(content_file_path, PREFIX_CLEAN, WP_TERMS, STOP_WORDS, name_space)

    # Upsert the cleansed data to Pinecone
    upsert_to_pinecone(cleaned_file_path, name_space, INDEX_NAME)

def main(driver, url):
    if driver == 'hybrid':
        main_hybrid(url)
    else: # if driver == 'ada':
        main_ada(url)
        
    
if __name__ == "__main__":
    url = 'https://websitenearme.onlin'
    
    driver = 'hybrid' #'ada'
    main(driver, url)
