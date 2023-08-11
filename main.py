'''Runs the full end to end generation of a wordpress website to 
pinecone db vector insertion and produces sample outputs
'''

from scrape_website import make_content_file
from data_cleanser import create_cleansed_file
from upsert_pinecone_data_script import upsert_to_pinecone

PREFIX_CLEAN = 'clean_'
WP_TERMS = ["/wp:heading", "/wp:paragraph", "/wp:table", "/wp:image", "wp:generateblocks/container"]
STOP_WORDS = ["OIL", "TIRES", "FOODIES", "BREAKFAST", "LUNCH", "DINNER", "DESERT", "MECHANIC", "Drinks", "drinks", "Ensalada", "Mariscos"]
INDEX_NAME = "websitenearme-fast-api"

def main():
    """
    Entry point for the script.
    Scrapes a website, cleanses the data, and saves it in a structured directory.
    """
    url = 'https://ai-architects.cloud'
    
    # First, turn the xml file into valid data
    name_space, xml_file_path, xml_file_name, content_file_path = make_content_file(url)
    
    # Create a cleansed file from the scraped content
    cleaned_file_path = create_cleansed_file(content_file_path, PREFIX_CLEAN, WP_TERMS, STOP_WORDS, name_space)

    # Upsert the cleaned data to Pinecone
    upsert_to_pinecone(cleaned_file_path, name_space, INDEX_NAME)

if __name__ == "__main__":
    main()


# '''Runs the full end to end generation of a wordpress website to 
# pinecone db vector insertion and produces sample outputs
# '''

# from scrape_website import make_content_file
# from data_cleanser import create_cleansed_file

# PREFIX_CLEAN = 'clean_'
# WP_TERMS = ["/wp:heading", "/wp:paragraph", "/wp:table", "/wp:image", "wp:generateblocks/container"]
# STOP_WORDS = ["OIL", "TIRES", "FOODIES", "BREAKFAST", "LUNCH", "DINNER", "DESERT", "MECHANIC", "Drinks", "drinks", "Ensalada", "Mariscos"]


# if __name__ == "__main__":
#     name_space = '' #  https_websitenearme , used for pinecone namespace
#     xml_file_path = '' # websitenearme  file name without the protocol
#     content_file_path = '' # 'website_content/https_websitenearme/scraped_https_websitenearme.txt'
#     url = 'https://websitenearme.online'
#     # first turn the xml file into valid data
#     name_space, xml_file_path, xml_file_name, content_file_path = make_content_file(url)
#     # import pdb; pdb.set_trace()
#     cleaned_file_path = cleansed_file_path = create_cleansed_file(content_file_path, PREFIX_CLEAN, WP_TERMS, STOP_WORDS)
#     import pdb; pdb.set_trace()