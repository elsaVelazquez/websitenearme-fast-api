import os
import re
from bs4 import BeautifulSoup, Comment

def remove_sections_based_on_comments(content, drop_keywords):
    """
    Removes sections of the content surrounded by comments containing keywords from drop_keywords.
    Also removes standalone comment keywords.
    Returns the cleaned content as a string.
    """
    for keyword in drop_keywords:
        # Remove sections surrounded by the keyword
        content = re.sub(r'<!--\s*' + keyword + r'\s*-->(.*?)<!--\s*/' + keyword + r'\s*-->', '', content, flags=re.DOTALL)
        
        # Remove standalone comment keywords
        content = re.sub(r'<!--\s*(/)?' + keyword + r'\s*-->', '', content)
        
    return content.strip()

def filter_out_stop_words(content, stop_words):
    """
    Filters out lines in the content that contain any of the stop words.
    """
    lines = content.split('\n')
    filtered_lines = [line for line in lines if not any(word in line.upper() for word in stop_words)]
    return '\n'.join(filtered_lines)


def save_to_file(content, output_file_path):
    """
    Saves the content to the specified file.
    """
    with open(output_file_path, 'w') as file:
        file.write(content)


def remove_html_tags_and_comments(content):
    """
    Extracts text content using BeautifulSoup, discarding all tags.
    Returns the cleaned content as a string.
    """
    soup = BeautifulSoup(content, 'lxml')
    
    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    return soup.get_text(separator="\n", strip=True)


def create_cleansed_file(INPUT_FILE, OUTPUT_PREFIX, WP_TERMS, STOP_WORDS):
    drop_sentences = WP_TERMS + STOP_WORDS

    # Read content from the input file
    with open(INPUT_FILE, 'r') as file:
        content = file.read()

    # Remove sections based on comments
    cleaned_content = remove_sections_based_on_comments(content, drop_sentences)
    
    # Filter out stop words
    filtered_content = filter_out_stop_words(cleaned_content, STOP_WORDS)

    no_html_content = remove_html_tags_and_comments(filtered_content)
    
    # Save the cleaned and filtered content to a new file
    output_file_path = os.path.join(OUTPUT_PREFIX + os.path.basename(INPUT_FILE).replace('.xml', '.txt'))
    save_to_file(no_html_content, output_file_path)

    print(f"Cleaned content saved to {output_file_path}")




if __name__ == "__main__":
    INPUT_FILE = 'websitenearme.WordPress.2023-08-09.xml'
    OUTPUT_PREFIX = 'clean_'
    
    WP_TERMS = ["/wp:heading", "/wp:paragraph", "/wp:table", "/wp:image", "wp:generateblocks/container"]
    STOP_WORDS = ["OIL", "TIRES", "FOODIES", "BREAKFAST", "LUNCH", "DINNER", "DESERT", "MECHANIC"]
    create_cleansed_file(INPUT_FILE, OUTPUT_PREFIX, WP_TERMS, STOP_WORDS)

