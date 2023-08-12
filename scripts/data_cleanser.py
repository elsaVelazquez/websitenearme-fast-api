import os
import re
from bs4 import BeautifulSoup, Comment
from typing import List


def remove_sections_based_on_comments(content: str, drop_keywords: List[str]) -> str:
    """
    Removes sections of the content surrounded by comments containing keywords from drop_keywords.
    Also removes standalone comment keywords.
    Args:
    - content: The content string to be cleansed.
    - drop_keywords: List of keywords to drop from the content.

    Returns:
    - Cleaned content as a string.
    """
    for keyword in drop_keywords:
        # Remove sections surrounded by the keyword
        content = re.sub(r'<!--\s*' + keyword + r'\s*-->(.*?)<!--\s*/' + keyword + r'\s*-->', '', content, flags=re.DOTALL)
        
        # Remove standalone comment keywords
        content = re.sub(r'<!--\s*(/)?' + keyword + r'\s*-->', '', content)
        
    return content.strip()


def filter_out_stop_words(content: str, stop_words: List[str]) -> str:
    """
    Filters out lines in the content that contain any of the stop words.
    Args:
    - content: The content string to be filtered.
    - stop_words: List of stop words to filter out from the content.

    Returns:
    - Content string after filtering stop words.
    """
    lines = content.split('\n')
    filtered_lines = [line for line in lines if not any(word in line.upper() for word in stop_words)]
    return '\n'.join(filtered_lines)


def save_to_file(content: str, output_file_path: str) -> None:
    """
    Saves the content to the specified file.
    Args:
    - content: The content string to be saved.
    - output_file_path: The path where content will be saved.

    Returns:
    - None
    """
    with open(output_file_path, 'w') as file:
        file.write(content)


def remove_html_tags_and_comments(content: str) -> str:
    """
    Extracts text content using BeautifulSoup, discarding all tags.
    Args:
    - content: The content string from which HTML tags and comments need to be removed.

    Returns:
    - Cleaned content as a string.
    """
    soup = BeautifulSoup(content, 'lxml')
    
    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    return soup.get_text(separator="\n", strip=True)


def remove_duplicates(content):
    seen = set()
    result = []

    for line in content.splitlines():
        if line not in seen:
            seen.add(line)
            result.append(line)

    return "\n".join(result)


def remove_css_block(content, start_comment, end_comment):
    # Regular expression pattern to match content between start and end comments
    pattern = re.compile(re.escape(start_comment) + ".*?" + re.escape(end_comment), re.DOTALL)
    return re.sub(pattern, '', content)


def create_cleansed_file(input_file: str, PREFIX_CLEAN: str, WP_TERMS: List[str], STOP_WORDS: List[str], name_space: str) -> str:
    """
    Cleanses the input file content based on given terms and stop words.
    Args:
    - input_file: Path to the input file.
    - PREFIX_CLEAN: Prefix for the cleansed file.
    - WP_TERMS: List of WordPress terms to be removed.
    - STOP_WORDS: List of stop words to be filtered out.
    - name_space: Namespace used for directory structure.

    Returns:
    - Path to the cleansed file.
    """
    drop_sentences = WP_TERMS + STOP_WORDS

    # Read content from the input file
    with open(input_file, 'r') as file:
        content = file.read()

    # Remove sections based on comments
    cleaned_content = remove_sections_based_on_comments(content, drop_sentences)
    
    # Filter out stop words
    filtered_content = filter_out_stop_words(cleaned_content, STOP_WORDS)

    no_html_content = remove_html_tags_and_comments(filtered_content)
    
    no_dups = remove_duplicates(no_html_content)
    
    no_css_block = remove_css_block(no_dups, '/* GeneratePress Site CSS */', '/* End GeneratePress Site CSS */')
    
    # Derive the output file path based on the desired structure
    output_file_name = f"{PREFIX_CLEAN}{os.path.basename(input_file).replace('scraped_', '').replace('.xml', '.txt')}"
    output_dir = os.path.join("data/processed/website_content", name_space)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, output_file_name)
    
    save_to_file(no_css_block, output_file_path)

    print(f"Cleaned content saved to {output_file_path}")
    return output_file_path


if __name__ == "__main__":
    cleaned_file_path = create_cleansed_file(input_file, PREFIX_CLEAN, WP_TERMS, STOP_WORDS, name_space)
