import os
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from typing import List, Tuple, Optional


def parse_xml(file_path: str, exclusions: List[str]) -> List[str]:
    """
    Parse the XML file and extract relevant content.

    :param file_path: Path to the XML file.
    :param exclusions: List of domains to exclude.
    :return: List of extracted sentences from the XML content.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    all_sentences = []

    for item in root.findall('.//item'):
        link = item.find('link').text
        title = item.find('title').text
        content = item.find('content:encoded', namespaces={'content': 'http://purl.org/rss/1.0/modules/content/'}).text

        if any(exclusion in link for exclusion in exclusions):
            continue

        if title:
            all_sentences.append(title)
        if content:
            all_sentences.append(content)

    return all_sentences


def save_content_to_file(content: List[str], namespace: str) -> str:
    """
    Save extracted content to a file.

    :param content: List of sentences to save.
    :param namespace: Namespace to determine the directory and filename.
    :return: Path to the saved file.
    """
    directory = os.path.join('data/processed/website_content', namespace)
    content_file_path = os.path.join(directory, f'scraped_{namespace}.txt')

    if not os.path.exists(directory):
        os.makedirs(directory)

    content_str = "\n".join(content)
    with open(content_file_path, 'w') as file:
        file.write(content_str)

    print(f"Content saved to {content_file_path}")
    return content_file_path


def get_namespace_from_url(url: str) -> str:
    """
    Extract the namespace from the given URL.

    :param url: The URL to extract namespace from.
    :return: Namespace in the format scheme_domain.
    """
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme
    domain = parsed_url.netloc.replace('www.', '')
    name_part = domain.split('.')[0]

    return f"{scheme}_{name_part}"


def find_xml_file(name_space: str) -> str:
    """
    Find the XML file that matches the given namespace in the 'raw_data' directory.

    :param name_space: Namespace to search for.
    :return: Name of the matching XML file.
    """
    files = os.listdir('data/raw_data')
    matching_files = [f for f in files if name_space in f and f.endswith('.xml')]

    if matching_files:
        return matching_files[0]
    else:
        raise FileNotFoundError(f"No XML file found for namespace '{name_space}'")


def get_file_identifier_from_url(url: str) -> str:
    """
    Extract the domain name from the URL.

    :param url: The URL to extract domain name from.
    :return: Extracted domain name.
    """
    domain = urlparse(url).netloc.replace('www.', '')
    return domain.split('.')[0]


def make_content_file(url: str) -> Optional[Tuple[str, str]]:
    """
    Process the XML content file and save the extracted content.

    :param url: URL to determine the XML file and namespace.
    :return: Tuple containing namespace and path to the saved content file.
    """
    name_space = get_namespace_from_url(url)
    file_identifier = get_file_identifier_from_url(url)
    exclusions = ["excluded_domain1.com", "excluded_domain2.com"]

    xml_file_name = find_xml_file(file_identifier)
    xml_file_path = os.path.join('data/raw_data', xml_file_name)

    if not os.path.exists(xml_file_path):
        print(f"File {xml_file_path} not found!")
        return

    content = parse_xml(xml_file_path, exclusions)
    content_file_path = save_content_to_file(content, name_space)
    return name_space, content_file_path


if __name__ == "__main__":
    # Define 'url' before calling the make_content_file function
    url = "YOUR_URL_HERE"
    name_space, content_file_path = make_content_file(url)
