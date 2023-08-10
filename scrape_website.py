import os
import xml.etree.ElementTree as ET
from urllib.parse import urlparse


def parse_xml(file_path, exclusions):
    tree = ET.parse(file_path)
    root = tree.getroot()

    all_sentences = []

    for item in root.findall('.//item'):
        # Extract the data from the XML
        link = item.find('link').text
        title = item.find('title').text
        content = item.find('content:encoded', namespaces={'content': 'http://purl.org/rss/1.0/modules/content/'}).text

        # Check for exclusions
        skip = any(exclusion in link for exclusion in exclusions)
        if skip:
            continue

        # Append to all_sentences only if they are not None
        if title:
            all_sentences.append(title)
        if content:
            all_sentences.append(content)

    return all_sentences


def save_content_to_file(content, namespace) -> str:
    # Define the directory and file paths
    directory = os.path.join('website_content', namespace)
    content_file_path = os.path.join(directory, f'scraped_{namespace}.txt')

    # Check if the directory exists; if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Convert the list of sentences into a single string
    content_str = "\n".join(content)

    # Write (or overwrite) the content to the file
    with open(content_file_path, 'w') as file:
        file.write(content_str)

    print(f"Content saved to {content_file_path}")
    return content_file_path



def get_namespace_from_url(url):
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme
    domain = parsed_url.netloc.replace('www.', '')
    name_part = domain.split('.')[0]

    return f"{scheme}_{name_part}"


def find_xml_file(name_space):
    files = os.listdir()
    matching_files = [f for f in files if name_space in f and f.endswith('.xml')]

    if matching_files:
        return matching_files[0]
    else:
        raise FileNotFoundError(f"No XML file found for namespace '{name_space}'")


def get_file_identifier_from_url(url):
    """Extract only the domain name from the URL for file searching."""
    domain = urlparse(url).netloc.replace('www.', '')
    return domain.split('.')[0]

def make_content_file(url):
    name_space = get_namespace_from_url(url)
    file_identifier = get_file_identifier_from_url(url)
    exclusions = ["excluded_domain1.com", "excluded_domain2.com"]

    xml_file_name = find_xml_file(file_identifier)
    xml_file_path = xml_file_name

    if not os.path.exists(xml_file_path):
        # Do not raise error, it will crash whole instance
        # instead just print a statement for now
        print(f"File {xml_file_path} not found!")

    content = parse_xml(xml_file_path, exclusions)
    content_file_path = save_content_to_file(content, name_space)
    return name_space, file_identifier, xml_file_name, content_file_path


if __name__ == "__main__":
    name_space, xml_file_path, xml_file_name, content_file_path = make_content_file(url)

