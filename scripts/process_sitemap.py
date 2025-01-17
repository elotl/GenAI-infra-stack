
import click
import json
import os

from langchain_community.document_loaders.sitemap import SitemapLoader
from urllib.parse import urlparse


def build_filename_from_url(url: str, extension: str = "json") -> str:
    parsed_url = urlparse(url)
    # Use the netloc and path as base components
    base_name = f"{parsed_url.netloc}{parsed_url.path}".replace("/", "_").strip("_")
    # Remove query parameters and fragments if present
    base_name = base_name.split("?")[0].split("#")[0]
    # Construct the filename
    filename = f"{base_name}.{extension}"
    # Ensure filename is valid on the filesystem
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)


@click.command()
@click.argument("input_url")
@click.argument("output_dir", type=click.Path())
def process_data(input_url: str, output_dir: str):
    """
    Process data from INPUT_URL and save to OUTPUT_FILE.
    """
    sitemap_loader = SitemapLoader(
        web_path=input_url, filter_urls=["^((?!.*/v.*).)*$"]
    )
    sitemap_loader.requests_per_second = 1
    docs = sitemap_loader.load()
    print("Count of sitemap docs loaded:", len(docs))

    for doc in docs:
        data = {
            "text": doc.page_content,
            # source key exists in doc metadata
            "metadata": doc.metadata,
        }
    
        output_filename = build_filename_from_url(doc.metadata["source"])
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    process_data()