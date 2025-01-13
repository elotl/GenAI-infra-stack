# AI Generated script

import os
import json
from configparser import ConfigParser
from typing import Any, Dict, List, Union

import click

def clean_text(text: Any) -> str:
    """Clean and standardize text content."""
    if text is None or text == "":
        return ""
    return str(text).strip().replace("\n", " ").replace("\r", " ")


def parse_list(value: Any) -> List[str]:
    """Convert a field value to a list of strings."""
    if not value:  # Handles None, empty string, empty list
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return [str(value)]


def extract_field_values(
    data: Dict[str, Any], 
    keys: List[str], 
    matcher: callable
) -> List[str]:
    """Extract values from fields that match a given condition."""
    return [
        str(data[key])
        for key in keys
        if key in data and data[key] and matcher(key)
    ]


def process_item(
    data: Dict[str, Any],
    keys: List[str],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a single JSON item using the provided configuration."""
    # Extract data using different matching criteria
    result_text = []
    
    # Process composite text fields
    for field, column in config["composite_text_fields"].items():
        if column in data and data[column]:
            result_text.append(f"{field.lower()}: {clean_text(data[column])}")
    
    # Process prefix fields
    for target_field, prefix in config["prefix_fields"].items():
        values = extract_field_values(
            data, keys, 
            lambda k: k.startswith(prefix)
        )
        if values:
            result_text.append(f"{target_field.lower()}: {' '.join(values)}")
    
    # Process substring fields
    for target_field, substring in config["substring_fields"].items():
        values = extract_field_values(
            data, keys,
            lambda k: substring in k.lower()
        )
        if values:
            result_text.append(f"{target_field.lower()}: {' '.join(values)}")
    
    # Process list fields
    for field in config["list_fields"]:
        if field in data:
            values = parse_list(data[field])
            if values:
                result_text.append(f"{field.lower()}: {', '.join(values)}")
    
    # Build metadata
    metadata = {
        field.lower(): clean_text(data.get(column, ""))
        for field, column in config["metadata_fields"].items()
    }
    
    # Add source URL
    metadata_field = config["metadata_field"]
    metadata["source"] = config["zendesk_url"] + metadata[metadata_field] + ".json"
    
    return {
        "text": "\n".join(result_text),
        "metadata": metadata
    }


def prepare_documents(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Prepare documents for embedding using the provided configuration."""
    # Ensure data is a list
    items = [data] if isinstance(data, dict) else data
    
    # Get all unique keys from the JSON data
    keys = sorted({key for item in items for key in item})
    
    # Process each item
    return [process_item(item, keys, config) for item in items]


def load_config(config_file: str) -> Dict[str, Any]:
    """Load and parse configuration from INI file."""
    config = ConfigParser()
    config.read(config_file)
    
    return {
        "prefix_fields": dict(config["PrefixFields"]) if "PrefixFields" in config else {},
        "substring_fields": dict(config["SubstringFields"]) if "SubstringFields" in config else {},
        "list_fields": [
            f.strip() 
            for f in config.get("ListFields", "fields", fallback="").split(",")
            if f.strip()
        ],
        "composite_text_fields": dict(config["CompositeTextFields"]) if "CompositeTextFields" in config else {},
        "metadata_fields": dict(config["MetadataFields"]) if "MetadataFields" in config else {},
        "zendesk_url": config.get("TicketUrl", "zendesk_url", fallback=""),
        "metadata_field": config.get("TicketUrl", "metadata_field", fallback=""),
    }


def save_documents(documents: List[Dict], output_dir: str) -> None:
    """Save documents as individual JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, doc in enumerate(documents):
        output_file = os.path.join(output_dir, f"item_{i}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("config_file", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def main(input_file: str, config_file: str, output_dir: str) -> None:
    """
    Process JSON data using the provided configuration and save results.
    
    INPUT_FILE: JSON file containing the data to process
    CONFIG_FILE: INI configuration file
    OUTPUT_DIR: Directory to save the processed documents
    """
    # Load configuration
    click.echo(f"Loading configuration from {config_file}")
    config = load_config(config_file)
    
    # Load JSON data
    click.echo(f"Reading data from {input_file}")
    data = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    click.echo(f"Warning: Skipping invalid JSON line: {str(e)}")
                    continue
    
    if not data:
        click.echo("Error: No valid JSON objects found in the input file")
        return
    
    click.echo(f"Found {len(data)} JSON objects")
    
    # Process documents
    click.echo("Processing documents...")
    documents = prepare_documents(data, config)
    
    # Save results
    click.echo(f"Saving {len(documents)} documents to {output_dir}")
    save_documents(documents, output_dir)


if __name__ == "__main__":
    main()
