import os
import re
import json
from configparser import ConfigParser
from typing import Any, Dict, List, Union

import click

def clean_text(text: Any) -> str:
    """Clean and standardize text content."""
    if text is None or text == "":
        return ""
    result = str(text).strip().replace("\n", " ").replace("\r", " ")
    result = re.sub(r'-{3,}', '-', result)
    return re.sub(r' {2,}', ' ', result)

def get_nested_value(data: Dict[str, Any], field_path: str) -> Any:
    """Extract value from nested dictionary using dot notation.
    
    Args:
        data: Dictionary containing the data
        field_path: Path to the field using dot notation (e.g., "submitter.name")
    
    Returns:
        The value at the specified path or None if not found
    """
    current = data
    parts = field_path.split('.')
    
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and current:
            # If it's a list, try to get the first item's attribute
            if isinstance(current[0], dict):
                current = current[0].get(part)
            else:
                return None
        else:
            return None
        
        if current is None:
            return None
            
    return current

def parse_list(value: Any) -> List[str]:
    """Convert a field value to a list of strings."""
    if not value:  # Handles None, empty string, empty list
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        # Handle cases where list items might be dictionaries
        processed_items = []
        for item in value:
            if isinstance(item, dict):
                # Extract name or id from dictionary
                item_str = item.get('name', item.get('id', ''))
                if item_str:
                    processed_items.append(str(item_str))
            elif item:
                processed_items.append(str(item))
        return processed_items
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

def extract_comments_text(comments: List[Dict[str, Any]]) -> str:
    """Extract readable text from comments array."""
    if not comments:
        return ""
    
    comment_texts = []
    for comment in comments:
        if comment.get("public", True):  # Only include public comments
            author_name = ""
            if comment.get("author_id"):
                # You might want to add author mapping here
                author_name = f"Comment {comment['id']}"
            
            body = comment.get("body", "").strip()
            if body:
                comment_texts.append(f"{author_name}: {body}")
    
    return "\n".join(comment_texts)

def process_item(
    data: Dict[str, Any],
    keys: List[str],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a single JSON item using the provided configuration."""
    # Extract data using different matching criteria
    result_text = []
    
    # First process title and description
    priority_fields = ['Title', 'Description', "Ticket"]
    for field in priority_fields:
        if field in config["composite_text_fields"]:
            column = config["composite_text_fields"][field]
            value = get_nested_value(data, column) if '.' in column else data.get(column)
            if value:
                result_text.append(f"{field.lower()}: {clean_text(value)}")
    
    # Then process other composite text fields
    for field, column in config["composite_text_fields"].items():
        if field not in priority_fields:  # Skip title and description as they're already processed
            value = get_nested_value(data, column) if '.' in column else data.get(column)
            if value:
                result_text.append(f"{field.lower()}: {clean_text(value)}")
    
    # Process comments last
    # if "comments" in data and isinstance(data["comments"], list):
    #    comments_text = extract_comments_text(data["comments"])
    #    if comments_text:
    #        result_text.append(f"comments: {comments_text}")
    
    # Process composite text fields
    #for field, column in config["composite_text_fields"].items():
    #    value = get_nested_value(data, column) if '.' in column else data.get(column)
    #    if value:
    #        result_text.append(f"{field.lower()}: {clean_text(value)}")
    
    # Process prefix fields
    for target_field, prefix in config["prefix_fields"].items():
        if '.' in prefix:
            # Handle nested prefix fields
            parent, child = prefix.split('.', 1)
            if parent in data and isinstance(data[parent], list):
                values = [item.get(child, '') for item in data[parent] if item.get(child)]
                if values:
                    result_text.append(f"{target_field.lower()}: {' '.join(map(clean_text, values))}")
        else:
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
    
    # Build metadata
    metadata = {}
    for field, column in config["metadata_fields"].items():
        if '.' in column:
            # Handle nested fields
            parent, child = column.split('.', 1)
            if parent in data and isinstance(data[parent], dict):
                value = data[parent].get(child, "")
            else:
                value = ""
        else:
            value = data.get(column, "")
        metadata[field.lower()] = clean_text(value)

    # Process list fields
    for field in config["list_fields"]:
        if field in data:
            values = parse_list(data[field])
            if values:
                metadata[field.lower()] = ", ".join(values)
    
    # Add source URL
    metadata_unique_id = config["metadata_unique_id"]
    metadata["source"] = config["zendesk_url"] + metadata[metadata_unique_id] + ".json"
    
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
        "metadata_unique_id": config.get("TicketUrl", "metadata_unique_id", fallback=""),
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
