import os
import json
from configparser import ConfigParser
from typing import Any, Dict, List

import click
import pandas as pd


def preprocess_text(text: Any) -> str:
    """Clean and standardize text content."""
    if pd.isna(text):
        return ""
    return str(text).strip().replace("\n", " ").replace("\r", " ")


def get_list_from_field(value: Any) -> List[str]:
    """Safely convert a field value to a list of strings."""
    if pd.isna(value):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if pd.notna(item)]
    return [str(value)]


def extract_prefixed_columns(
    row: pd.Series, df_columns: List[str], prefix: str
) -> List[str]:
    """Extract and concatenate non-empty values from columns starting with a prefix."""
    return [
        str(row[col])
        for col in df_columns
        if col.startswith(prefix) and pd.notna(row[col])
    ]


def extract_containing_columns(
    row: pd.Series, df_columns: List[str], substring: str
) -> List[str]:
    """Extract and concatenate non-empty values from columns containing a specific substring."""
    return [
        str(row[col])
        for col in df_columns
        if substring in col.lower() and pd.notna(row[col])
    ]


def extract_composite_text(
    row: pd.Series, composite_text_fields: Dict[str, str]
) -> str:
    """Extract and format fields for composite text."""
    parts = [
        f"{field.lower()}: {preprocess_text(row.get(column, ''))}"
        for field, column in composite_text_fields.items()
    ]
    return "\n".join(parts)


def extract_metadata(row: pd.Series, metadata_fields: Dict[str, str]) -> Dict[str, str]:
    """Extract fields for metadata."""
    metadata = {
        field.lower(): preprocess_text(row.get(column, ""))
        for field, column in metadata_fields.items()
    }
    return metadata


def process_row(
    row: pd.Series,
    df_columns: List[str],
    prefix_fields: Dict[str, str],
    substring_fields: Dict[str, str],
    list_fields: List[str],
    composite_text_fields: Dict[str, str],
    metadata_fields: Dict[str, str],
    jira_url: str,
    metadata_field: str,
) -> Dict[str, Any]:
    """Process a single row using the provided configuration."""
    # Extract prefixed fields
    prefixed_data = {
        target_field.lower(): extract_prefixed_columns(row, df_columns, prefix)
        for target_field, prefix in prefix_fields.items()
    }

    # Extract substring fields
    substring_data = {
        target_field.lower(): extract_containing_columns(row, df_columns, substring)
        for target_field, substring in substring_fields.items()
    }

    # Extract list fields
    list_data = {
        field.lower(): get_list_from_field(row.get(field, "")) for field in list_fields
    }

    # Extract composite text
    composite_text = extract_composite_text(row, composite_text_fields)

    # Add prefixed and substring fields to composite text
    for field, values in prefixed_data.items():
        if values:
            composite_text += f"\n{field}: {' '.join(values)}"
    for field, values in substring_data.items():
        if values:
            composite_text += f"\n{field}: {' '.join(values)}"

    # Add list fields to composite text
    for field, values in list_data.items():
        if values:
            composite_text += f"\n{field.lower()}: {', '.join(values)}"

    # Extract metadata
    metadata = extract_metadata(row, metadata_fields)

    # Add source
    metadata["source"] = jira_url + metadata[metadata_field]

    return {"text": composite_text, "metadata": metadata}


def prepare_data_for_embedding(
    df: pd.DataFrame,
    prefix_fields: Dict[str, str],
    substring_fields: Dict[str, str],
    list_fields: List[str],
    composite_text_fields: Dict[str, str],
    metadata_fields: Dict[str, str],
    jira_url: str,
    metadata_field: str,
):
    """Prepare documents for embedding using the provided configuration."""
    documents = []

    for _, row in df.iterrows():
        processed_data = process_row(
            row=row,
            df_columns=df.columns,
            prefix_fields=prefix_fields,
            substring_fields=substring_fields,
            list_fields=list_fields,
            composite_text_fields=composite_text_fields,
            metadata_fields=metadata_fields,
            jira_url=jira_url,
            metadata_field=metadata_field,
        )
        documents.append(processed_data)

    return documents


def load_config(config_file):
    config = ConfigParser()
    config.read(config_file)

    # Parse configuration sections into variables
    prefix_fields = dict(config["PrefixFields"]) if "PrefixFields" in config else {}
    substring_fields = (
        dict(config["SubstringFields"]) if "SubstringFields" in config else {}
    )

    # Handle ListFields specially since we need to split the string
    list_fields = []
    if "ListFields" in config and "fields" in config["ListFields"]:
        fields_str = config["ListFields"]["fields"]
        if fields_str:
            list_fields = [f.strip() for f in fields_str.split(",")]

    composite_text_fields = (
        dict(config["CompositeTextFields"]) if "CompositeTextFields" in config else {}
    )
    metadata_fields = (
        dict(config["MetadataFields"]) if "MetadataFields" in config else {}
    )

    return {
        "prefix_fields": prefix_fields,
        "substring_fields": substring_fields,
        "list_fields": list_fields,
        "composite_text_fields": composite_text_fields,
        "metadata_fields": metadata_fields,
        "jira_url": config["TicketUrl"]["jira_url"],
        "metadata_field": config["TicketUrl"]["metadata_field"],
    }


def save_to_json_files(data: List[Dict], output_dir: str):
    """Save each dictionary in the list as a separate JSON file in the specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, item in enumerate(data):
        output_file = os.path.join(output_dir, f"item_{i}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False)


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("config_file", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def process_data(input_file: str, config_file: str, output_dir: str):
    """
    Process data from INPUT_FILE using CONFIG_FILE and save to OUTPUT_FILE.

    The input file should be a CSV file containing the data to process.
    The config file should be an INI file with the processing configuration.
    The output file will be saved in JSON format.
    """
    # Load configuration
    click.echo(f"Loading configuration from {config_file}")
    config = load_config(config_file)

    # Get configuration sections into variables
    prefix_fields = config["prefix_fields"]
    substring_fields = config["substring_fields"]
    list_fields = config["list_fields"]
    composite_text_fields = config["composite_text_fields"]
    metadata_fields = config["metadata_fields"]

    # Load DataFrame
    click.echo(f"Reading data from {input_file}")
    df = pd.read_csv(input_file)

    # Prepare embedding data
    click.echo("Processing data...")
    embedding_data = prepare_data_for_embedding(
        df=df,
        prefix_fields=prefix_fields,
        substring_fields=substring_fields,
        list_fields=list_fields,
        composite_text_fields=composite_text_fields,
        metadata_fields=metadata_fields,
        jira_url=config["jira_url"],
        metadata_field=config["metadata_field"],
    )

    # Save to file
    click.echo(f"Saving processed data to {output_dir}")
    save_to_json_files(embedding_data, output_dir)
    click.echo(f"Processed {len(embedding_data)} documents")


if __name__ == "__main__":
    process_data()
