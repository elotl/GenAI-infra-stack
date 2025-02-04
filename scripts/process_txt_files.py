import json
import os

import click

@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def process_data(input_dir: str, output_dir: str):
    """
    Process data from INPUT_FILE and save to OUTPUT_FILE.
    """
    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
            data = {
                "text": text_content,
                "metadata": {
                    "source": filename
                }
            }
            
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=4)



if __name__ == "__main__":
    process_data()
