import os
import json

def merge_json_files(input_dir: str, output_file: str) -> None:
    """
    Merges all JSON files in the specified directory into a single JSON file.
    If there there are multiple JSON files covering the same data,
    the data from the last file will be used.

    Args:
        input_dir (str): Directory containing the JSON files to merge.
        output_file (str): Path to the output JSON file.
    """

    merged_data = {}

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            if filename.startswith("merge"):
                continue
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r") as f:
                data = json.load(f)
                # Update the merged_data with the current file's data
                merged_data.update(data)

    # sort the merged data by keys, but keep "metadata" at the top
    # do this by popping "metadata" out, sorting the rest, and then adding it back
    metadata = merged_data.pop("metadata")
    merged_data = dict(
        sorted(
            merged_data.items(), 
            key=lambda item: int(item[0]) if item[0].isdigit() else item[0]
        )
    )
    merged_data = {"metadata": metadata, **merged_data}

    # Write the merged data to the output file
    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=4)

if __name__ == "__main__":
    subdirs = [directory for directory in os.listdir("data/chen-et-al/") if os.path.isdir(os.path.join("data/chen-et-al", directory))]
    for directory in subdirs:
        input_directory = os.path.join("data/chen-et-al", directory)
        output_file_path = os.path.join(input_directory, "merged_data.json")

        merge_json_files(input_directory, output_file_path)
        print(f"Merged JSON files from {input_directory} into {output_file_path}")