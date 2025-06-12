import os
import json
from utils import parse_args

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
    metadata_exists = False
    if "metadata" in merged_data:
        metadata = merged_data.pop("metadata")
        metadata_exists = True
    merged_data = dict(
        sorted(
            merged_data.items(), 
            key=lambda item: int(item[0]) if item[0].isdigit() else item[0]
        )
    )
    if metadata_exists:
        merged_data = {"metadata": metadata, **merged_data}

    # Write the merged data to the output file
    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    if args.merge_path is None:
        print("Setting default merge path to 'data/chen-et-al/'")
        print("You can change this by using the --merge_path argument.")
        args.merge_path = "data/chen-et-al/"

    # subdirs = [directory for directory in os.listdir(args.merge_path) if os.path.isdir(os.path.join(args.merge_path, directory))]
    # for directory in subdirs:
        # input_directory = os.path.join(args.merge_path, directory)
        # output_file_path = os.path.join(input_directory, "merged_data.json")

        # if os.path.exists(output_file_path):
        #     print(f"Output file {output_file_path} already exists. Skipping merge.")
        #     continue

        # print(input_directory)
        # print(output_file_path)

        # merge_json_files(input_directory, output_file_path)
        # print(f"Merged JSON files from {input_directory} into {output_file_path}")

    input_directory = args.merge_path
    output_file_path = os.path.join(input_directory, "merged_data.json")
    if os.path.exists(output_file_path):
        print(f"Output file {output_file_path} already exists. Skipping merge.")
        exit(0)
    merge_json_files(input_directory, output_file_path)
    print(f"Merged JSON files from {input_directory} into {output_file_path}")