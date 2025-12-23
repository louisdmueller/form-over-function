import json
import os
import sys

# -> export PYTHONPATH=src/
from utils.utils import parse_args


def check_if_all_data_processed(input_dir=None, data_1_path=None):
    """
    Check if all data has been processed by comparing the last index in the newest JSON file
    with the expected length from the reference file.
    
    Args:
        input_dir (str, optional): Input directory path. If None, uses parsed args.
        data_1_path (str, optional): Path to reference data file. If None, uses parsed args.
    
    Returns:
        bool: True if all data is processed, False otherwise.
        int: The last index found in the newest JSON file.
    """
    if input_dir is None or data_1_path is None:
        args = parse_args()
        input_dir = input_dir or args.input_dir
        data_1_path = data_1_path or args.data_1_path
        input_dir = os.path.dirname(input_dir)
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    newest_file = max(
        files,
        key=lambda f: os.path.getmtime(os.path.join(input_dir, f))
    )

    print(f"Newest file found: {os.path.join(input_dir, newest_file)}", file=sys.stderr)

    with open(os.path.join(input_dir, newest_file), 'r') as f:
        data = json.load(f)

    indices = [int(indice) for indice in data.keys() if indice != "metadata"]
    max_index = max(indices)

    with open(data_1_path, 'r') as f:
        length_reference_file = len(f.readlines()) - 1  # index starts at 0

    if max_index == length_reference_file:
        print(f"Data processed completely. Last index in judge file: {max_index}, expected: {length_reference_file}", file=sys.stderr)
        return True, max_index
    else:
        print(f"Data not processed completely. Last index in judge file: {max_index}, expected: {length_reference_file}", file=sys.stderr)
        return False, max_index


if __name__ == "__main__":
    result, _ = check_if_all_data_processed()
    exit(0 if result else 1)