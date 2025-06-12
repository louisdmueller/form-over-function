import json
import os
from utils import parse_args

args = parse_args()
print(args.input_dir)
print(os.path.dirname(args.input_dir))
data_directory = os.path.dirname(args.input_dir)
files = [f for f in os.listdir(data_directory) if f.endswith('.json')]
newest_file = max(
    files,
    key=lambda f: os.path.getmtime(os.path.join(data_directory, f))
)

print(f"Newest file found: {os.path.join(data_directory, newest_file)}")

with open(os.path.join(data_directory, newest_file), 'r') as f:
    data = json.load(f)

indices = [int(indice) for indice in data.keys() if indice != "metadata"]
max_index = max(indices)

with open(args.data_1_path, 'r') as f:
    length_reference_file = len(f.readlines()) - 1 # index starts at 0

if max_index == length_reference_file:
    print(f"Data processed completely. Last index in judge file: {max_index}, expected: {length_reference_file}")
    exit(0)
else:
    print(f"Data not processed completely. Last index in judge file: {max_index}, expected: {length_reference_file}")
    exit(1)
