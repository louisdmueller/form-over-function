import argparse

from utils import get_df_from_file, parse_args

parser = argparse.ArgumentParser(description="Argument parser for our research project.")

def main() -> None:
    args = parse_args()
    data_df = get_df_from_file(args.data_path)

if __name__ == "__main__":
    main()