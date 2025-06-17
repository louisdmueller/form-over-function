import json
import sys
from pathlib import Path


def transform_json_structure(input_data):
    """
    Transform JSON structure according to the specified requirements:
    1. Replace question content with AAE version from question_aae
    2. Replace answer1 and answer2 content with AAE versions from answer1_aae and answer2_aae
    3. Set model_name to "gpt-4"
    4. Add metadata with translation_model_name "gpt-4.1" and generation_model_name "gpt-4"
    5. Remove question_aae, answer1_aae and answer2_aae after using them
    """

    transformed_data = json.loads(json.dumps(input_data))

    transformed_data["question"] = transformed_data["question_aae"]
    del transformed_data["question_aae"]

    if "answers" in transformed_data:
        answers = transformed_data["answers"]

        if isinstance(answers["answer1"], dict) and "answer" in answers["answer1"]:
            answers["answer1"]["answer"] = (
                answers["answer1_aae"]["answer"]
                if isinstance(answers["answer1_aae"], dict)
                else answers["answer1_aae"]
            )
        else:
            answers["answer1"] = answers["answer1_aae"]

        if isinstance(answers["answer2"], dict) and "answer" in answers["answer2"]:
            answers["answer2"]["answer"] = (
                answers["answer2_aae"]["answer"]
                if isinstance(answers["answer2_aae"], dict)
                else answers["answer2_aae"]
            )
        else:
            answers["answer2"] = answers["answer2_aae"]

        if "answer1_aae" in answers:
            del answers["answer1_aae"]
        if "answer2_aae" in answers:
            del answers["answer2_aae"]

    transformed_data["model_name"] = "gpt-4"

    transformed_data["metadata"] = {
        "translation_model_name": "gpt-4.1",
        "generation_model_name": "gpt-4",
    }

    return transformed_data


def process_json_file(input_file_path, output_file_path):
    """
    Process a JSON file and transform its structure
    Handles both single JSON objects and JSONL (JSON Lines) format

    Args:
        input_file_path (str): Path to the input JSON file
        output_file_path (str): Path to the output JSON file (optional)
    """

    try:

        # Read the input file
        with open(input_file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()

        lines = content.split("\n")
        is_jsonl = len(lines) > 1

        if is_jsonl:
            transformed_objects = []
            processed_count = 0

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    data = json.loads(line)
                    transformed_data = transform_json_structure(data)
                    transformed_objects.append(transformed_data)
                    processed_count += 1
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON on line {line_num}: {e}")
                    continue

            # Write the transformed data to the output file (JSONL format)
            with open(output_file_path, "w", encoding="utf-8") as file:
                for obj in transformed_objects:
                    json.dump(obj, file, ensure_ascii=False)
                    file.write("\n")

            print(f" Successfully transformed JSONL file:")
            print(f"   Input:  {input_file_path}")
            print(f"   Output: {output_file_path}")
            print(f"   Processed: {processed_count} JSON objects")

            return transformed_objects

        else:
            data = json.loads(content)
            transformed_data = transform_json_structure(data)

            with open(output_file_path, "w", encoding="utf-8") as file:
                json.dump(transformed_data, file, indent=2, ensure_ascii=False)

            print(f"Successfully transformed JSON file:")
            print(f"   Input:  {input_file_path}")
            print(f"   Output: {output_file_path}")

            return transformed_data

    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{input_file_path}': {e}")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def process_json_string(json_string):
    """
    Process a JSON string and return the transformed structure

    Args:
        json_string (str): JSON string to transform

    Returns:
        dict: Transformed JSON data
    """
    try:
        data = json.loads(json_string)
        return transform_json_structure(data)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        return None


def main():
    """
    Main function to handle command line arguments
    """
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python script.py <input_file.json> [output_file.json]")
        print("\nExample:")
        print("  python script.py data.json")
        print("  python script.py data.json transformed_data.json")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    process_json_file(input_file, output_file)


if __name__ == "__main__":
    # Example usage with sample data
    sample_input = {
        "question_id": "Analyzing_0",
        "question": "How can you demonstrate the process of historical analysis using a historical event?",
        "question_aae": "Hey, I'm kinda stuck on this question and was wonderin' if you could help me out. The question is: How you show the process of doin' historical analysis usin' a historical event?",
        "prompt": "How can you demonstrate the process of historical analysis using a historical event? Your answer MUST NOT contain rich text. Your answer should be within 150 words.",
        "temperature": 0.5,
        "model_id": "gpt-4",
        "level": "Analyzing",
        "answers": {
            "answer1": {
                "answer": "Historical analysis involves examining past events...",
                "answer_id": "aSrKYwua",
            },
            "answer2": {
                "answer": "Historical analysis involves the evaluation of data...",
                "answer_id": "DKYok3xi",
            },
            "answer1_aae": {
                "answer": "We bouta break down how Archduke Franz Ferdinand got got...",
                "answer_id": "ssgyx6NX",
            },
            "answer2_aae": {
                "answer": "Aight, let's break down what went down when the Titanic sunk...",
                "answer_id": "tjnBOcnV",
            },
        },
    }

    expected_output = {
        "question_id": "Analyzing_0",
        "question": "Hey, I'm kinda stuck on this question and was wonderin' if you could help me out. The question is: How you show the process of doin' historical analysis usin' a historical event?",
        "prompt": "How can you demonstrate the process of historical analysis using a historical event? Your answer MUST NOT contain rich text. Your answer should be within 150 words.",
        "temperature": 0.5,
        "model_id": "gpt-4",
        "level": "Analyzing",
        "answers": {
            "answer1": {
                "answer": "We bouta break down how Archduke Franz Ferdinand got got...",
                "answer_id": "ssgyx6NX",
            },
            "answer2": {
                "answer": "Aight, let's break down what went down when the Titanic sunk...",
                "answer_id": "tjnBOcnV",
            },
        },
        "model_name": "gpt-4",
        "metadata": {
            "translation_model_name": "gpt-4.1",
            "generation_model_name": "gpt-4",
        },
    }

    # If run directly (not as command line), show example
    if len(sys.argv) == 1:
        print("🔄 Example transformation:")
        print("\n📥 Input structure:")
        print(json.dumps(sample_input, indent=2)[:200] + "...")

        transformed = transform_json_structure(sample_input)
        print("\n📤 Output structure:")
        print(json.dumps(transformed, indent=2)[:300] + "...")

        print("\n" + "=" * 50)
        print("To use with files, run:")
        print("python script.py <input_file.json> [output_file.json]")
    else:
        main()
