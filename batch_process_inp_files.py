import os
import argparse
from extract_shot_positions import extract_shot_positions

def batch_process_inp_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.inp'):
            inp_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            out_name = f"{base_name}_shot_positions.csv"
            out_path = os.path.join(output_dir, out_name)

            print(f"Processing: {filename}")
            extract_shot_positions(inp_path, out_path)

    print(f"All .inp files in {input_dir} processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch extract SHOT particle positions from .inp files.")
    parser.add_argument("input_dir", help="Directory containing .inp files")
    parser.add_argument("output_dir", help="Directory to save extracted .csv files")
    args = parser.parse_args()

    batch_process_inp_files(args.input_dir, args.output_dir)