# import re
# import argparse
# import pandas as pd

# def extract_shot_positions(inp_file_path, output_csv_path):
#     with open(inp_file_path, 'r') as f:
#         lines = f.readlines()

#     inside_assembly = False
#     inside_instance = False
#     current_instance = None
#     positions = []

#     for line in lines:
#         line_strip = line.strip()
#         line_lower = line_strip.lower()

#         if line_lower.startswith("*assembly"):
#             inside_assembly = True

#         elif inside_assembly and line_lower.startswith("*instance"):
#             match = re.search(r'name=([\w\d_]+)', line_strip)
#             if match:
#                 inst_name = match.group(1)
#                 inside_instance = inst_name.startswith("SHOT")
#                 current_instance = inst_name if inside_instance else None

#         elif inside_assembly and line_lower.startswith("*end instance"):
#             inside_instance = False
#             current_instance = None

#         elif inside_instance and current_instance:
#             try:
#                 coords = [float(val) for val in line_strip.split(',')[:3]]
#                 if len(coords) == 3:
#                     positions.append((current_instance, *coords))
#             except ValueError:
#                 continue  # Skip malformed lines

#         elif line_lower.startswith("*end assembly"):
#             inside_assembly = False

#     # Save to CSV
#     df = pd.DataFrame(positions, columns=["Instance", "X", "Y", "Z"])
#     df.to_csv(output_csv_path, index=False)
#     print(f"Extracted {len(df)} SHOT particle positions to {output_csv_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Extract SHOT particle positions from Abaqus INP file.")
#     parser.add_argument("inp_file", help="Path to the .inp file")
#     parser.add_argument("output_csv", help="Path to output CSV file")
#     args = parser.parse_args()

#     extract_shot_positions(args.inp_file, args.output_csv)

import os
import re
import csv

def extract_shot_positions(inp_file_path, output_csv_path):
    with open(inp_file_path, 'r') as file:
        lines = file.readlines()

    shot_data = {}
    current_instance = None
    capture_coords = False

    for line in lines:
        line = line.strip()
        if line.startswith("*Instance, name=SHOT"):
            match = re.search(r'name=(SHOT\d+)', line)
            if match:
                current_instance = match.group(1)
                shot_data[current_instance] = None
                capture_coords = True
        elif line.startswith("*End Instance"):
            capture_coords = False
        elif capture_coords and current_instance:
            coords = list(map(float, line.split(",")[:3]))
            if shot_data[current_instance] is None:
                shot_data[current_instance] = coords

    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Instance", "X", "Y", "Z"])
        for instance, coords in shot_data.items():
            if coords:
                writer.writerow([instance] + coords)