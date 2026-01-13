import csv
import sys
import os

def center_coordinates(file_path):
    """
    Reads coordinates from a CSV file, centers them around their average,
    and saves the result to a new CSV file in the same directory.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Read coordinates
    coordinates = []
    try:
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                # Skip empty rows
                if not row:
                    continue
                try:
                    # Assuming the first two columns are x and y
                    x = float(row[0])
                    y = float(row[1])
                    coordinates.append((x, y))
                except (ValueError, IndexError):
                    # Skip rows that don't contain valid numbers (headers, etc.)
                    continue
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not coordinates:
        print("No valid coordinate data found in the file.")
        return

    # Calculate the average (centroid)
    n = len(coordinates)
    sum_x = sum(x for x, y in coordinates)
    sum_y = sum(y for x, y in coordinates)
    avg_x = sum_x / n
    avg_y = sum_y / n

    print(f"Calculated center (average): X={avg_x}, Y={avg_y}")

    # Recalculate coordinates relative to the average
    centered_coordinates = [(x - avg_x, y - avg_y) for x, y in coordinates]

    # Generate output file path
    dir_name, base_name = os.path.dirname(file_path), os.path.basename(file_path)
    file_name_no_ext, ext = os.path.splitext(base_name)
    new_file_name = f"{file_name_no_ext}_centered{ext}"
    output_path = os.path.join(dir_name, new_file_name)

    # Write to the new CSV file
    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # You can uncomment the following line if you want to add a header
            # writer.writerow(['x', 'y'])
            writer.writerows(centered_coordinates)
        print(f"Centered coordinates saved to: {output_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # Prompt user if no argument provided
        input_path = input("Please enter the relative path to the CSV file: ").strip()
    
    center_coordinates(input_path)
