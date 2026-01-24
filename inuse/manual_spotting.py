import cv2
import argparse
import numpy as np
import csv
import os

# A3 paper dimensions in mm (Landscape)
A3_WIDTH_MM = 150
A3_HEIGHT_MM = 150

# Button configuration
BUTTON_X, BUTTON_Y = 20, 20
BUTTON_W, BUTTON_H = 100, 40
BUTTON_COLOR = (220, 220, 220)
BUTTON_TEXT_COLOR = (0, 0, 0)
BUTTON_TEXT = "Save"

def draw_button(img, text=BUTTON_TEXT, color=BUTTON_COLOR):
    """Draws the save button on the image."""
    cv2.rectangle(img, (BUTTON_X, BUTTON_Y), (BUTTON_X + BUTTON_W, BUTTON_Y + BUTTON_H), color, -1)
    cv2.rectangle(img, (BUTTON_X, BUTTON_Y), (BUTTON_X + BUTTON_W, BUTTON_Y + BUTTON_H), (100, 100, 100), 1)
    
    # Center text in button
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = BUTTON_X + (BUTTON_W - text_w) // 2
    text_y = BUTTON_Y + (BUTTON_H + text_h) // 2
    
    cv2.putText(img, text, (text_x, text_y), font, font_scale, BUTTON_TEXT_COLOR, thickness)

def save_points_to_csv(filepath, points):
    """Saves the list of points to a CSV file."""
    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write data: x, y
            for p in points:
                writer.writerow([f"{p[0]:.4f}", f"{p[1]:.4f}"])
        print(f"\n[System] Successfully saved {len(points)} points to: {filepath}")
    except Exception as e:
        print(f"\n[Error] Failed to save file: {e}")

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img = param['img']
        points = param['points']
        image_path = param['image_path']
        
        # Check if the click is inside the Save button
        if BUTTON_X <= x <= BUTTON_X + BUTTON_W and BUTTON_Y <= y <= BUTTON_Y + BUTTON_H:
            # Generate CSV filename based on image filename
            dir_name = os.path.dirname(image_path)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            csv_filename = f"{base_name}_points.csv"
            csv_path = os.path.join(dir_name, csv_filename)
            
            save_points_to_csv(csv_path, points)
            
            # Visual feedback (flash green)
            draw_button(img, "Saved!", (150, 255, 150))
            cv2.imshow("Image", img)
            cv2.waitKey(300) # Pause briefly
            draw_button(img) # Restore button
            cv2.imshow("Image", img)
            return

        # Normal point marking logic
        height, width = img.shape[:2]
        
        # Center of the image in pixels
        center_x = width / 2
        center_y = height / 2
        
        # Calculate mm per pixel
        mm_per_pixel_x = A3_WIDTH_MM / width
        mm_per_pixel_y = A3_HEIGHT_MM / height
        
        # Convert pixel coordinate to physical coordinate (mm)
        # Origin at center. X increases to the right. Y increases upwards.
        coord_x = (x - center_x) * mm_per_pixel_x
        coord_y = (center_y - y) * mm_per_pixel_y # Invert Y axis so up is positive
        
        print(f"Pixel: ({x}, {y}) -> Coord: ({coord_x:.2f}, {coord_y:.2f}) mm")
        
        # Store the point
        points.append((coord_x, coord_y))
        
        # Draw a small circle where the user clicked
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        # Redraw button to ensure it stays on top if points are near it (unlikely but good practice)
        draw_button(img)
        cv2.imshow("Image", img)

def main():
    parser = argparse.ArgumentParser(description="Manual spotting tool for A3 calibration board.")
    parser.add_argument("image_path", help="Path to the image file.")
    args = parser.parse_args()
    
    image_path = args.image_path
    if not os.path.exists(image_path):
        print(f"Error: File not found '{image_path}'")
        return

    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not open or find the image '{image_path}'")
        return

    print("Click on the image to mark points.")
    print("Click the 'Save' button (top-left) to save points to a CSV file.")
    print(f"Assuming image covers A3 Landscape: {A3_WIDTH_MM}mm x {A3_HEIGHT_MM}mm")
    print("Origin (0,0) is at the center of the image.")
    print("Press 'q' or ESC to exit.")

    # Draw the save button initially
    draw_button(img)

    cv2.imshow("Image", img)
    
    # List to store recorded points
    points = []
    
    # Pass image, points list, and image path in a dictionary
    params = {
        'img': img, 
        'points': points,
        'image_path': image_path
    }
    cv2.setMouseCallback("Image", click_event, params)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # 'q' or ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
