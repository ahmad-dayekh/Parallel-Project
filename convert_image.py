import cv2

def gen_text_from_image(in_path, out_path):
    """
    Converts an image to its pixel-value text representation and saves it to a file.
    """
    img = cv2.imread(in_path)
    if img is None:
        print(f"Error: Could not read image at {in_path}")
        return False
    if img.shape != (224, 224, 3):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    with open(out_path, "w") as out:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    out.write(str(img[i, j, k]) + " ")
            out.write("\n")
    print(f"Converted {in_path} to {out_path}")
    return True

# Input paths
image_file = "tv.jpg"  # image path
output_text_file = "tv.txt"  # output text file path

gen_text_from_image(image_file, output_text_file)