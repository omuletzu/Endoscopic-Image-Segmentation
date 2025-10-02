import os
import shutil
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

def delete_endo_mask(path):
    for subdir, _, files in os.walk(path):
        for file in files:
            if "endo_mask" in file:
                try:
                    bmp_path = os.path.join(subdir, file)
                    os.remove(bmp_path)
                    #print(f"Deleted BMP: {bmp_path}")
                except Exception as e:
                    print(f"Error deleting {bmp_path}: {e}")

def delete_color_mask(path):
    for subdir, _, files in os.walk(path):
        for file in files:
            if "color_mask" in file:
                try:
                    bmp_path = os.path.join(subdir, file)
                    os.remove(bmp_path)
                    #print(f"Deleted BMP: {bmp_path}")
                except Exception as e:
                    print(f"Error deleting {bmp_path}: {e}")

def delete_bmps(path):
    for subdir, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".bmp"):
                try:
                    bmp_path = os.path.join(subdir, file)
                    os.remove(bmp_path)
                    print(f"Deleted BMP: {bmp_path}")
                except Exception as e:
                    print(f"Error deleting {bmp_path}: {e}")


def move_to_path(from_path, dest_path):
    from_path = Path(from_path)
    dest_path = Path(dest_path)

    for subdir, _, files in os.walk(from_path):
        if os.path.basename(subdir) == "training":
            continue
        for file in files:
            if file.lower().endswith(".png"):
                from_file_path = Path(subdir) / file
                to_file_path = dest_path / file

                shutil.move(str(from_file_path), str(to_file_path))
                print(f"Moved ${from_file_path}")

def convert_png_bmp(path):
    for subdir, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".png"):
                png_path = os.path.join(subdir, file)
                bmp_path = os.path.join(subdir, file.replace(".png", ".bmp"))

                try:
                    png_image = Image.open(png_path)
                    png_image.save(bmp_path, "BMP")
                    os.remove(png_path)
                    #print(f"Converted and deleted: {png_path}")
                except Exception as e:
                    print(f"Error converting {png_path}: {e}")


def split_train_test(path, dest_path):
    path = Path(path)
    dest_path = Path(dest_path)

    all_images = sorted(os.listdir(path))
    original_names = [img for img in all_images if img.endswith('_endo.bmp')]

    mask_names = [img.replace('_endo.bmp', '_endo_watershed_mask.bmp') for img in original_names]

    train_original, test_original, train_mask, test_mask = train_test_split(
        original_names, mask_names, test_size=0.2, random_state=42
    )

    for ind in range(len(test_original)):
        from_test_original_path = path / test_original[ind]
        from_test_mask_path = path / test_mask[ind]
        to_test_original_path = dest_path / test_original[ind]
        to_test_mask_path = dest_path / test_mask[ind]

        try:
            shutil.move(str(from_test_original_path), str(to_test_original_path))
            shutil.move(str(from_test_mask_path), str(to_test_mask_path))
        except Exception as e:
            print(f"Error moving {from_test_original_path} or {from_test_mask_path}: {e}")


input_folder = r"D:\FACULTATE\AN3\Sem2\PI\Project\dataset\training"
dest_folder = r"D:\FACULTATE\AN3\Sem2\PI\Project\dataset\testing"
