import random
import shutil
from pathlib import Path
from PIL import Image, ImageFilter

RAW_DIR = Path("raw_cubicasa")
OUT_DIR = Path("floorplan_errors_dataset")

# Class IDs in cubicasa5k-2: 0 = door, 1 = wall, 2 = window
DOOR_ID, WALL_ID, WINDOW_ID = 0, 1, 2

def yolo_to_pixels(xc, yc, bw, bh, img_w, img_h):
    x1 = int((xc - bw / 2) * img_w)
    y1 = int((yc - bh / 2) * img_h)
    x2 = int((xc + bw / 2) * img_w)
    y2 = int((yc + bh / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

def pixels_to_yolo(class_id, x1, y1, x2, y2, img_w, img_h):
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

for split_in, split_out in [("train", "train"), ("valid", "val")]:
    in_imgs = RAW_DIR / split_in / "images"
    in_lbls = RAW_DIR / split_in / "labels"
    
    out_imgs = OUT_DIR / split_out / "images"
    out_lbls = OUT_DIR / split_out / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    if not in_imgs.exists():
        continue

    for img_path in in_imgs.glob("*.*"):
        lbl_path = in_lbls / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        walls, doors, windows = [], [], []
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(float(parts[0]))
            box = yolo_to_pixels(*map(float, parts[1:]), w, h)
            # Skip tiny boxes smaller than 12x12 pixels
            if (box[2] - box[0]) < 12 or (box[3] - box[1]) < 12:
                continue
            if cid == WALL_ID:
                walls.append(box)
            elif cid == DOOR_ID:
                doors.append(box)
            elif cid == WINDOW_ID:
                windows.append(box)

        new_error_labels = []

        # 1. Inject Wall Gap Errors (erase center 60% of up to 2 walls)
        for wx1, wy1, wx2, wy2 in random.sample(walls, min(2, len(walls))):
            bw, bh = wx2 - wx1, wy2 - wy1
            if bw >= bh:  # Horizontal wall
                gx1 = wx1 + int(bw * 0.2)
                gx2 = wx2 - int(bw * 0.2)
                gy1, gy2 = max(0, wy1 - 2), min(h, wy2 + 2)
            else:         # Vertical wall
                gx1, gx2 = max(0, wx1 - 2), min(w, wx2 + 2)
                gy1 = wy1 + int(bh * 0.2)
                gy2 = wy2 - int(bh * 0.2)

            if gx2 > gx1 + 4 and gy2 > gy1 + 4:
                img.paste(Image.new("RGB", (gx2 - gx1, gy2 - gy1), (255, 255, 255)), (gx1, gy1))
                new_error_labels.append(pixels_to_yolo(0, gx1, gy1, gx2, gy2, w, h))

        # 2. Inject Door/Window Clash (paste a real door onto a window or another door)
        targets = windows + doors
        if doors and len(targets) >= 2:
            src_door = random.choice(doors)
            door_crop = img.crop(src_door)
            tx1, ty1, tx2, ty2 = random.choice([t for t in targets if t != src_door])
            
            # Paste door at target's top-left corner
            px2 = min(w, tx1 + door_crop.width)
            py2 = min(h, ty1 + door_crop.height)
            cropped_to_fit = door_crop.crop((0, 0, px2 - tx1, py2 - ty1))
            img.paste(cropped_to_fit, (tx1, ty1))
            
            # Label spans the combined clash bounding box
            cx1, cy1 = min(tx1, tx1), min(ty1, ty1)
            cx2, cy2 = max(tx2, px2), max(ty2, py2)
            new_error_labels.append(pixels_to_yolo(1, cx1, cy1, cx2, cy2, w, h))

        # 3. Inject Unclear/Blurred Annotation over a door or window
        if targets:
            bx1, by1, bx2, by2 = random.choice(targets)
            blurred = img.crop((bx1, by1, bx2, by2)).filter(ImageFilter.GaussianBlur(radius=4))
            img.paste(blurred, (bx1, by1))
            new_error_labels.append(pixels_to_yolo(2, bx1, by1, bx2, by2, w, h))

        # Save image and its new error labels (empty .txt is valid in YOLO for clean negatives)
        img.save(out_imgs / img_path.name)
        (out_lbls / f"{img_path.stem}.txt").write_text("\n".join(new_error_labels))

# Zip the final folder for BlueprintAI
shutil.make_archive("floorplan_errors_dataset", "zip", OUT_DIR)
print("Created floorplan_errors_dataset.zip!")