from PIL import Image, ImageDraw, ImageFont
import os
import json

INPUT_DIR  = "data/id_samples"
OUTPUT_DIR = "data/id_synthetic"
os.makedirs(f"{OUTPUT_DIR}/authentic", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/tampered",  exist_ok=True)

# ── Tamper config ─────────────────────────────────────────────────────────────
# For each image define: which region to cover + what fake text to write
# bbox = (x1, y1, x2, y2) — the region to inpaint (cover with background color)
# text = replacement text to write
# text_xy = where to start writing the new text
# font_size = approximate size to match original
TAMPER_CONFIG = {
    "pan_rahul.jpeg": {
        # Image: 1024x662 — DOB "30/01/1997" is at bottom-left area
        "description": "Change DOB from 30/01/1997 to 12/08/1985",
        "bbox":     (60, 595, 350, 640),
        "fill":     (160, 195, 220),
        "text":     "12/08/1985",
        "text_xy":  (62, 598),
        "font_size": 28,
        "text_color": (0, 0, 0)
    },
    "pan_applicant.jpeg": {
        # Image: 631x397 — PAN number "ABCDE1234F" is center-top area
        "description": "Change PAN number from ABCDE1234F to ZXKQT9921M",
        "bbox":     (200, 115, 500, 155),
        "fill":     (185, 210, 228),
        "text":     "ZXKQT9921M",
        "text_xy":  (202, 118),
        "font_size": 26,
        "text_color": (0, 0, 0)
    },
    "pan_template.jpeg": {
        # Image: 974x634 — "YOUR NAME HERE" is upper-left area
        "description": "Change name from YOUR NAME HERE to SURESH KUMAR",
        "bbox":     (20, 190, 550, 240),
        "fill":     (175, 195, 225),
        "text":     "SURESH KUMAR",
        "text_xy":  (22, 193),
        "font_size": 32,
        "text_color": (0, 0, 0)
    },
    "aadhar_specimen.jpeg": {
        # Image: 1024x328 — "XXXX XXXX XXXX" is bottom-center of left panel
        "description": "Change Aadhaar number XXXX XXXX XXXX to 8821 4493 7710",
        "bbox":     (50, 255, 480, 300),
        "fill":     (255, 240, 235),
        "text":     "8821  4493  7710",
        "text_xy":  (52, 258),
        "font_size": 30,
        "text_color": (180, 0, 0)
    }
}
def tamper_image(img_path: str, config: dict, save_path: str):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Step 1: Cover original text with background color patch
    draw.rectangle(config["bbox"], fill=config["fill"])

    # Step 2: Write replacement text
    try:
        # Try to load Arial or fallback to default
        font = ImageFont.truetype("arial.ttf", config["font_size"])
    except:
        try:
            font = ImageFont.truetype(
                "C:/Windows/Fonts/arial.ttf", config["font_size"]
            )
        except:
            font = ImageFont.load_default()

    draw.text(
        config["text_xy"],
        config["text"],
        fill=config["text_color"],
        font=font
    )

    # Step 3: Save with slight JPEG compression to simulate re-encoding
    # This creates ELA-detectable artifacts in the tampered region
    img.save(save_path, format="JPEG", quality=82)
    print(f"  Tampered: {save_path}")


def main():
    log = []

    for filename, config in TAMPER_CONFIG.items():
        src = os.path.join(INPUT_DIR, filename)

        if not os.path.exists(src):
            print(f"MISSING: {src} — skipping")
            continue

        print(f"\nProcessing: {filename}")
        print(f"  Change: {config['description']}")

        # Save authentic copy
        auth_path = os.path.join(OUTPUT_DIR, "authentic", filename)
        img = Image.open(src).convert("RGB")
        img.save(auth_path, format="JPEG", quality=95)
        print(f"  Authentic: {auth_path}")

        # Save tampered copy
        tamp_name = filename.replace(".", "_tampered.")
        tamp_path = os.path.join(OUTPUT_DIR, "tampered", tamp_name)
        tamper_image(src, config, tamp_path)

        log.append({
            "authentic": auth_path,
            "tampered": tamp_path,
            "change": config["description"]
        })

    # Save log
    with open(f"{OUTPUT_DIR}/tamper_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nDone — {len(log)} pairs created")
    print(f"Authentic: {OUTPUT_DIR}/authentic/")
    print(f"Tampered:  {OUTPUT_DIR}/tampered/")
    print(f"Log:       {OUTPUT_DIR}/tamper_log.json")


if __name__ == "__main__":
    main()