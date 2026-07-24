import re
import cv2
import numpy as np
from PIL import Image
import easyocr

# ── Initialize OCR reader (cached, loads once) ─────────────────────────────────
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        print("Loading EasyOCR model (first run only)...")
        _reader = easyocr.Reader(['en'], gpu=True)
    return _reader


# ── Regex patterns for Indian ID formats ───────────────────────────────────────
PAN_PATTERN     = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b')
AADHAAR_PATTERN = re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b')
DOB_PATTERN     = re.compile(r'\b\d{2}[/\-]\d{2}[/\-]\d{4}\b')


# ── Verhoeff algorithm for Aadhaar checksum validation ─────────────────────────
_VERHOEFF_TABLE_D = [
    [0,1,2,3,4,5,6,7,8,9], [1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6], [3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8], [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2], [7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4], [9,8,7,6,5,4,3,2,1,0]
]
_VERHOEFF_TABLE_P = [
    [0,1,2,3,4,5,6,7,8,9], [1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2], [8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0], [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5], [7,0,4,6,9,1,3,2,5,8]
]
_VERHOEFF_TABLE_INV = [0,4,3,2,1,5,6,7,8,9]


def verhoeff_validate(number: str) -> bool:
    """Validates a number string using the Verhoeff checksum algorithm (used by Aadhaar)."""
    number = number.replace(" ", "")
    if not number.isdigit():
        return False

    c = 0
    digits = [int(d) for d in reversed(number)]
    for i, digit in enumerate(digits):
        c = _VERHOEFF_TABLE_D[c][_VERHOEFF_TABLE_P[i % 8][digit]]
    return c == 0


# ── Field presence checks ───────────────────────────────────────────────────────
PAN_REQUIRED_FIELDS = {
    "income tax department": "Income Tax Department header",
    "permanent account number": "PAN card title",
    "name": "Name field",
    "father": "Father's Name field",
}

AADHAAR_REQUIRED_FIELDS = {
    "government of india": "Government of India header",
    "unique identification": "UIDAI header",
    "aadhaar": "Aadhaar branding",
}


def extract_text(image_path: str) -> list:
    """Runs OCR and returns list of detected text strings."""
    reader = get_reader()
    results = reader.readtext(image_path, detail=0)
    return [r.lower().strip() for r in results]


def validate_pan(image_path: str) -> dict:
    """Layer 1 structural validation for PAN card."""
    checks = {}
    detected_text = extract_text(image_path)
    full_text = " ".join(detected_text)

    # Check 1: Required fields present
    missing_fields = []
    for keyword, label in PAN_REQUIRED_FIELDS.items():
        if keyword not in full_text:
            missing_fields.append(label)
    checks["fields_present"] = {
        "passed": len(missing_fields) == 0,
        "missing": missing_fields
    }

    # Check 2: PAN number format
    pan_matches = PAN_PATTERN.findall(full_text.upper())
    checks["pan_format"] = {
        "passed": len(pan_matches) > 0,
        "detected": pan_matches[0] if pan_matches else None
    }

    # Check 3: DOB format
    dob_matches = DOB_PATTERN.findall(full_text)
    checks["dob_format"] = {
        "passed": len(dob_matches) > 0,
        "detected": dob_matches[0] if dob_matches else None
    }

    # Check 4: Layout — text should be distributed across image (not clustered)
    checks["layout_distribution"] = check_text_layout(image_path)

    all_passed = all(c["passed"] for c in checks.values())

    return {
        "document_type": "PAN",
        "overall_pass": all_passed,
        "checks": checks
    }


def validate_aadhaar(image_path: str) -> dict:
    """Layer 1 structural validation for Aadhaar card."""
    checks = {}
    detected_text = extract_text(image_path)
    full_text = " ".join(detected_text)

    # Check 1: Required fields present
    missing_fields = []
    for keyword, label in AADHAAR_REQUIRED_FIELDS.items():
        if keyword not in full_text:
            missing_fields.append(label)
    checks["fields_present"] = {
        "passed": len(missing_fields) == 0,
        "missing": missing_fields
    }

    # Check 2: Aadhaar number format + checksum
    aadhaar_matches = AADHAAR_PATTERN.findall(full_text)
    valid_checksum = False
    detected_number = None
    if aadhaar_matches:
        detected_number = aadhaar_matches[0]
        valid_checksum = verhoeff_validate(detected_number)

    checks["aadhaar_format"] = {
        "passed": len(aadhaar_matches) > 0,
        "detected": detected_number
    }
    checks["checksum_valid"] = {
        "passed": valid_checksum,
        "note": "Verhoeff checksum validation"
    }

    # Check 3: Layout
    checks["layout_distribution"] = check_text_layout(image_path)

    all_passed = all(c["passed"] for c in checks.values())

    return {
        "document_type": "Aadhaar",
        "overall_pass": all_passed,
        "checks": checks
    }


def check_text_layout(image_path: str) -> dict:
    """
    Uses OpenCV to check if text regions are distributed naturally
    across the document (not all clustered in one corner, which
    could indicate a poorly composited fake).
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"passed": False, "note": "Could not load image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Edge detection to find text-dense regions
    edges = cv2.Canny(gray, 50, 150)

    # Divide into quadrants, check edge density in each
    quadrants = {
        "top_left":     edges[:h//2, :w//2],
        "top_right":    edges[:h//2, w//2:],
        "bottom_left":  edges[h//2:, :w//2],
        "bottom_right": edges[h//2:, w//2:],
    }

    densities = {k: float(np.sum(v > 0)) / v.size for k, v in quadrants.items()}
    active_quadrants = sum(1 for d in densities.values() if d > 0.01)

    return {
        "passed": active_quadrants >= 2,  # text should span at least 2 quadrants
        "densities": {k: round(v, 4) for k, v in densities.items()},
        "note": f"{active_quadrants}/4 quadrants have text activity"
    }


def validate_document(image_path: str, document_type: str = "auto") -> dict:
    """
    Main entry point. document_type: 'pan', 'aadhaar', or 'auto'
    """
    if document_type == "auto":
        # Quick heuristic — check for keywords to decide doc type
        text = " ".join(extract_text(image_path))
        if "permanent account" in text or "pan" in text:
            document_type = "pan"
        elif "aadhaar" in text or "unique identification" in text:
            document_type = "aadhaar"
        else:
            return {
                "document_type": "unknown",
                "overall_pass": False,
                "checks": {"type_detection": {"passed": False, "note": "Could not identify document type"}}
            }

    if document_type == "pan":
        return validate_pan(image_path)
    elif document_type == "aadhaar":
        return validate_aadhaar(image_path)
    else:
        raise ValueError(f"Unknown document_type: {document_type}")


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python validator.py <image_path> [pan|aadhaar|auto]")
        sys.exit(1)

    img_path = sys.argv[1]
    doc_type = sys.argv[2] if len(sys.argv) > 2 else "auto"

    result = validate_document(img_path, doc_type)
    print(json.dumps(result, indent=2))