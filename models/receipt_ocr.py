"""
models/receipt_ocr.py

OCR-based receipt verification pipeline.
Extracts amount, vendor, date from uploaded receipt image/PDF
and compares against the submitted expense claim.

Dependencies: pytesseract, pillow, rapidfuzz, pdf2image
"""

import re
import pytesseract
from PIL import Image
from rapidfuzz import fuzz
from datetime import datetime

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# ── Main verifier class ────────────────────────────────────────────────────────

class ReceiptVerifier:

    def verify(self, image_path: str, claimed: dict) -> dict:
        """
        Full pipeline: load → OCR → extract fields → compare vs claim.

        claimed dict keys:
          amount      (float)   — what employee claimed
          vendor      (str)     — vendor name employee entered
          date        (str)     — YYYY-MM-DD

        Returns:
          {
            verified:         bool,     overall pass/fail
            confidence:       float,    0–100
            extracted:        dict,     what OCR found
            mismatches:       list,     list of mismatch descriptions
            ocr_text:         str,      raw OCR output (for debugging)
          }
        """
        # Step 1: load image
        image = self._load_image(image_path)
        if image is None:
            return self._error_result("Could not load receipt image")

        # Step 2: run OCR
        ocr_text = pytesseract.image_to_string(image, config="--psm 6")

        # Step 3: extract fields from OCR text
        extracted = {
            "amount": self._extract_amount(ocr_text),
            "vendor": self._extract_vendor(ocr_text),
            "date":   self._extract_date(ocr_text),
        }

        # Step 4: compare against claim
        mismatches = []
        scores = []

        # Compare amount
        amount_result = self._compare_amount(claimed.get("amount"), extracted["amount"])
        scores.append(amount_result["score"])
        if not amount_result["match"]:
            mismatches.append(amount_result["message"])

        # Compare vendor
        vendor_result = self._compare_vendor(claimed.get("vendor", ""), extracted["vendor"])
        scores.append(vendor_result["score"])
        if not vendor_result["match"]:
            mismatches.append(vendor_result["message"])

        # Compare date
        date_result = self._compare_date(claimed.get("date", ""), extracted["date"])
        scores.append(date_result["score"])
        if not date_result["match"]:
            mismatches.append(date_result["message"])

        confidence = round(sum(scores) / len(scores), 1)
        verified   = len(mismatches) == 0 and confidence >= 70.0

        return {
            "verified":   verified,
            "confidence": confidence,
            "extracted":  extracted,
            "mismatches": mismatches,
            "ocr_text":   ocr_text.strip(),
        }

    # ── Image loading ──────────────────────────────────────────────────────────

    def _load_image(self, path: str):
        """Load image or convert first page of PDF to image."""
        try:
            if path.lower().endswith(".pdf"):
                if not PDF_SUPPORT:
                    return None
                pages = convert_from_path(path, dpi=200, first_page=1, last_page=1)
                return pages[0] if pages else None
            else:
                return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[ReceiptVerifier] Image load error: {e}")
            return None

    # ── Field extraction ───────────────────────────────────────────────────────

    def _extract_amount(self, text: str) -> float | None:
        """
        Find the largest currency amount in the OCR text.
        Handles formats: ₹1,200.00  |  Rs.1200  |  1200.50  |  TOTAL 1,200
        """
        patterns = [
            r"(?:total|amount|grand\s*total|net\s*total|bill\s*amount)[^\d]*(\d[\d,]*\.?\d{0,2})",
            r"(?:₹|rs\.?|inr)\s*(\d[\d,]*\.?\d{0,2})",
            r"(\d[\d,]*\.\d{2})",        # any decimal amount
        ]

        candidates = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for m in matches:
                try:
                    val = float(m.replace(",", ""))
                    if val > 0:
                        candidates.append(val)
                except ValueError:
                    pass

        return max(candidates) if candidates else None

    def _extract_vendor(self, text: str) -> str | None:
        """
        Extract vendor name from first 3 lines of receipt
        (business name is usually at the top).
        """
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if not lines:
            return None

        # Take first non-empty line with at least 3 chars
        for line in lines[:4]:
            if len(line) >= 3 and not re.match(r"^\d", line):
                # Clean up common OCR noise
                cleaned = re.sub(r"[^a-zA-Z0-9\s&\-\.]", "", line).strip()
                if cleaned:
                    return cleaned
        return None

    def _extract_date(self, text: str) -> str | None:
        """
        Extract date from OCR text.
        Handles: DD/MM/YYYY  DD-MM-YYYY  YYYY-MM-DD  DD Mon YYYY
        """
        patterns = [
            r"\b(\d{2})[/\-](\d{2})[/\-](\d{4})\b",   # DD/MM/YYYY or DD-MM-YYYY
            r"\b(\d{4})[/\-](\d{2})[/\-](\d{2})\b",   # YYYY-MM-DD
            r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})\b",
        ]

        months = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                  "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}

        for pattern in patterns:
            m = re.search(pattern, text.lower())
            if m:
                try:
                    g = m.groups()
                    if len(g[0]) == 4:            # YYYY-MM-DD
                        return f"{g[0]}-{g[1]}-{g[2]}"
                    elif g[1].isalpha():           # DD Mon YYYY
                        month = months.get(g[1][:3], 1)
                        return f"{g[2]}-{month:02d}-{int(g[0]):02d}"
                    else:                          # DD/MM/YYYY
                        return f"{g[2]}-{g[1]}-{g[0]}"
                except Exception:
                    continue
        return None

    # ── Comparison helpers ─────────────────────────────────────────────────────

    def _compare_amount(self, claimed, extracted) -> dict:
        if extracted is None:
            return {"match": False, "score": 40.0,
                    "message": "Could not extract amount from receipt"}
        if claimed is None:
            return {"match": True, "score": 70.0, "message": ""}

        diff_pct = abs(claimed - extracted) / max(claimed, 1) * 100

        if diff_pct <= 1:                      # within 1% — exact match
            return {"match": True,  "score": 100.0, "message": ""}
        elif diff_pct <= 5:                    # within 5% — likely taxes/rounding
            return {"match": True,  "score": 85.0,  "message": ""}
        elif diff_pct <= 15:                   # suspicious
            return {"match": False, "score": 50.0,
                    "message": f"Amount mismatch: claimed ₹{claimed:.0f}, receipt shows ₹{extracted:.0f}"}
        else:                                  # clear mismatch
            return {"match": False, "score": 10.0,
                    "message": f"Severe amount mismatch: claimed ₹{claimed:.0f}, receipt shows ₹{extracted:.0f}"}

    def _compare_vendor(self, claimed: str, extracted: str | None) -> dict:
        if extracted is None:
            return {"match": False, "score": 50.0,
                    "message": "Could not extract vendor name from receipt"}
        if not claimed:
            return {"match": True, "score": 70.0, "message": ""}

        similarity = fuzz.partial_ratio(claimed.lower(), extracted.lower())

        if similarity >= 80:
            return {"match": True,  "score": 100.0, "message": ""}
        elif similarity >= 60:
            return {"match": True,  "score": 75.0,  "message": ""}
        else:
            return {"match": False, "score": 20.0,
                    "message": f"Vendor mismatch: claimed '{claimed}', receipt shows '{extracted}'"}

    def _compare_date(self, claimed: str, extracted: str | None) -> dict:
        if extracted is None:
            return {"match": True, "score": 60.0, "message": ""}  # date is optional
        if not claimed:
            return {"match": True, "score": 70.0, "message": ""}

        try:
            d_claimed   = datetime.strptime(claimed,   "%Y-%m-%d")
            d_extracted = datetime.strptime(extracted, "%Y-%m-%d")
            diff_days   = abs((d_claimed - d_extracted).days)

            if diff_days == 0:
                return {"match": True,  "score": 100.0, "message": ""}
            elif diff_days <= 1:
                return {"match": True,  "score": 85.0,  "message": ""}
            elif diff_days <= 3:
                return {"match": False, "score": 50.0,
                        "message": f"Date mismatch: claimed {claimed}, receipt shows {extracted}"}
            else:
                return {"match": False, "score": 10.0,
                        "message": f"Severe date mismatch: claimed {claimed}, receipt shows {extracted}"}
        except ValueError:
            return {"match": True, "score": 60.0, "message": ""}

    @staticmethod
    def _error_result(message: str) -> dict:
        return {
            "verified":   False,
            "confidence": 0.0,
            "extracted":  {"amount": None, "vendor": None, "date": None},
            "mismatches": [message],
            "ocr_text":   "",
        }
