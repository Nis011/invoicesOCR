from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import fitz
import base64
import requests
import json
import re
import time
import io
import os

app = FastAPI(title="Invoice OCR API", version="1.2.0")

# Overridable via env var so this works both locally (Ollama on the same
# machine) and in Docker (Ollama running in a separate container, reached
# by its service name instead of localhost).
OLLAMA_URL = os.environ.get(
    "OLLAMA_URL", "http://localhost:11434/api/generate"
)

# Overridable via env var so switching models (e.g. for comparing 3b vs
# 7b) doesn't require editing code or rebuilding the Docker image.
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:3b")

# Number of pages sent to Ollama at the same time. Raise/lower based on
# available VRAM/CPU - too high and requests will just queue on Ollama's side.
MAX_CONCURRENT_PAGES = 3

# Zoom factor for PDF page -> image rendering. Higher = sharper image but
# more visual tokens sent to the model = slower. Lower this first if you
# need more speed and accuracy holds up.
IMAGE_ZOOM_FACTOR = 2

# Vision models occasionally return slightly malformed JSON or hit a
# transient network error on a single try. Retrying the same page once
# before giving up meaningfully improves reliability at low cost.
MAX_ATTEMPTS_PER_PAGE = 2

PROMPT = """You are an invoice data extraction assistant.
Extract the following fields from this invoice image and return ONLY a valid JSON object, nothing else. No explanation, no markdown, just the raw JSON.

Rules for reading numbers correctly:
- Numbers on these invoices may use a SPACE as a thousands separator and a
  COMMA as a decimal separator. For example "101 150,00" means one hundred
  one thousand one hundred fifty (101150.00) - NOT 101.15. Always read the
  full number, including any space-separated group before the comma.
- Only extract values that are literally printed on the invoice. Do NOT
  calculate, derive, or multiply values yourself - for example, do not
  compute "total" as quantity x unit_price if that exact total is not
  printed. If a field isn't clearly present, use 0 or an empty string
  rather than guessing or computing one.
- If a number that looks like a quantity is actually part of the item
  description text (not a separate quantity column), do not extract it as
  quantity.

Rules for picking the right total (read carefully - this is the most
common mistake, so follow these steps in order):
- Step 1: Look across the ENTIRE document for a small tax breakdown table
  with column headers like "Taux", "H.T", "TVA", "TTC" (it may be near the
  bottom, in small print, separate from a bigger standalone total).
- Step 2: If that breakdown table exists, montant_ht, tva, and montant_ttc
  MUST all three come from that table's row of numbers - not from
  anywhere else on the page.
- Step 3: A standalone "TOTAL" line printed elsewhere on the document
  (often bigger/more prominent than the breakdown table) is virtually
  always equal to the TTC (tax-included) amount, never the H.T amount.
  Completely ignore that standalone "TOTAL" line when deciding
  montant_ht - it is a decoy for this field, even though it looks like
  the most obvious number on the page.
- Step 4: montant_ht will always be SMALLER than montant_ttc (since HT
  excludes tax). If your chosen montant_ht is equal to (or larger than)
  montant_ttc, you picked the wrong number - go back to the breakdown
  table and re-read the H.T column specifically.
- Step 5: Only if no breakdown table exists anywhere on the document
  should you fall back to using a standalone "TOTAL" line, and in that
  case treat it as montant_ttc, not montant_ht.
- "currency" must be an actual currency code or symbol (e.g. MAD, DH, EUR,
  USD) - never an accounting term like HT, TVA, or TTC.
- Some invoices show a tax RATE (e.g. "TVA 20%") separately from the tax
  AMOUNT (e.g. "20 230,00 MAD") in a breakdown table. "tva" must always be
  the tax AMOUNT in currency - never the percentage rate.

Rules for the number format in your JSON output:
- montant_ht, tva, montant_ttc, quantity, unit_price, and total must be
  plain numbers only (e.g. 800, 101150.00) - never include a currency
  symbol, currency code, or a percent sign in these fields, even if one
  is printed next to the number on the invoice.

{
  "invoice_number": "",
  "date": "",
  "supplier": "",
  "client": "",
  "currency": "",
  "montant_ht": 0,
  "tva": 0,
  "montant_ttc": 0,
  "line_items": [
    {
      "description": "",
      "quantity": 0,
      "unit_price": 0,
      "total": 0
    }
  ]
}"""


def nanoseconds_to_seconds(value):
    """
    Ollama returns duration values in nanoseconds.
    Convert them to seconds for easier reading.
    """
    if value is None:
        return None

    return round(value / 1_000_000_000, 3)


def to_float(value):
    """Parse a number that may use a comma decimal separator, spaces as a
    thousands separator, or have currency units/percent signs stuck to it
    (e.g. "101 150,00", "800.00 DH", "20%") - despite the prompt asking
    for plain numbers, the model doesn't always comply, so this strips
    anything that isn't a digit, comma, period, or minus sign first."""
    text = re.sub(r"[^\d,.\-]", "", str(value))
    text = text.replace(",", ".")

    # If stripping left more than one period (e.g. a thousands-dot format
    # like "1.234.567,89" after the comma->period swap above), only the
    # last one is the real decimal separator.
    if text.count(".") > 1:
        head, _, tail = text.rpartition(".")
        text = head.replace(".", "") + "." + tail

    return float(text)


def validate_invoice(data):
    errors = []

    required_fields = ["montant_ht", "tva", "montant_ttc"]
    missing = [f for f in required_fields if data.get(f) in (None, "")]

    if missing:
        errors.append(
            f"Cannot validate totals - required field(s) missing: "
            f"{', '.join(missing)}. This document may not be a valid "
            f"invoice, or the model could not find these fields."
        )
        return errors

    try:
        ht = to_float(data["montant_ht"])
        tva = to_float(data["tva"])
        ttc = to_float(data["montant_ttc"])

        data["montant_ht"] = ht
        data["tva"] = tva
        data["montant_ttc"] = ttc

        if abs((ht + tva) - ttc) > 1:
            errors.append(
                f"Hallucination detected: "
                f"HT({ht}) + TVA({tva}) = {ht + tva} ≠ TTC({ttc})"
            )

        line_items = data.get("line_items") or []
        line_items_total = 0.0

        for item in line_items:
            try:
                item_total = to_float(item.get("total", 0))
                item["total"] = item_total
                line_items_total += item_total
            except (TypeError, ValueError):
                pass

        # Wider tolerance than the HT+TVA=TTC check: rounding on each
        # individual line item accumulates as more items are summed, so a
        # flat 1-unit tolerance produces false positives on invoices with
        # many line items.
        line_items_tolerance = max(1.0, abs(ht) * 0.01)

        if line_items and abs(line_items_total - ht) > line_items_tolerance:
            errors.append(
                f"Line items total ({line_items_total}) does not match "
                f"montant_ht ({ht})"
            )

    except Exception as e:
        errors.append(f"Validation error: {str(e)}")

    return errors


def merge_invoice_group(group):
    """Combine one or more same-invoice pages into a single invoice record."""
    pages = group["pages"]
    datas = [p["data"] for p in pages]

    def first_nonempty(field):
        for d in datas:
            value = d.get(field)
            if value not in (None, "", 0):
                return value
        return datas[0].get(field) if datas else None

    def last_nonempty(field):
        for d in reversed(datas):
            value = d.get(field)
            if value not in (None, "", 0):
                return value
        return datas[-1].get(field) if datas else None

    line_items = []
    for d in datas:
        line_items.extend(d.get("line_items") or [])

    merged_data = {
        "invoice_number": first_nonempty("invoice_number"),
        "supplier": first_nonempty("supplier"),
        "client": first_nonempty("client"),
        "date": first_nonempty("date"),
        "currency": first_nonempty("currency"),
        # Totals usually only appear once, on the last page of a
        # multi-page invoice, so prefer the last non-empty value.
        "montant_ht": last_nonempty("montant_ht"),
        "tva": last_nonempty("tva"),
        "montant_ttc": last_nonempty("montant_ttc"),
        "line_items": line_items,
    }

    errors = validate_invoice(merged_data)

    return {
        "source_pages": group["source_pages"],
        "status": "success",
        "data": merged_data,
        "validation": {
            "passed": len(errors) == 0,
            "errors": errors
        }
    }


def group_pages_into_invoices(results):
    """Group consecutive pages that belong to the same invoice.

    Pages are grouped by invoice_number: consecutive pages sharing the same
    (non-empty) invoice_number are treated as one multi-page invoice. A page
    with a blank/missing invoice_number is assumed to continue whatever
    invoice came before it, since only the first page of an invoice usually
    repeats the header. This is a heuristic, not ground truth - if a PDF has
    back-to-back invoices that both fail to extract an invoice_number, they
    will be merged incorrectly.
    """
    groups = []
    current_group = None
    current_key = None

    for page_result in results:
        if page_result is None or page_result.get("status") != "success":
            groups.append({
                "source_pages": [page_result["page"]] if page_result else [],
                "status": "error",
                "pages": [page_result] if page_result else []
            })
            current_group = None
            current_key = None
            continue

        invoice_number = str(
            page_result["data"].get("invoice_number") or ""
        ).strip().lower()

        starts_new_invoice = (
            current_group is None
            or (invoice_number and invoice_number != current_key)
        )

        if starts_new_invoice:
            current_key = invoice_number or current_key
            current_group = {
                "source_pages": [],
                "status": "success",
                "pages": []
            }
            groups.append(current_group)
        elif invoice_number:
            current_key = invoice_number

        current_group["source_pages"].append(page_result["page"])
        current_group["pages"].append(page_result)

    return [
        group if group["status"] == "error" else merge_invoice_group(group)
        for group in groups
    ]


def render_page_to_image(page, page_num):
    """PDF page -> PNG -> base64, plus timing/size metrics."""
    image_conversion_start = time.perf_counter()

    pix = page.get_pixmap(
        matrix=fitz.Matrix(IMAGE_ZOOM_FACTOR, IMAGE_ZOOM_FACTOR),
        alpha=False
    )

    image_bytes = pix.tobytes("png")

    image_conversion_duration = time.perf_counter() - image_conversion_start

    base64_start = time.perf_counter()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    base64_duration = time.perf_counter() - base64_start

    return {
        "page_num": page_num,
        "image_b64": image_b64,
        "width": pix.width,
        "height": pix.height,
        "size_mb": round(len(image_bytes) / (1024 * 1024), 3),
        "image_conversion_duration": image_conversion_duration,
        "base64_duration": base64_duration,
    }


def render_image_upload(image_bytes, page_num=0):
    """Standalone image upload (png/jpg/webp/...) -> normalized PNG -> base64.

    Always re-encodes to PNG regardless of the source format, so formats
    that don't decode reliably everywhere (webp in particular) get
    converted before Ollama ever sees them. Pillow detects the format from
    the file content itself, not the filename/extension.
    """
    image_conversion_start = time.perf_counter()

    image = Image.open(io.BytesIO(image_bytes))
    image.load()

    if image.mode != "RGB":
        image = image.convert("RGB")

    png_buffer = io.BytesIO()
    image.save(png_buffer, format="PNG")
    png_bytes = png_buffer.getvalue()

    image_conversion_duration = time.perf_counter() - image_conversion_start

    base64_start = time.perf_counter()

    image_b64 = base64.b64encode(png_bytes).decode("utf-8")

    base64_duration = time.perf_counter() - base64_start

    return {
        "page_num": page_num,
        "image_b64": image_b64,
        "width": image.width,
        "height": image.height,
        "size_mb": round(len(png_bytes) / (1024 * 1024), 3),
        "image_conversion_duration": image_conversion_duration,
        "base64_duration": base64_duration,
    }


def call_ollama(image_b64):
    """Single HTTP call to Ollama. Raises RequestException on failure."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": PROMPT,
            "images": [image_b64],
            "stream": False,
            "keep_alive": "30m",
        },
        timeout=600
    )

    response.raise_for_status()

    return response.json()


def build_ollama_metrics(ollama_data):
    """Extract + derive Ollama's native performance metrics for one call."""
    metrics = {
        "total_duration_seconds": nanoseconds_to_seconds(
            ollama_data.get("total_duration")
        ),
        "load_duration_seconds": nanoseconds_to_seconds(
            ollama_data.get("load_duration")
        ),
        "prompt_eval_duration_seconds": nanoseconds_to_seconds(
            ollama_data.get("prompt_eval_duration")
        ),
        "eval_duration_seconds": nanoseconds_to_seconds(
            ollama_data.get("eval_duration")
        ),
        "prompt_eval_count": ollama_data.get("prompt_eval_count"),
        "eval_count": ollama_data.get("eval_count")
    }

    prompt_eval_duration = metrics["prompt_eval_duration_seconds"]
    eval_duration = metrics["eval_duration_seconds"]
    prompt_eval_count = metrics["prompt_eval_count"]
    eval_count = metrics["eval_count"]

    metrics["prompt_tokens_per_second"] = (
        round(prompt_eval_count / prompt_eval_duration, 2)
        if prompt_eval_duration and prompt_eval_count is not None
        and prompt_eval_duration > 0
        else None
    )

    metrics["output_tokens_per_second"] = (
        round(eval_count / eval_duration, 2)
        if eval_duration and eval_count is not None and eval_duration > 0
        else None
    )

    return metrics


def process_page(page_info):
    """Send one page's image to Ollama, parse + validate the result.

    Runs inside a worker thread so multiple pages can be in flight at once.
    Retries up to MAX_ATTEMPTS_PER_PAGE times on network failure or bad
    JSON, and never raises - a failure on this page always comes back as
    an error result for just this page, so one bad page can't take down
    extraction for the rest of the document.
    """
    page_num = page_info["page_num"]
    page_start = time.perf_counter()

    last_error_message = "Unknown error"
    last_raw_result = ""

    for attempt in range(1, MAX_ATTEMPTS_PER_PAGE + 1):
        raw_result = ""

        try:
            ollama_request_start = time.perf_counter()
            ollama_data = call_ollama(page_info["image_b64"])
            ollama_http_duration = time.perf_counter() - ollama_request_start

            ollama_metrics = build_ollama_metrics(ollama_data)

            parsing_start = time.perf_counter()

            # .get(..., "") only covers a *missing* key - fall back on the
            # `or` too in case Ollama ever returns "response": null, which
            # would otherwise crash re.sub() on a None value.
            raw_result = ollama_data.get("response") or ""

            clean = re.sub(
                r"```(?:json)?|```",
                "",
                raw_result,
                flags=re.IGNORECASE
            ).strip()

            data = json.loads(clean)

            parsing_duration = time.perf_counter() - parsing_start

            validation_start = time.perf_counter()

            errors = validate_invoice(data)

            validation_duration = time.perf_counter() - validation_start

            page_total_duration = time.perf_counter() - page_start

            page_metrics = {
                "image_conversion_seconds": round(
                    page_info["image_conversion_duration"], 3
                ),
                "base64_encoding_seconds": round(
                    page_info["base64_duration"], 3
                ),
                "ollama_http_request_seconds": round(ollama_http_duration, 3),
                "json_parsing_seconds": round(parsing_duration, 3),
                "validation_seconds": round(validation_duration, 3),
                "page_total_seconds": round(page_total_duration, 3),
                "attempts": attempt,
                "image_width": page_info["width"],
                "image_height": page_info["height"],
                "image_size_mb": page_info["size_mb"],
                "ollama": ollama_metrics
            }

            print(f"\n--- Page {page_num + 1} benchmark ---")
            print(json.dumps(page_metrics, indent=2, ensure_ascii=False))

            return page_num, {
                "page": page_num + 1,
                "status": "success",
                "data": data,
                "validation": {
                    "passed": len(errors) == 0,
                    "errors": errors
                },
                "performance": page_metrics
            }

        except requests.exceptions.RequestException as e:
            last_error_message = f"Ollama request failed: {e}"
            last_raw_result = raw_result
        except json.JSONDecodeError as e:
            last_error_message = f"JSON parsing failed: {e}"
            last_raw_result = raw_result
        except Exception as e:
            # Catch-all so a page can NEVER take the whole request down -
            # any unexpected shape/error just becomes this page's error.
            last_error_message = f"Unexpected error while processing page: {e}"
            last_raw_result = raw_result

    page_total_duration = time.perf_counter() - page_start

    return page_num, {
        "page": page_num + 1,
        "status": "error",
        "message": last_error_message,
        "raw_model_response": last_raw_result,
        "attempts": MAX_ATTEMPTS_PER_PAGE,
        "performance": {
            "page_total_seconds": round(page_total_duration, 3)
        }
    }


@app.get("/")
def root():
    return {
        "message": "Invoice OCR API is running!",
        "model": MODEL
    }


@app.post("/extract-invoice")
async def extract_invoice(file: UploadFile = File(...)):
    request_start = time.perf_counter()
    doc = None

    try:
        # -----------------------------
        # 1. Read uploaded file
        # -----------------------------
        upload_start = time.perf_counter()

        contents = await file.read()

        upload_duration = time.perf_counter() - upload_start

        if not contents:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Uploaded file is empty."}
            )

        # -----------------------------
        # 2. Detect file type from content, not filename - the filename
        #    from a client upload can't be trusted to be accurate.
        # -----------------------------
        pdf_open_duration = 0.0
        is_pdf = contents[:5] == b"%PDF-"

        if is_pdf:
            pdf_open_start = time.perf_counter()

            try:
                doc = fitz.open(stream=contents, filetype="pdf")
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": f"Could not open PDF: {e}"
                    }
                )

            pdf_open_duration = time.perf_counter() - pdf_open_start

            total_pages = len(doc)

            if total_pages == 0:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "PDF has no pages."}
                )

            # -----------------------------
            # 3. Render every page to an image (sequential: PyMuPDF page
            #    rendering is not safe to run concurrently across threads).
            #    A single corrupted page must not take down the rest of
            #    the document, so render failures are isolated here too.
            # -----------------------------
            results = [None] * total_pages
            page_images = []

            for page_num in range(total_pages):
                try:
                    page_images.append(
                        render_page_to_image(doc[page_num], page_num)
                    )
                except Exception as e:
                    results[page_num] = {
                        "page": page_num + 1,
                        "status": "error",
                        "message": f"Failed to render page: {e}"
                    }

        else:
            try:
                page_images = [render_image_upload(contents, page_num=0)]
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": (
                            "Could not read file as a PDF or a supported "
                            f"image format (png, jpg, webp, bmp, tiff, "
                            f"...): {e}"
                        )
                    }
                )
            results = [None]

        # -----------------------------
        # 4. Send pages to Ollama concurrently and collect results
        # -----------------------------
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PAGES) as executor:
            futures = [
                executor.submit(process_page, page_info)
                for page_info in page_images
            ]

            for future in as_completed(futures):
                page_num, page_result = future.result()
                results[page_num] = page_result

        grouped_invoices = group_pages_into_invoices(results)

        total_request_duration = time.perf_counter() - request_start

        request_metrics = {
            "file_read_seconds": round(upload_duration, 3),
            "pdf_open_seconds": round(pdf_open_duration, 3),
            "request_total_seconds": round(total_request_duration, 3)
        }

        print("\n--- Complete request benchmark ---")
        print(json.dumps(request_metrics, indent=2, ensure_ascii=False))

        pages_failed = sum(1 for r in results if r["status"] != "success")

        return JSONResponse(content={
            "status": "success" if pages_failed == 0 else "partial_success",
            "total_pages": len(results),
            "pages_failed": pages_failed,
            "total_invoices": len(grouped_invoices),
            "model": MODEL,
            "performance": request_metrics,
            "pages": results,
            "invoices": grouped_invoices
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

    finally:
        if doc is not None:
            doc.close()
