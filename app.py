"""FastAPI web wrapper around predict.py, serves the classifier as a website.

Run it with:  uvicorn app:app --reload
Then open:    http://127.0.0.1:8000
"""
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from predict import FoodClassifier, open_image

# Vercel rejects request bodies over 4.5MB at the edge, but that's before this function is
# invoked. anything bigger never reaches the check below and the caller gets an
# opaque FUNCTION_PAYLOAD_TOO_LARGE instead. 4MB keeps our own clearer error the
# one people see in the band where we can still catch it.
# Mirrored in static/index.html (MAX_UPLOAD_BYTES). keep the two in sync!
MAX_UPLOAD_BYTES = 4 * 1024 * 1024

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """runs once on startup, before any request is served.

    this is THE reason we refactored predict.py -- loading 66mb of weights takes
    a second or two, and doing it per request would dominate every response.
    """
    t0 = time.perf_counter()
    # app.state is a namespace hanging off the app object. storing the model
    # here instead of at module scope means no `global` anywhere.
    app.state.clf = FoodClassifier()
    print(f"model loaded on {app.state.clf.device} "
          f"({len(app.state.clf.classes)} classes, {time.perf_counter() - t0:.1f}s)")
    yield          # <- the server runs here, for its whole life
    # nothing to clean up; python frees the model on exit


app = FastAPI(title="Food-101 Classifier", lifespan=lifespan)


def get_classifier(request: Request) -> FoodClassifier:
    """dependency: hands the loaded model to whichever endpoint asks for it.

    fastapi calls this before the endpoint runs and passes the return value in.
    swapping this function out is how a test injects a fake model without
    loading 66mb of weights.
    """
    clf = getattr(request.app.state, "clf", None)
    if clf is None:
        # lifespan runs before any request, so this shouldn't happen -- but if
        # the model failed to load, 503 is honest where an AttributeError 500 isn't
        raise HTTPException(503, "Model is not loaded yet.")
    return clf


@app.post("/api/classify")
async def classify(image: UploadFile = File(...),
                   topk: int = Form(3),
                   fast: bool = Form(False),
                   clf: FoodClassifier = Depends(get_classifier)):
    """take an uploaded image, return the top-k guesses as json."""
    raw = await image.read()

    if not raw:
        raise HTTPException(400, "No image data received.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image is over {MAX_UPLOAD_BYTES // 1024 // 1024} MB. "
                                 f"Try a smaller photo.")

    # never trust the client. the slider says max 5, but anyone can POST anything.
    # classify() clamps too, but failing here gives a clearer error
    topk = max(1, min(topk, len(clf.classes)))

    try:
        img = open_image(raw)
    except UnidentifiedImageError:
        raise HTTPException(400, "That file isn't an image we can read. Try a JPG or PNG.")

    # inference is blocking cpu work. in an `async def` endpoint, running it
    # directly would freeze the event loop -- no other request, not even a css
    # file, gets served until it finishes. run_in_threadpool hands it to a
    # worker thread so the server stays responsive.
    t0 = time.perf_counter()
    predictions = await run_in_threadpool(clf.classify, img, topk, fast)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    return {
        "filename": image.filename,
        "device": str(clf.device),
        "passes": 2 if fast else 6,
        "elapsed_ms": elapsed_ms,
        "predictions": [{"label": name, "confidence": conf} for name, conf in predictions],
    }


@app.get("/api/classes")
async def classes(clf: FoodClassifier = Depends(get_classifier)):
    """the full label list, so the page can show what the model actually knows."""
    return {"count": len(clf.classes), "classes": clf.classes}


@app.get("/api/health")
async def health(request: Request):
    """cheap endpoint for uptime checks / confirming the model actually loaded.

    deliberately does NOT use Depends(get_classifier): that dependency raises
    503 when the model is missing, and reporting the missing model is this
    endpoint's whole job. so it reads app.state directly.
    """
    clf = getattr(request.app.state, "clf", None)
    return {"ok": clf is not None,
            "device": str(clf.device) if clf else None,
            "classes": len(clf.classes) if clf else 0}


# mounting at "/" catches every path that didn't match a route above, so this
# MUST come last. html=True makes "/" serve static/index.html.
#
# guarded because StaticFiles raises at IMPORT time if the directory is missing.
# locally static/ is right here and this serves the page. on vercel the CDN
# serves the page and this function only ever sees /api/*, so the directory may
# not be in the bundle -- without the guard the whole function would fail to boot.
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
