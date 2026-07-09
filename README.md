# MFMF Laboratory API

The computational backend for the *Mathematical Foundations of Modern Finance*
laboratory webapp. A FastAPI service exposing the book's fourteen Modern
Finance Laboratory modules as REST endpoints, with seeded, reproducible runs
whose numbers match the course-site notebooks and Excel workbooks by
construction.

## Design

- One router per chapter, mounted at `/api/ch01/…` through `/api/ch14/…`.
- Every chapter exposes `GET /api/chNN/validation`, returning the module's
  validation checks (this powers the webapp's header Validation pill).
- Compute endpoints are `POST` with a typed request body; all fields are
  optional and default to the book's calibrated values, so a client may send
  a partial parameter set.
- Seeds follow the `2026CCNN` convention and are surfaced in every response.
- CORS is open (the webapp is served from a different origin).

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for interactive documentation.

## Deploy (Railway)

Railway auto-detects the `Procfile`. Connect this repo as the service source;
the start command is:

```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Structure

```
app/
  main.py          FastAPI app; mounts the 14 routers, CORS, root
  common.py        JSON-safe serialization; request-override helper
  models.py        Pydantic request models (one per chapter family)
  engines/         the 14 seeded engines (mfmf_engine_chNN.py)
  routers/         the 14 chapter routers (chNN.py)
requirements.txt
Procfile
railway.toml
openapi.json       the generated API contract
```

## The engines

Each `engines/mfmf_engine_chNN.py` reproduces its chapter's book values
exactly and ships a `validation_checks()` routine. The same engine files
drive the course-site notebooks and workbooks, so the three artifacts
(notebook, workbook, webapp) always agree.
