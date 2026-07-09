"""MFMF Laboratory API.

Computational laboratory for 'Mathematical Foundations of Modern Finance'.
Every endpoint implements a Modern Finance Laboratory module of the book;
every module carries the book's validation checks with seeded, reproducible
runs. The numbers returned here match the course-site notebooks and Excel
workbooks by construction — the same seeded engines drive all three.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import (ch01, ch02, ch03, ch04, ch05, ch06, ch07,
                      ch08, ch09, ch10, ch11, ch12, ch13, ch14)

app = FastAPI(
    title="MFMF Laboratory API",
    description=(
        "Computational laboratory for 'Mathematical Foundations of Modern "
        "Finance'. Every endpoint implements a Modern Finance Laboratory "
        "module of the book; every module carries the book's validation "
        "checks with seeded, reproducible runs."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (ch01, ch02, ch03, ch04, ch05, ch06, ch07,
          ch08, ch09, ch10, ch11, ch12, ch13, ch14):
    app.include_router(r.router)


@app.get("/")
def root():
    return {
        "name": "MFMF Laboratory API",
        "book": "Mathematical Foundations of Modern Finance",
        "chapters": 14,
        "docs": "/docs",
        "convention": "Seeds follow 2026CCNN; every chapter exposes GET /api/chNN/validation.",
    }
