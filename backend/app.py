from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.lib import startup
from backend.routers import cluster, route, similarity, visualize


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup.initialize()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cluster.router)
app.include_router(visualize.router)
app.include_router(route.router)
app.include_router(similarity.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve React frontend (must be last — catches all non-API routes)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=7860, reload=True)
