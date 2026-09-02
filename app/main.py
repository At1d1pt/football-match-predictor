from fastapi import FastAPI
from contextlib import asynccontextmanager
import pathlib

from .schemas import MatchRequest
from .predictor import predict, get_elo_map
from .cache import fetch_and_cache_standings, get_standings

@asynccontextmanager
async def lifespan(app: FastAPI):
    fetch_and_cache_standings()
    yield

env = pathlib.Path(".env")
if not env.exists():
    raise FileNotFoundError(".env not found. Refer to README.md for instructions on setup")

app = FastAPI(
    title="Golazo",
    description="backend for the api",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def home():
    return {
        "message": "running"
    }

@app.get("/stadings")
def get_league_standings():
    return get_standings()

@app.post("/predict")
def predict_match(request: MatchRequest):
    return predict(request.home, request.away, request.date)

@app.get("/elo")
def get_elo():
    return get_elo_map()