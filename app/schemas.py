from pydantic import BaseModel

class MatchRequest(BaseModel):
    home: str
    away: str
    #competition: str
    date: str

class PredictionResponse(BaseModel):
    winner: str
    probabilities: dict
    predicted_score: dict
    confidence: float
    explanation: list[str]