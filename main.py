from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
from transformers import pipeline

app = FastAPI()

# Initialize sentiment analysis pipeline
sentiment_model = pipeline("sentiment-analysis")

# Define the input data structure
class Review(BaseModel):
    product_id: str
    review_id: str
    review_text: str

    @validator('review_text')
    def review_text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Review text cannot be empty.")
        return v

class ReviewSentimentResponse(BaseModel):
    review_id: str
    review_text: str
    sentiment: str
    confidence_score: float

# API to analyze sentiment for a single review
@app.post("/analyze", response_model=ReviewSentimentResponse)
async def analyze_sentiment(review: Review):
    try:
        result = sentiment_model(review.review_text)[0]
        sentiment = result['label'].lower()
        score = result['score']
        
        if sentiment == "positive":
            label = "positive"
        elif sentiment == "negative":
            label = "negative"
        else:
            label = "neutral"

        return {
            "review_id": review.review_id,
            "review_text": review.review_text,
            "sentiment": label,
            "confidence_score": score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API to analyze sentiment for multiple reviews
@app.post("/analyze_batch", response_model=List[ReviewSentimentResponse])
async def analyze_sentiment_batch(reviews: List[Review]):
    responses = []
    for review in reviews:
        try:
            result = sentiment_model(review.review_text)[0]
            sentiment = result['label'].lower()
            score = result['score']
            
            if sentiment == "positive":
                label = "positive"
            elif sentiment == "negative":
                label = "negative"
            else:
                label = "neutral"

            responses.append({
                "review_id": review.review_id,
                "review_text": review.review_text,
                "sentiment": label,
                "confidence_score": score
            })
        except Exception as e:
            responses.append({
                "review_id": review.review_id,
                "review_text": review.review_text,
                "sentiment": "error",
                "confidence_score": 0.0
            })
    return responses
