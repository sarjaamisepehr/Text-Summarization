"""
Created on Jan 6 2023
@author: Sepehr Sarjami
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import functools
from starlette.middleware.cors import CORSMiddleware

from summa_score_sentences import summarize as summarize_textrank


class HighlightRequest(BaseModel):
    text: str
    model: str = "textrank"


class Sentence(BaseModel):
    paragraph: int
    index: int
    text: str
    score: float


class HighlightResults(BaseModel):
    success: bool
    message: str = ""
    sentences: List[Sentence] = []
    lang: str = ""

# 2. Create the app object
app = FastAPI()

origins = [
    # "http:localhost",
    # "http:localhost:8080",
    # "http:localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get("/")
def read_root():
    return {"Hello": "World There"}


def sentence_sort_function(sent_1, sent_2) -> bool:
    if sent_1.paragraph == sent_2.paragraph:
        return sent_1.index - sent_2.index
    return sent_1.paragraph - sent_2.paragraph

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post("/highlight/", response_model=HighlightResults)
def read_item(highlight_request: HighlightRequest):
    text_lang=""
    if highlight_request.model == "textrank":
        sentences, _, text_lang = summarize_textrank(highlight_request.text)
    else:
        return HighlightResults(
            success=False,
            message=f"'{highlight_request.model}' is not supported."
        )
    # Sort sentences
    sentences = sorted(
        sentences, key=functools.cmp_to_key(sentence_sort_function))
    # Create sentence obj
    sentence_objs = [
        Sentence(paragraph=x.paragraph, score=x.score,
                 text=x.text, index=x.index)
        for x in sentences
    ]
    # print(sentence_objs)
    return HighlightResults(
        success=True,
        sentences=sentence_objs,
        lang=text_lang
    )

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)