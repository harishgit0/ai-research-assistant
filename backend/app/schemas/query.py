from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


class SourceResponse(BaseModel):
    filename: str
    page_number: int
    chunk_index: int
    reranker_score: float
    text: str

    
class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]