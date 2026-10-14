from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from engine.interfaces import SourceType
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: int = Field(description="User identifier")
    current_time: Optional[datetime] = Field(
        default=None, description="User's current time of request"
    )
    source_page: SourceType = Field(
        description="App page, for which recommendations are needed",
    )
    num_candidates: int = Field(
        gt=0, description="Number of items to return in response"
    )
    context_item_ids: Optional[List[int]] = Field(
        default=None,
        description="Context item identifiers (item to item recommendations)",
    )
    page_token: Optional[str] = Field(
        default=None,
        description="Pagination token, empty for first or from previous page response",
    )


class RecommendResponse(BaseModel):
    candidates: List[int] = Field(
        description="Sorted list of recommended items identifiers"
    )
    next_page_token: str = Field(
        description="New pagination token from request with added candidates"
    )
