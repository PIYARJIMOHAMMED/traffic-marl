"""
Input validation for traffic-control requests.
"""

from pydantic import BaseModel, Field, field_validator


class TrafficControlRequest(BaseModel):
    intersection: str = Field(min_length=1, max_length=20)
    action: int = Field(ge=0, le=1)
    green_timer: float = Field(ge=0, le=3600)
    in_yellow: bool = False

    @field_validator("intersection")
    @classmethod
    def validate_intersection(cls, value: str) -> str:
        value = value.strip().upper()

        allowed = {"A", "B", "C", "D"}

        if value not in allowed:
            raise ValueError(
                f"intersection must be one of {sorted(allowed)}"
            )

        return value


class ScenarioRequest(BaseModel):
    scenario: str = Field(min_length=1, max_length=50)

    @field_validator("scenario")
    @classmethod
    def validate_scenario(cls, value: str) -> str:
        value = value.strip().lower()

        allowed = {
            "baseline",
            "low_demand",
            "balanced",
            "morning_rush",
            "evening_rush",
            "high_demand",
            "north_south_heavy",
            "east_west_heavy",
            "random",
            "stress_test",
        }

        if value not in allowed:
            raise ValueError("Unsupported traffic scenario")

        return value