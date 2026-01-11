"""
Pydantic Models for Quantum Traffic Optimization API.

This module defines the data schemas for API requests and responses,
with validation for the Vijayawada bounding box.
"""

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# Vijayawada bounding box constraints (smaller area for faster OSM download)
BBOX_SOUTH = 16.50
BBOX_NORTH = 16.55
BBOX_WEST = 80.62
BBOX_EAST = 80.68


class DeliveryPoint(BaseModel):
    """
    A delivery location with coordinates and priority.
    
    Attributes:
        lat: Latitude within Vijayawada bbox (16.5 - 16.7).
        lng: Longitude within Vijayawada bbox (80.6 - 80.7).
        priority: Delivery priority (0.0 - 10.0, higher = more urgent).
        id: Optional unique identifier for the delivery.
        name: Optional human-readable name.
    """
    lat: float = Field(..., ge=BBOX_SOUTH, le=BBOX_NORTH, description="Latitude")
    lng: float = Field(..., ge=BBOX_WEST, le=BBOX_EAST, description="Longitude")
    priority: float = Field(default=1.0, ge=0.0, le=10.0, description="Delivery priority")
    id: Optional[str] = Field(default=None, description="Unique delivery ID")
    name: Optional[str] = Field(default=None, description="Location name")
    
    @field_validator('id', mode='before')
    @classmethod
    def set_default_id(cls, v):
        """Generate ID if not provided."""
        return v or str(uuid4())[:8]
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (lat, lng) tuple."""
        return (self.lat, self.lng)


class OptimizeRequest(BaseModel):
    """
    Request schema for route optimization.

    Attributes:
        current_loc: Current location as (lat, lng) tuple.
        deliveries: List of delivery points to visit.
        traffic_level: Traffic condition ('low', 'medium', 'high').
        include_map: Whether to include map HTML in response.
    """
    current_loc: Tuple[float, float] = Field(
        ...,
        description="Current location (lat, lng)"
    )
    deliveries: List[DeliveryPoint] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Delivery locations to optimize"
    )
    traffic_level: str = Field(
        default='low',
        pattern='^(low|medium|high)$',
        description="Traffic condition"
    )
    include_map: bool = Field(
        default=True,
        description="Include Folium map HTML in response"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "current_loc": [16.5063, 80.6480],
                "deliveries": [
                    {"lat": 16.5175, "lng": 80.6198, "priority": 2},
                    {"lat": 16.5412, "lng": 80.6352, "priority": 1}
                ],
                "traffic_level": "medium",
                "include_map": True
            }
        }
    
    @field_validator('current_loc')
    @classmethod
    def validate_current_loc(cls, v):
        """Validate current location is within bbox."""
        lat, lng = v
        if not (BBOX_SOUTH <= lat <= BBOX_NORTH):
            raise ValueError(f"Latitude {lat} out of bounds [{BBOX_SOUTH}, {BBOX_NORTH}]")
        if not (BBOX_WEST <= lng <= BBOX_EAST):
            raise ValueError(f"Longitude {lng} out of bounds [{BBOX_WEST}, {BBOX_EAST}]")
        return v
    
    @field_validator('deliveries')
    @classmethod
    def validate_deliveries(cls, v):
        """Ensure at least one delivery."""
        if len(v) == 0:
            raise ValueError("At least one delivery location required")
        return v


class SequenceStop(BaseModel):
    """
    A stop in the optimized sequence.
    
    Attributes:
        position: Position in sequence (0 = first stop after depot).
        delivery: The delivery point.
        distance_from_prev: Distance from previous stop in meters.
        eta_from_prev: Estimated time from previous stop in minutes.
        cumulative_distance: Total distance traveled so far.
        cumulative_eta: Total time elapsed so far.
    """
    position: int = Field(..., ge=0, description="Position in sequence")
    delivery: DeliveryPoint
    distance_from_prev: float = Field(..., ge=0, description="Distance from previous (m)")
    eta_from_prev: float = Field(..., ge=0, description="Time from previous (min)")
    cumulative_distance: float = Field(..., ge=0, description="Total distance (m)")
    cumulative_eta: float = Field(..., ge=0, description="Total time (min)")


class OptimizeResult(BaseModel):
    """
    Response schema for route optimization.
    
    Attributes:
        route_id: Unique identifier for this route.
        sequence: Ordered list of stops.
        total_distance: Total route distance in meters.
        total_eta: Total estimated time in minutes.
        optimization_time: Time taken to compute in seconds.
        traffic_level: Traffic condition used.
        map_html: Optional Folium map HTML.
        improvement_over_greedy: Percentage improvement vs greedy.
    """
    route_id: str = Field(..., description="Unique route identifier")
    sequence: List[SequenceStop] = Field(..., description="Ordered stops")
    total_distance: float = Field(..., ge=0, description="Total distance (m)")
    total_eta: float = Field(..., ge=0, description="Total time (min)")
    optimization_time: float = Field(..., ge=0, description="Compute time (s)")
    traffic_level: str = Field(..., description="Traffic condition used")
    map_html: Optional[str] = Field(default=None, description="Folium map HTML")
    improvement_over_greedy: Optional[float] = Field(
        default=None,
        description="% improvement over greedy"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy", description="Service status")
    version: str = Field(default="1.0.0", description="API version")
    graph_loaded: bool = Field(..., description="Graph cache status")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")


class ReoptimizeMessage(BaseModel):
    """
    WebSocket message for re-optimization.
    
    Attributes:
        type: Message type ('update', 'traffic_change', 'new_delivery').
        data: Message payload.
    """
    type: str = Field(..., pattern='^(update|traffic_change|new_delivery|ack)$')
    data: Dict[str, Any] = Field(default_factory=dict)


# Sample data for testing (within smaller bbox)
SAMPLE_DELIVERIES = [
    DeliveryPoint(lat=16.51, lng=80.63, priority=2.0, name="Delivery A"),
    DeliveryPoint(lat=16.52, lng=80.64, priority=1.0, name="Delivery B"),
    DeliveryPoint(lat=16.53, lng=80.65, priority=3.0, name="Delivery C"),
    DeliveryPoint(lat=16.54, lng=80.66, priority=1.5, name="Delivery D"),
]

SAMPLE_REQUEST = OptimizeRequest(
    current_loc=(16.505, 80.625),
    deliveries=SAMPLE_DELIVERIES,
    traffic_level='medium'
)
