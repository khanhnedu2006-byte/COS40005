"""Regression coverage for the web route-result response contract."""

import asyncio

from fastapi.routing import serialize_response

from Web.app import create_app


def test_completed_road_route_status_preserves_geometry_and_distances():
    app = create_app()
    endpoint = next(
        route
        for route in app.routes
        if route.path == "/api/runs/{run_id}" and "GET" in route.methods
    )
    outcome = {
        "status": "result",
        "result": {
            "route": ["H01", "H02"],
            "score": 4.5,
            "distance_km": 12.7,
            "duration_min": 35.0,
            "collected": ["P001"],
            "geometry": {
                "type": "LineString",
                "coordinates": [[108.2, 16.0], [108.3, 16.1]],
            },
            "narrow_m": 100.0,
            "unknown_width_m": 500.0,
            "instructions": [
                {
                    "action": "Start on",
                    "road": "Lê Duẩn",
                    "distance_m": 120.0,
                    "coordinate": [108.2, 16.0],
                    "coordinates": [[108.2, 16.0], [108.3, 16.1]],
                }
            ],
        },
        "message": None,
    }
    serialized = asyncio.run(
        serialize_response(field=endpoint.response_field, response_content=outcome)
    )
    assert serialized == outcome
