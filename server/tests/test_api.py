from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "server"
for path in (REPO_ROOT, SERVER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastapi.testclient import TestClient

from server.api import main


def test_health_endpoint() -> None:
    client = TestClient(main.app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_no_vig_probabilities_normalize_moneylines() -> None:
    favorite, underdog = main.no_vig_probabilities(-150, 130)

    assert favorite is not None
    assert underdog is not None
    assert round(favorite + underdog, 8) == 1
    assert favorite > underdog


def test_edge_payload_flags_five_percent_edge() -> None:
    payload = main.edge_payload(model_probability=0.58, implied_probability=0.52)

    assert payload["has_edge"] is True
    assert round(payload["edge"], 4) == 0.06
    assert round(payload["kelly_fraction"], 4) == 0.125


def test_cached_next_event_card_sorts_by_importance(tmp_path: Path) -> None:
    card_path = tmp_path / "ufc_next_event_card.csv"
    card_path.write_text(
        "\n".join(
            [
                "event,date,source_url,scraped_at,importance_order,fighter1,fighter2,fighter1_current_odds,fighter2_current_odds",
                "UFC Test,2026-05-09,https://example.test,2026-05-06T00:00:00Z,2,Prelim A,Prelim B,-110,-110",
                "UFC Test,2026-05-09,https://example.test,2026-05-06T00:00:00Z,1,Main A,Main B,-150,130",
            ]
        )
    )

    payload = main.cached_next_event_card(card_path)

    assert payload["event"] == "UFC Test"
    assert [fight["fighter1"] for fight in payload["fights"]] == ["Main A", "Prelim A"]


def test_next_event_edges_uses_cached_card_and_model(monkeypatch, tmp_path: Path) -> None:
    card_path = tmp_path / "ufc_next_event_card.csv"
    card_path.write_text(
        "\n".join(
            [
                "event,date,source_url,scraped_at,importance_order,fighter1,fighter2,fighter1_open_odds,fighter2_open_odds,fighter1_current_odds,fighter2_current_odds",
                "UFC Test,2026-05-09,https://example.test,2026-05-06T00:00:00Z,1,Fighter A,Fighter B,-140,120,-150,130",
            ]
        )
    )

    class FakePrediction:
        def as_dict(self) -> dict:
            return {
                "fighter1": "Fighter A",
                "fighter2": "Fighter B",
                "fighter1_win_probability": 0.66,
                "fighter2_win_probability": 0.34,
                "predicted_winner": "Fighter A",
                "predicted_winner_probability": 0.66,
            }

    class FakePredictor:
        def resolve_fighter_name(self, name: str) -> str:
            return name

        def predict(self, **kwargs) -> FakePrediction:
            assert kwargs["fighter1_close_odds"] == -150
            assert kwargs["fighter2_close_odds"] == 130
            return FakePrediction()

    monkeypatch.setattr(main, "NEXT_EVENT_CARD_PATH", card_path)
    monkeypatch.setattr(main, "get_predictor", lambda: FakePredictor())

    payload = main.upcoming_event_edges()

    assert payload["fights"][0]["recommended_side"] == "Fighter A"
    assert payload["fights"][0]["recommended_edge"]["has_edge"] is True
