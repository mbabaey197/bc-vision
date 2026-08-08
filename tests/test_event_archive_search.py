import html
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

import app.database as database
import app.main as main


def _as_operator(monkeypatch):
    monkeypatch.setattr(main, "auth", lambda request: "operator")


@pytest.fixture
def isolated_archive(tmp_path, monkeypatch):
    db_path = tmp_path / "archive.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    _as_operator(monkeypatch)
    return db_path


def _insert_event(
    con,
    *,
    plate_text="12-ب-345-67",
    plate_norm="12ب34567",
    plate_region="67",
    camera_name="Gate",
    camera_id=None,
    city="",
    created_at="2026-07-30 12:00:00",
    vehicle_type="سواری",
    vehicle_color="سفید",
    image_path="",
    plate_image_path="",
    video_path="",
):
    return int(
        con.execute(
            "INSERT INTO plate_events("
            "plate_text,plate_norm,plate_region,confidence,"
            "camera_id,camera_name,city,created_at,"
            "vehicle_type,vehicle_color,image_path,plate_image_path,"
            "video_path"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                plate_text,
                plate_norm,
                plate_region,
                0.91,
                camera_id,
                camera_name,
                city,
                created_at,
                vehicle_type,
                vehicle_color,
                image_path,
                plate_image_path,
                video_path,
            ),
        ).lastrowid
    )


def _dashboard_event_ids(text):
    return [
        int(value)
        for value in re.findall(
            r"action='/events/(\d+)/correct'",
            text,
        )
    ]


def _report_event_ids(text):
    return [
        int(value)
        for value in re.findall(
            r"href='/events/(\d+)'>جزئیات و پخش",
            text,
        )
    ]


def test_dashboard_defaults_to_twelve_rows_and_pages_twenty_seven_events(
    isolated_archive,
):
    with database.connect() as con:
        for index in range(1, 28):
            _insert_event(
                con,
                camera_name=f"Dashboard-{index:02d}",
            )

    with TestClient(main.app) as client:
        first = client.get("/dashboard")
        second = client.get(
            "/dashboard",
            params={"events_page": 2, "events_snapshot": 27},
        )
        third = client.get(
            "/dashboard",
            params={"events_page": 3, "events_snapshot": 27},
        )

    assert first.status_code == 200
    assert _dashboard_event_ids(first.text) == list(range(27, 15, -1))
    assert "صفحه ۱ از ۳" in first.text
    assert (
        "/dashboard?events_snapshot=27&amp;events_page=2"
        in first.text
    )

    assert second.status_code == 200
    assert _dashboard_event_ids(second.text) == list(range(15, 3, -1))
    assert "صفحه ۲ از ۳" in second.text

    assert third.status_code == 200
    assert _dashboard_event_ids(third.text) == [3, 2, 1]
    assert "نمایش ۲۵ تا ۲۷ از ۲۷ رکورد" in third.text


def test_dashboard_polling_returns_only_after_a_new_event(
    isolated_archive,
):
    with database.connect() as con:
        for index in range(1, 14):
            _insert_event(
                con,
                camera_name=f"Polling-{index:02d}",
            )
        previous_latest = int(
            con.execute(
                "SELECT MAX(id) FROM plate_events"
            ).fetchone()[0]
        )

    with TestClient(main.app) as client:
        unchanged = client.get(
            "/api/dashboard/recent-events",
            params={"after": previous_latest},
        )
        with database.connect() as con:
            new_id = _insert_event(
                con,
                camera_name="POLLING-NEW-EVENT",
            )
        changed = client.get(
            "/api/dashboard/recent-events",
            params={"after": previous_latest},
        )

    assert unchanged.status_code == 200
    assert unchanged.json() == {
        "latest_id": previous_latest,
        "rows_html": "",
    }

    payload = changed.json()
    assert changed.status_code == 200
    assert payload["latest_id"] == new_id
    assert _dashboard_event_ids(payload["rows_html"]) == list(
        range(new_id, new_id - 12, -1)
    )
    assert "POLLING-NEW-EVENT" in payload["rows_html"]
    assert "صفحه ۱ از ۲" in payload["pagination_html"]


def test_events_pagination_preserves_every_active_filter(
    isolated_archive,
):
    with database.connect() as con:
        for _ in range(27):
            _insert_event(
                con,
                camera_name="Gate-A",
                city="تهران",
                vehicle_type="سواری",
                vehicle_color="سفید",
            )

    filters = {
        "q": "۱۲",
        "camera": "Gate-A",
        "city": "تهران",
        "region": "۶۷",
        "status": "unknown",
        "vehicle_type": "سواری",
        "vehicle_color": "سفید",
        "per_page": 25,
    }
    with TestClient(main.app) as client:
        first = client.get("/events", params=filters)
        second = client.get(
            "/events",
            params={**filters, "events_page": 2},
        )

    assert first.status_code == 200
    assert _report_event_ids(first.text) == list(range(27, 2, -1))
    assert "صفحه ۱ از ۲" in first.text

    page_two_urls = []
    for raw_url in re.findall(
        r"<a class='page-number' href='([^']+)'>۲</a>",
        first.text,
    ):
        parsed = urlparse(html.unescape(raw_url))
        if parse_qs(parsed.query).get("events_page") == ["2"]:
            page_two_urls.append(parsed)
    assert len(page_two_urls) == 1
    query = parse_qs(page_two_urls[0].query)
    assert query == {
        "q": ["۱۲"],
        "camera": ["Gate-A"],
        "city": ["تهران"],
        "region": ["۶۷"],
        "status": ["unknown"],
        "vehicle_type": ["سواری"],
        "vehicle_color": ["سفید"],
        "per_page": ["25"],
        "events_page": ["2"],
    }

    assert second.status_code == 200
    assert _report_event_ids(second.text) == [2, 1]
    assert "نمایش ۲۶ تا ۲۷ از ۲۷ رکورد" in second.text


@pytest.mark.parametrize(
    "query",
    ("1", "۱", "١", "12", "۱۲", "١٢"),
)
def test_partial_plate_search_accepts_one_or_two_latin_persian_arabic_digits(
    isolated_archive,
    query,
):
    with database.connect() as con:
        matching_id = _insert_event(
            con,
            camera_name="MATCH-PARTIAL-PLATE",
            plate_text="12-ب-345-67",
            plate_norm="12ب34567",
        )
        _insert_event(
            con,
            camera_name="OTHER-PLATE",
            plate_text="98-ط-555-43",
            plate_norm="98ط55543",
            plate_region="43",
        )

    with TestClient(main.app) as client:
        response = client.get("/events", params={"q": query})

    assert response.status_code == 200
    assert "MATCH-PARTIAL-PLATE" in response.text
    assert _report_event_ids(response.text) == [matching_id]


def test_jalali_date_and_time_are_applied_in_sql_before_pagination(
    isolated_archive,
    monkeypatch,
):
    assert main._gregorian_to_jalali(2024, 3, 20) == (1403, 1, 1)
    with database.connect() as con:
        _insert_event(
            con,
            camera_name="OLD-TARGET-AMONG-OVER-500",
            # UTC; displayed and searched as 14:35 in Tehran.
            created_at="2024-03-20 11:05:00",
        )
        con.executemany(
            "INSERT INTO plate_events("
            "plate_text,plate_norm,plate_region,confidence,camera_name,"
            "city,created_at"
            ") VALUES(?,?,?,?,?,?,?)",
            [
                (
                    "98-ط-555-43",
                    "98ط55543",
                    "43",
                    0.80,
                    f"Recent-{index:03d}",
                    "تهران",
                    "2026-07-30 12:00:00",
                )
                for index in range(501)
            ],
        )

    statements = []
    original_connect = main.connect

    def traced_connect():
        con = original_connect()
        con.set_trace_callback(statements.append)
        return con

    monkeypatch.setattr(main, "connect", traced_connect)
    with TestClient(main.app) as client:
        response = client.get(
            "/events",
            params={
                "date_from": "۱۴۰۳/۰۱/۰۱",
                "time_from": "۱۴:۳۵",
                "date_to": "۱۴۰۳/۰۱/۰۱",
                "time_to": "۱۴:۳۵",
            },
        )

    assert response.status_code == 200
    assert _report_event_ids(response.text) == [1]
    event_selects = [
        statement.lower()
        for statement in statements
        if " from plate_events e" in statement.lower()
        and " limit " in statement.lower()
    ]
    assert event_selects
    assert any(
        "e.created_at>=" in statement
        and "e.created_at<" in statement
        for statement in event_selects
    )


def test_jalali_time_range_uses_tehran_midnight_boundaries(
    isolated_archive,
):
    with database.connect() as con:
        included = _insert_event(
            con,
            camera_name="LOCAL-00-15",
            # 1405/05/08 00:15 in Tehran.
            created_at="2026-07-29 20:45:00",
        )
        _insert_event(
            con,
            camera_name="NEXT-LOCAL-DAY",
            created_at="2026-07-30 21:00:00",
        )

    with TestClient(main.app) as client:
        response = client.get(
            "/events",
            params={
                "date_from": "۱۴۰۵/۰۵/۰۸",
                "time_from": "۰۰:۰۰",
                "date_to": "۱۴۰۵/۰۵/۰۸",
                "time_to": "۰۰:۳۰",
            },
        )

    assert response.status_code == 200
    assert _report_event_ids(response.text) == [included]
    assert "LOCAL-00-15" in response.text


def test_city_is_historical_snapshot_and_region_filter_is_normalized(
    isolated_archive,
):
    with database.connect() as con:
        camera_id = int(
            con.execute(
                "INSERT INTO cameras(name,rtsp_url,location,city) "
                "VALUES(?,?,?,?)",
                ("Historical Gate", "rtsp://gate", "ورودی شمالی", "تهران"),
            ).lastrowid
        )
        event_id = _insert_event(
            con,
            camera_id=camera_id,
            camera_name="CITY-SNAPSHOT-EVENT",
            city="تهران",
            plate_region="67",
        )
        con.execute(
            "UPDATE cameras SET city=? WHERE id=?",
            ("شیراز", camera_id),
        )

    with TestClient(main.app) as client:
        historical = client.get(
            "/events",
            params={"city": "تهران", "region": "۶۷"},
        )
        changed_camera = client.get(
            "/events",
            params={"city": "شیراز", "region": "67"},
        )

    assert historical.status_code == 200
    assert _report_event_ids(historical.text) == [event_id]
    assert "CITY-SNAPSHOT-EVENT" in historical.text
    assert "کد پلاک: ۶۷" in historical.text

    assert changed_camera.status_code == 200
    assert _report_event_ids(changed_camera.text) == []


@pytest.mark.parametrize(
    "params",
    (
        {"q": "' OR 1=1 --"},
        {"q": "%%%%__"},
        {"city": "' OR 1=1 --"},
        {"region": "' OR 1=1 --"},
        {"date_from": "۱۴۰۵/۱۳/۰۱"},
        {"time_from": "۲۵:۹۹"},
        {"date_from": "۱۴۰۵/۰۵/۰۹", "date_to": "۱۴۰۵/۰۵/۰۸"},
    ),
)
def test_invalid_or_sql_like_filters_fail_closed(
    isolated_archive,
    params,
):
    with database.connect() as con:
        _insert_event(
            con,
            camera_name="MUST-NOT-BE-BROADLY-RETURNED",
            city="تهران",
        )

    with TestClient(main.app) as client:
        response = client.get("/events", params=params)

    assert response.status_code == 200
    assert _report_event_ids(response.text) == []
    with database.connect() as con:
        assert con.execute(
            "SELECT COUNT(*) FROM plate_events"
        ).fetchone()[0] == 1


def test_html_like_plate_filter_is_escaped_and_does_not_match_all(
    isolated_archive,
):
    with database.connect() as con:
        _insert_event(
            con,
            camera_name="XSS-SECRET-EVENT",
        )
    payload = "<script>alert(1)</script>"

    with TestClient(main.app) as client:
        response = client.get("/events", params={"q": payload})

    assert response.status_code == 200
    assert _report_event_ids(response.text) == []
    assert (
        "value='&lt;script&gt;alert(1)&lt;/script&gt;'"
        in response.text
    )


def test_archive_video_events_persists_vehicle_plate_and_video_paths(
    isolated_archive,
    tmp_path,
):
    vehicle = tmp_path / "archived-vehicle.jpg"
    plate = tmp_path / "archived-plate.jpg"
    video = tmp_path / "archived-video.mp4"
    vehicle.write_bytes(b"vehicle")
    plate.write_bytes(b"plate")
    video.write_bytes(b"video")

    archived = main._archive_video_test_events(
        [{
            "plate": "31-ط-556-74",
            "plate_norm": "31ط55674",
            "raw_guess_text": "99-ب-999-99",
            "raw_guess_norm": "99ب99999",
            "valid": True,
            "confidence": 0.93,
            "ocr_confidence": 0.92,
            "ocr_engine": "test-ocr",
            "image_path": str(vehicle),
            "plate_path": str(plate),
            "media_status": "complete",
            "video_second": 4.25,
            "vehicle_type": "سواری",
            "vehicle_color": "سفید",
        }],
        video,
        "نمونه آرشیو",
    )

    assert len(archived) == 1
    assert archived[0]["event_id"] > 0
    with database.connect() as con:
        row = con.execute(
            "SELECT * FROM plate_events WHERE id=?",
            (archived[0]["event_id"],),
        ).fetchone()

    assert row["plate_text"] == "31-ط-556-74"
    assert row["plate_norm"] == "31ط55674"
    assert row["raw_guess_text"] == "99-ب-999-99"
    assert row["plate_region"] == "74"
    assert row["image_path"] == str(vehicle)
    assert row["plate_image_path"] == str(plate)
    assert row["video_path"] == str(video)
    assert row["video_second"] == pytest.approx(4.25)
    assert row["vehicle_type"] == "سواری"
    assert row["vehicle_color"] == "سفید"
    assert row["source"] == "video-test"
    assert row["camera_name"] == "تست ویدئو: نمونه آرشیو"
    assert row["media_status"] == "complete"


def test_unverified_video_event_cannot_be_archived_as_confirmed(
    isolated_archive,
    tmp_path,
):
    archived = main._archive_video_test_events(
        [{
            "plate": "ناخوانا",
            "raw_guess_text": "12-ب-345-67",
            "raw_guess_norm": "12ب34567",
            "valid": False,
            "needs_review": False,
            "confidence": 0.40,
        }],
        tmp_path / "video.mp4",
        "نمونه ناخوانا",
    )

    with database.connect() as con:
        row = con.execute(
            "SELECT plate_text,plate_norm,review_status "
            "FROM plate_events WHERE id=?",
            (archived[0]["event_id"],),
        ).fetchone()

    assert dict(row) == {
        "plate_text": "ناخوانا",
        "plate_norm": "",
        "review_status": "unreadable",
    }


def test_event_detail_matches_watchlist_by_canonical_plate_norm(
    isolated_archive,
):
    with database.connect() as con:
        event_id = _insert_event(
            con,
            plate_text="۱۲-ب-۳۴۵-۶۷",
            plate_norm="12ب34567",
        )
        con.execute(
            "INSERT INTO plate_watchlist("
            "plate_text,plate_norm,status,owner_name"
            ") VALUES(?,?,?,?)",
            ("۱۲ ب ۳۴۵ ایران ۶۷", "12ب34567", "blocked", "مالک تست"),
        )

    with TestClient(main.app) as client:
        response = client.get(f"/events/{event_id}")

    assert response.status_code == 200
    assert "غیرمجاز" in response.text
    assert "مالک تست" in response.text


def test_media_allows_historical_references_outside_current_root_only(
    isolated_archive,
    tmp_path,
):
    current_root = tmp_path / "current-storage"
    snapshots = current_root / "snapshots"
    plates = current_root / "plates"
    videos = current_root / "videos"
    backups = current_root / "backups"
    for folder in (snapshots, plates, videos, backups):
        folder.mkdir(parents=True)

    historical_root = tmp_path / "retired-storage"
    historical_root.mkdir()
    historical_vehicle = historical_root / "vehicle.jpg"
    historical_plate = historical_root / "plate.jpg"
    historical_video = historical_root / "video.mp4"
    unreferenced = historical_root / "not-in-database.jpg"
    historical_vehicle.write_bytes(b"historical-vehicle")
    historical_plate.write_bytes(b"historical-plate")
    historical_video.write_bytes(b"historical-video")
    unreferenced.write_bytes(b"private")

    with database.connect() as con:
        con.executemany(
            "UPDATE settings SET value=? WHERE key=?",
            (
                (str(current_root), "storage_root"),
                (str(snapshots), "snapshot_path"),
                (str(plates), "plate_path"),
                (str(videos), "video_path"),
                (str(backups), "backup_path"),
                (
                    json.dumps([str(historical_root)]),
                    "media_roots_history",
                ),
            ),
        )
        _insert_event(
            con,
            camera_name="HISTORICAL-MEDIA",
            image_path=str(historical_vehicle),
            plate_image_path=str(historical_plate),
            video_path=str(historical_video),
        )

    with TestClient(main.app) as client:
        vehicle_response = client.get(
            "/media",
            params={"path": str(historical_vehicle)},
        )
        plate_response = client.get(
            "/media",
            params={"path": str(historical_plate)},
        )
        video_response = client.get(
            "/media",
            params={"path": str(historical_video)},
        )
        blocked = client.get(
            "/media",
            params={"path": str(unreferenced)},
        )

    assert vehicle_response.status_code == 200
    assert vehicle_response.content == b"historical-vehicle"
    assert plate_response.status_code == 200
    assert plate_response.content == b"historical-plate"
    assert video_response.status_code == 200
    assert video_response.content == b"historical-video"
    assert blocked.status_code == 404
