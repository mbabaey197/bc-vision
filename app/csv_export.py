"""Bounded, WAL-friendly CSV export for plate events."""
from __future__ import annotations

import csv
import unicodedata
from io import StringIO

_SQL_CELL_PREFIX_BYTES = 4097 * 4
_EVENT_EXPORT_COLUMNS = (
    "id,"
    f"substr(CAST(plate_text AS BLOB),1,{_SQL_CELL_PREFIX_BYTES}) "
    "AS plate_text,"
    f"substr(CAST(confidence AS BLOB),1,{_SQL_CELL_PREFIX_BYTES}) "
    "AS confidence,"
    f"substr(CAST(camera_name AS BLOB),1,{_SQL_CELL_PREFIX_BYTES}) "
    "AS camera_name,"
    f"substr(CAST(city AS BLOB),1,{_SQL_CELL_PREFIX_BYTES}) AS city,"
    f"substr(CAST(plate_region AS BLOB),1,{_SQL_CELL_PREFIX_BYTES}) "
    "AS plate_region,"
    f"substr(CAST(vehicle_type AS BLOB),1,{_SQL_CELL_PREFIX_BYTES}) "
    "AS vehicle_type,"
    f"substr(CAST(vehicle_color AS BLOB),1,{_SQL_CELL_PREFIX_BYTES}) "
    "AS vehicle_color,"
    f"substr(CAST(vehicle_confidence AS BLOB),1,{_SQL_CELL_PREFIX_BYTES}) "
    "AS vehicle_confidence,"
    f"substr(CAST(created_at AS BLOB),1,{_SQL_CELL_PREFIX_BYTES}) "
    "AS created_at"
)
EVENT_EXPORT_QUERY = (
    f"SELECT {_EVENT_EXPORT_COLUMNS} FROM plate_events "
    "ORDER BY id DESC LIMIT ?"
)
_EVENT_EXPORT_AFTER_QUERY = (
    f"SELECT {_EVENT_EXPORT_COLUMNS} FROM plate_events "
    "WHERE id < ? ORDER BY id DESC LIMIT ?"
)
EVENT_EXPORT_HEADER = (
    "ردیف",
    "پلاک",
    "اطمینان پلاک",
    "دوربین",
    "شهر محل ثبت",
    "کد ناحیه پلاک",
    "نوع خودرو",
    "رنگ خودرو",
    "اطمینان خودرو",
    "تاریخ و ساعت شمسی",
)

# These are hard ceilings, not tuning hints. They keep an authenticated export
# from monopolising a worker or producing an effectively unbounded response.
MAX_EVENT_EXPORT_ROWS = 100_000
MAX_EVENT_EXPORT_BYTES = 64 * 1024 * 1024
MAX_EVENT_EXPORT_CHUNK_BYTES = 64 * 1024
# A suspended slow-client generator retains its current fetched page. Keep
# this hard ceiling small because each hostile SQLite cell may consume a
# bounded ~16 KiB prefix before Python truncates it to characters.
MAX_EVENT_EXPORT_BATCH_SIZE = 16
MAX_EVENT_EXPORT_CELL_CHARS = 4096

_MIN_CHUNK_BYTES = 64
_FORMULA_SIGILS = frozenset("=+-@")
_TRUNCATED_CELL_SUFFIX = "…[کوتاه‌شده]"


def _starts_with_whitespace_or_control(text: str) -> bool:
    if not text:
        return False
    first = text[0]
    return first.isspace() or unicodedata.category(first).startswith("C")


def csv_safe_cell(value) -> str:
    """Return a cell that spreadsheet programs cannot evaluate as a formula.

    Formula parsing is not limited to a literal ``=`` in byte zero. Excel and
    similar programs also recognise ``+``, ``-`` and ``@`` and may ignore a
    leading whitespace/control character. Prefix all of those cases with the
    spreadsheet text marker. This function intentionally accepts every Python
    value because SQLite's dynamic typing permits text in a ``REAL`` column.
    """

    text = "" if value is None else str(value)
    if (
        (text and text[0] in _FORMULA_SIGILS)
        or _starts_with_whitespace_or_control(text)
    ):
        return "'" + text
    return text


def _bounded_safe_cell(value) -> str:
    text = csv_safe_cell(value)
    if len(text) <= MAX_EVENT_EXPORT_CELL_CHARS:
        return text
    keep = MAX_EVENT_EXPORT_CELL_CHARS - len(_TRUNCATED_CELL_SUFFIX)
    return text[:keep] + _TRUNCATED_CELL_SUFFIX


def _csv_row(values) -> str:
    buffer = StringIO(newline="")
    csv.writer(buffer).writerow(
        tuple(_bounded_safe_cell(value) for value in values)
    )
    return buffer.getvalue()


def _sql_cell_text(value):
    """Decode the bounded BLOB prefix selected by SQLite.

    SQLite's text ``substr`` treats an embedded NUL as an end marker. Selecting
    a byte-bounded BLOB prefix both keeps hostile multi-megabyte cells out of
    Python and preserves control bytes so the formula-injection guard can see
    them. Four bytes per requested character covers valid UTF-8; replacement
    decoding is a fail-safe for a corrupt database value.
    """

    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    return value


def _event_row(row, datetime_formatter) -> str:
    # Apply the same defence to every column, including SQLite values declared
    # REAL and the result returned by the caller's date formatter.
    return _csv_row((
        row["id"],
        _sql_cell_text(row["plate_text"]),
        _sql_cell_text(row["confidence"]),
        _sql_cell_text(row["camera_name"]),
        _sql_cell_text(row["city"]),
        _sql_cell_text(row["plate_region"]),
        _sql_cell_text(row["vehicle_type"]),
        _sql_cell_text(row["vehicle_color"]),
        _sql_cell_text(row["vehicle_confidence"]),
        datetime_formatter(_sql_cell_text(row["created_at"])),
    ))


def _limit_notice(max_rows: int, max_bytes: int) -> str:
    message = (
        "[خروجی محدود شد: حداکثر "
        f"{max_rows} ردیف یا {max_bytes} بایت]"
    )
    return _csv_row(
        ("#LIMIT", message, "", "", "", "", "", "", "", "")
    )


def _bounded_int(value, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _fetch_page(connect, *, after_id, limit: int):
    """Fetch one page and close both cursor and connection before returning."""

    connection = connect()
    cursor = None
    try:
        if after_id is None:
            cursor = connection.execute(EVENT_EXPORT_QUERY, (limit,))
        else:
            cursor = connection.execute(
                _EVENT_EXPORT_AFTER_QUERY,
                (after_id, limit),
            )
        return cursor.fetchall()
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def _split_utf8(text: str, max_bytes: int):
    """Split text without breaking Unicode while enforcing a byte ceiling."""

    encoded = text.encode("utf-8")
    start = 0
    while start < len(encoded):
        end = min(start + max_bytes, len(encoded))
        if end < len(encoded):
            while end > start and encoded[end] & 0xC0 == 0x80:
                end -= 1
        # UTF-8 code points are at most four bytes and max_bytes is >= 64.
        if end == start:  # pragma: no cover - defensive for future constants
            end = min(start + max_bytes, len(encoded))
            while end < len(encoded) and encoded[end] & 0xC0 == 0x80:
                end += 1
        yield encoded[start:end].decode("utf-8")
        start = end


class _ChunkBuffer:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.parts: list[str] = []
        self.size = 0

    def add(self, text: str):
        ready: list[str] = []
        for piece in _split_utf8(text, self.max_bytes):
            piece_size = len(piece.encode("utf-8"))
            if self.parts and self.size + piece_size > self.max_bytes:
                ready.append("".join(self.parts))
                self.parts = []
                self.size = 0
            if piece_size == self.max_bytes:
                if self.parts:
                    ready.append("".join(self.parts))
                    self.parts = []
                    self.size = 0
                ready.append(piece)
            else:
                self.parts.append(piece)
                self.size += piece_size
        return ready

    def flush(self):
        if not self.parts:
            return None
        payload = "".join(self.parts)
        self.parts = []
        self.size = 0
        return payload


def iter_event_csv(
    connect,
    datetime_formatter,
    *,
    batch_size=16,
    max_rows=MAX_EVENT_EXPORT_ROWS,
    max_bytes=MAX_EVENT_EXPORT_BYTES,
    max_chunk_bytes=MAX_EVENT_EXPORT_CHUNK_BYTES,
):
    """Yield a bounded UTF-8-BOM CSV without retaining an SQLite snapshot.

    Each keyset page is fully fetched and its connection is closed before any
    bytes from that page are yielded. A slow downloader therefore cannot pin a
    WAL read transaction or prevent checkpoint/truncation. If either hard
    export ceiling is reached, the final CSV record explicitly reports it.
    """

    batch_size = _bounded_int(
        batch_size,
        MAX_EVENT_EXPORT_BATCH_SIZE,
        minimum=1,
        maximum=MAX_EVENT_EXPORT_BATCH_SIZE,
    )
    max_rows = _bounded_int(
        max_rows,
        MAX_EVENT_EXPORT_ROWS,
        minimum=1,
        maximum=MAX_EVENT_EXPORT_ROWS,
    )
    max_bytes = _bounded_int(
        max_bytes,
        MAX_EVENT_EXPORT_BYTES,
        minimum=1,
        maximum=MAX_EVENT_EXPORT_BYTES,
    )
    max_chunk_bytes = _bounded_int(
        max_chunk_bytes,
        MAX_EVENT_EXPORT_CHUNK_BYTES,
        minimum=_MIN_CHUNK_BYTES,
        maximum=MAX_EVENT_EXPORT_CHUNK_BYTES,
    )

    header = "\ufeff" + _csv_row(EVENT_EXPORT_HEADER)
    notice = _limit_notice(max_rows, max_bytes)
    header_size = len(header.encode("utf-8"))
    notice_size = len(notice.encode("utf-8"))
    # Even a deliberately tiny caller-provided limit must leave room for a
    # valid header and a clear limit record.
    max_bytes = max(max_bytes, header_size + notice_size)

    chunks = _ChunkBuffer(max_chunk_bytes)
    for ready in chunks.add(header):
        yield ready
    bytes_used = header_size
    emitted_rows = 0
    after_id = None

    while True:
        remaining_rows = max_rows - emitted_rows
        page_size = min(batch_size, remaining_rows)
        page = _fetch_page(connect, after_id=after_id, limit=page_size + 1)
        has_more = len(page) > page_size
        rows = page[:page_size]
        materialized = [
            (row["id"], _event_row(row, datetime_formatter))
            for row in rows
        ]
        row_count = len(materialized)
        # Do not retain sqlite3.Row/BLOB page objects while a slow network
        # consumer suspends this generator at a yield point.
        del rows
        del page
        truncated = False

        for index, (row_id, payload) in enumerate(materialized):
            payload_size = len(payload.encode("utf-8"))
            more_after = index + 1 < row_count or has_more
            byte_ceiling = max_bytes - notice_size if more_after else max_bytes
            if bytes_used + payload_size > byte_ceiling:
                truncated = True
                break
            for ready in chunks.add(payload):
                yield ready
            bytes_used += payload_size
            emitted_rows += 1
            after_id = row_id

        if truncated or (emitted_rows >= max_rows and has_more):
            for ready in chunks.add(notice):
                yield ready
            final = chunks.flush()
            if final:
                yield final
            return

        # Flushing at page boundaries keeps response latency bounded, while the
        # corresponding DB connection has already been closed by _fetch_page.
        final = chunks.flush()
        if final:
            yield final

        if not row_count or not has_more:
            return
