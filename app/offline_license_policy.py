from __future__ import annotations

import re
from functools import wraps

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRouter

_PATCH_MARKER = "_bcvision_offline_license_policy"
_ONLINE_CARD = re.compile(
    r"<div class='card'><h3>فعال‌سازی آنلاین</h3>.*?</form></div>",
    re.DOTALL,
)


def _offline_page(endpoint):
    if getattr(endpoint, _PATCH_MARKER, False):
        return endpoint

    @wraps(endpoint)
    def wrapper(*args, **kwargs):
        response = endpoint(*args, **kwargs)
        if not isinstance(response, HTMLResponse):
            return response
        text = response.body.decode("utf-8")
        text = text.replace(
            "فعال‌سازی امن آنلاین یا آفلاین BC Vision",
            "فعال‌سازی کاملاً آفلاین و محلی BC Vision",
        )
        text = text.replace(
            "محتوای فایل license.json",
            "محتوای فایل license.dat",
        )
        text = _ONLINE_CARD.sub("", text)
        response.body = text.encode("utf-8")
        response.headers["content-length"] = str(len(response.body))
        return response

    setattr(wrapper, _PATCH_MARKER, True)
    return wrapper


def _patch_router_type(router_type) -> None:
    current = router_type.add_api_route
    if getattr(current, _PATCH_MARKER, False):
        return

    @wraps(current)
    def add_api_route(self, path, endpoint, *args, **kwargs):
        methods = {
            str(method).upper()
            for method in (kwargs.get("methods") or [])
        }
        if path == "/license/online":
            # The compatibility function remains importable for old callers,
            # but no HTTP route is exposed and no network activation exists.
            return None
        if path == "/license" and (not methods or "GET" in methods):
            endpoint = _offline_page(endpoint)
        return current(self, path, endpoint, *args, **kwargs)

    setattr(add_api_route, _PATCH_MARKER, True)
    router_type.add_api_route = add_api_route


def install_offline_license_policy() -> None:
    _patch_router_type(FastAPI)
    _patch_router_type(APIRouter)


install_offline_license_policy()
