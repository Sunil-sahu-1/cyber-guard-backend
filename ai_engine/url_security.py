"""Combined URL security evidence layer."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from ai_engine.google_safe_browsing import check_url_with_google
from ai_engine.trained_classifiers import analyze_phishing_url_ml
from ai_engine.url_redirect import resolve_redirect_chain


def _severity(score: float) -> str:
    if score >= 80:
        return "CRITICAL"
    if score >= 60:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    if score >= 20:
        return "LOW"
    return "SAFE"


def _clamp(score: float) -> float:
    return round(max(0.0, min(100.0, float(score))), 2)


def analyze_url_security(url: str) -> dict[str, Any]:
    original = str(url or "").strip()
    candidate = original if "://" in original else f"https://{original}"

    parsed = urlparse(candidate)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return {
            "is_valid": False,
            "risk_score": 0.0,
            "severity": "SAFE",
            "error": "Invalid HTTP/HTTPS URL.",
        }

    ml = analyze_phishing_url_ml(candidate)
    redirect = resolve_redirect_chain(candidate)

    final_url = str(redirect.get("final_url") or candidate)
    google_original = check_url_with_google(candidate)
    google_final = (
        check_url_with_google(final_url)
        if final_url != candidate
        else google_original
    )

    ml_score = float((ml or {}).get("risk_score", 0.0))

    redirect_score = 0.0
    redirect_indicators: list[str] = []

    redirect_count = int(redirect.get("redirect_count", 0) or 0)
    if redirect_count:
        redirect_score += min(25.0, redirect_count * 8.0)
        redirect_indicators.append(
            f"URL redirect chain contains {redirect_count} redirect(s)."
        )

    original_host = (urlparse(candidate).hostname or "").lower()
    final_host = (urlparse(final_url).hostname or "").lower()

    if original_host and final_host and original_host != final_host:
        redirect_score += 20.0
        redirect_indicators.append(
            f"Final destination URL: {final_url}"
        )

    matches = []
    for source, result in (
        ("original_url", google_original),
        ("final_url", google_final),
    ):
        for match in result.get("matches", []):
            if isinstance(match, dict):
                matches.append({"source": source, **match})

    google_score = 90.0 if matches else 0.0
    final_score = _clamp(max(ml_score, redirect_score, google_score))

    indicators = list((ml or {}).get("indicators", []))
    indicators.extend(redirect_indicators)

    if matches:
        indicators.append(
            "Google Safe Browsing matched a listed threat for the original or final URL."
        )
    elif google_original.get("checked"):
        indicators.append(
            "Google Safe Browsing found no listed threat for the original URL."
        )
        if final_url != candidate and google_final.get("checked"):
            indicators.append(
                "Google Safe Browsing found no listed threat for the final destination URL."
            )
    elif google_original.get("error"):
        indicators.append(
            f"Google Safe Browsing unavailable: {google_original['error']}"
        )

    return {
        "is_valid": True,
        "analysis_type": "url_security_combined",
        "risk_score": final_score,
        "severity": _severity(final_score),
        "prediction": (ml or {}).get("prediction", "GOOGLE_SAFE_BROWSING_ONLY"),
        "confidence": (ml or {}).get("confidence", 0.0),
        "indicators": list(dict.fromkeys(indicators)),
        "features": {
            "trained_model": ml or {
                "available": False,
                "reason": "Trained phishing URL model is unavailable.",
            },
            "redirect_analysis": redirect,
            "google_safe_browsing": {
                "original_url": google_original,
                "final_url": google_final,
                "matched_threats": matches,
            },
            "risk_components": {
                "trained_phishing_url_model": _clamp(ml_score),
                "redirect_security": _clamp(redirect_score),
                "google_safe_browsing": google_score,
            },
            "original_url": candidate,
            "final_url": final_url,
            "original_domain": original_host,
            "final_domain": final_host,
        },
        "recommendation": (
            "Do not open the URL until it is verified."
            if final_score >= 60
            else "Review the trained model, redirect analysis and Google Safe Browsing evidence before trusting the URL."
        ),
    }
