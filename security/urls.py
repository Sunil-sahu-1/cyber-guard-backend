from django.urls import path

from .views import (
    BrowserPrivacyConnectView,
    BrowserPrivacyPairView,
    BrowserPrivacyScanCreateView,
    BrowserPrivacyScanListView,
)

urlpatterns = [
    path("pair/", BrowserPrivacyPairView.as_view(), name="browser-privacy-pair"),
    path("connect/", BrowserPrivacyConnectView.as_view(), name="browser-privacy-connect"),
    path("scans/", BrowserPrivacyScanCreateView.as_view(), name="browser-privacy-scan-create"),
    path("scans/history/", BrowserPrivacyScanListView.as_view(), name="browser-privacy-scan-history"),
]
