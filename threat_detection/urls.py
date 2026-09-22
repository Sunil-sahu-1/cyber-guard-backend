from django.urls import path

from .views import (
ThreatListCreateView,
ThreatDetailView,
ThreatAnalyzeView,
ThreatDeleteView,
MalwareAnalyzeView,
)

urlpatterns = [


path(
    "",
    ThreatListCreateView.as_view(),
    name="threat-list-create",
),

path(
    "analyze/",
    ThreatAnalyzeView.as_view(),
    name="threat-analyze",
),

path(
    "malware/analyze/",
    MalwareAnalyzeView.as_view(),
    name="malware-analyze",
),

path(
    "<int:pk>/",
    ThreatDetailView.as_view(),
    name="threat-detail",
),

path(
    "<int:pk>/delete/",
    ThreatDeleteView.as_view(),
    name="threat-delete",
),


]
