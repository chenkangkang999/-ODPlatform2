from django.urls import path, re_path

from . import monitor, views

urlpatterns = [
    path("api/status/", views.status),
    path("api/monitor/", monitor.monitor),
    path("api/dataset/", views.dataset),
    path("api/dataset/sample/<int:index>/", views.dataset_sample),
    path("api/experiments/", views.experiments),
    path("api/jobs/", views.job_list),
    re_path(r"^api/jobs/(?P<job_id>[a-f0-9]{32})/$", views.job_detail),
    re_path(r"^api/jobs/(?P<job_id>[a-f0-9]{32})/cancel/$", views.cancel),
    re_path(r"^api/jobs/(?P<job_id>[a-f0-9]{32})/files/(?P<name>[^/]+)$", views.job_file),
    path("api/inference/", views.inference),
    path("api/chat/", views.chat),
    re_path(r"^(?!api/)(?P<path>.*)$", views.frontend),
]
