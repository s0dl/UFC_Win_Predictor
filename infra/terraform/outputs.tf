output "app_url" {
  description = "Cloud Run URL for the combined app service."
  value       = google_cloud_run_v2_service.app.uri
}
