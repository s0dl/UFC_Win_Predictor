variable "project_id" {
  description = "Google Cloud project ID."
  type        = string
}

variable "region" {
  description = "Google Cloud region for Cloud Run."
  type        = string
  default     = "us-central1"
}

variable "image" {
  description = "Container image URL for the combined frontend/API app."
  type        = string
}

variable "service_name" {
  description = "Cloud Run service name."
  type        = string
  default     = "ufc-app"
}

variable "allow_unauthenticated" {
  description = "Whether to make the Cloud Run service public."
  type        = bool
  default     = true
}

variable "deletion_protection" {
  description = "Whether to enable deletion protection on the Cloud Run service."
  type        = bool
  default     = false
}

variable "cpu" {
  description = "CPU allocated to the app container."
  type        = string
  default     = "2"
}

variable "memory" {
  description = "Memory allocated to the app container."
  type        = string
  default     = "4Gi"
}

variable "min_instances" {
  description = "Minimum app instances."
  type        = number
  default     = 1
}

variable "max_instances" {
  description = "Maximum app instances."
  type        = number
  default     = 3
}

variable "custom_domain" {
    description = "Custom domain mapped to the Cloud Run app. Leave empty to disable."
    type        = string
    default     = ""
  }