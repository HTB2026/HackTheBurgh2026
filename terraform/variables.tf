# ===============================
# Required variables
# ===============================
variable "vultr_api_key" {
  type        = string
  description = "Api key for vultr (https://console.vultr.com/user/apiaccess/)"
}
variable "vultr_ssh_pub_key" {
  type        = string
  description = "Public key"
}
# ===============================
# Terraform optional variables
# ===============================
variable "vultr_region" {
  type        = string
  description = "Region to deploy the resources to"
  default     = "lhr" # London
}
variable "vultr_plan_web" {
  type        = string
  description = "Plan for the web instance"
  default     = "vc2-4c-8gb" # 8 vCPU | 32 GB RAM | 640 GB SSD
}
variable "vultr_plan_llm" {
  type        = string
  description = "Plan for the LLM instance"
  default     = "vcg-a16-12c-120g-32vram" # 12 vCPU | 120 GB RAM | 700 GB SSD
}
variable "vultr_os_id" {
  type        = string
  description = "Id of the operating system"
  default     = "2284" # Ubuntu 24.04 LTS x6
}
variable "vultr_tag" {
  type        = string
  description = "Tag and name prefix for all resources that support it"
  default     = "HackTheBurgh2026"
}
