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
# ===============================
# Terraform optional variables
# ===============================
variable "iac_ad_az_tenant_id" {
  type        = string
  description = "Id of the tenant who will deploy the resources on Azure. If null, the parameter will be read from the az cli configuration"
  default     = null
}
variable "iac_ad_az_subscription_id" {
  type        = string
  description = "Id of the subscription used to deploy the resources on Azure. If null, the parameter will be read from the az cli configuration"
  default     = null
}
variable "iac_ad_network_cidr" {
  type        = string
  description = "CIDR of the VPC to create. Meaning the range of IP addresses that will be available for the instances in the VPC."
  default     = "192.168.0.0/16"
  validation {
    condition     = can(cidrnetmask(var.iac_ad_network_cidr))
    error_message = "Must be a valid ip address with mask"
  }
}
variable "iac_ad_router_subnet_cidr" {
  type        = string
  description = "CIDR of the subnet to create for the router. Meaning the range of IP addresses that will be available for the instances in the subnet."
  default     = "192.168.0.0/24"
  validation {
    condition     = can(cidrnetmask(var.iac_ad_router_subnet_cidr))
    error_message = "Must be a valid ip address with mask"
  }
}
variable "iac_ad_server_subnet_cidr" {
  type        = string
  description = "CIDR of the subnet to create for the game system. Meaning the range of IP addresses that will be available for the instances in the subnet."
  default     = "192.168.1.0/24"
  validation {
    condition     = can(cidrnetmask(var.iac_ad_server_subnet_cidr))
    error_message = "Must be a valid ip address with mask"
  }
}
variable "iac_ad_vulnbox_subnet_cidr" {
  type        = string
  description = "CIDR of the subnet to create for the vulnboxes. Meaning the range of IP addresses that will be available for the instances in the subnet."
  default     = "192.168.2.0/24"
  validation {
    condition     = can(cidrnetmask(var.iac_ad_vulnbox_subnet_cidr))
    error_message = "Must be a valid ip address with mask"
  }
}
variable "iac_ad_vulnbox_count" {
  description = "Number of vulnboxes to create"
  type        = number
  default     = 1
  validation {
    condition     = var.iac_ad_vulnbox_count > 0 && var.iac_ad_vulnbox_count <= 250
    error_message = "The number of vulnboxes must be greater than 0 and no more than 250"
  }
}
variable "iac_ad_wireguard_port" {
  description = "Port to use for the wireguard VPN"
  type        = number
  default     = 51820
  validation {
    condition     = var.iac_ad_wireguard_port >= 80 && var.iac_ad_wireguard_port <= 65535
    error_message = "The wireguard port must be between 80 and 65535"
  }
}
variable "iac_ad_server_ports" {
  description = "Ports to open on the game system"
  type        = set(number)
  default     = [80, 443]

  validation {
    condition     = length(var.iac_ad_server_ports) > 0
    error_message = "At least one port must be specified"
  }

  validation {
    condition     = alltrue([for port in var.iac_ad_server_ports : port >= 20 && port <= 65535])
    error_message = "Ports must be between 20 and 65535"
  }
}
