terraform {
  required_providers {
    vultr = {
      source  = "vultr/vultr"
      version = "2.33.2"
    }
  }
}

# Configure the Vultr Provider
provider "vultr" {
  api_key     = var.vultr_api_key
  rate_limit  = 100
  retry_limit = 3
}

resource "vultr_vpc" "vpc" {
  description    = "HackTheBurgh2026VPC"
  region         = var.vultr_region
  v4_subnet      = "10.0.0.0"
  v4_subnet_mask = 24
}

resource "vultr_ssh_key" "ssh_key" {
  name    = "HackTheBurgh2026Key"
  ssh_key = var.vultr_ssh_pub_key
}

resource "vultr_firewall_group" "firewall_web" {
  description = "Firewall for the web instance"
}

resource "vultr_firewall_rule" "allow_22" {
  firewall_group_id = vultr_firewall_group.firewall_web.id
  protocol          = "tcp"
  ip_type           = "v4"
  subnet            = "0.0.0.0"
  subnet_size       = 0
  port              = "22"
  notes             = "Allow SSH"
}

resource "vultr_firewall_rule" "allow_80" {
  firewall_group_id = vultr_firewall_group.firewall_web.id
  protocol          = "tcp"
  ip_type           = "v4"
  subnet            = "0.0.0.0"
  subnet_size       = 0
  port              = "80"
  notes             = "Allow HTTP"
}

resource "vultr_firewall_rule" "allow_443" {
  firewall_group_id = vultr_firewall_group.firewall_web.id
  protocol          = "tcp"
  ip_type           = "v4"
  subnet            = "0.0.0.0"
  subnet_size       = 0
  port              = "443"
  notes             = "Allow HTTPs"
}

resource "vultr_instance" "web" {
  plan                = var.vultr_plan_web
  region              = var.vultr_region
  os_id               = var.vultr_os_id
  label               = "HackTheBurgh2026VPS"
  tags                = ["HackTheBurgh2026"]
  firewall_group_id   = vultr_firewall_group.firewall_web.id
  enable_ipv6         = false
  disable_public_ipv4 = false
  vpc_ids             = [vultr_vpc.vpc.id]
  backups             = "disabled"
  user_scheme         = "limited"
  ssh_key_ids      = [vultr_ssh_key.ssh_key.id]
  ddos_protection  = true
  activation_email = false
}

resource "vultr_firewall_group" "firewall_llm" {
  description = "Firewall for the LLM instance"
}

resource "vultr_firewall_rule" "allow_11434" {
  firewall_group_id = vultr_firewall_group.firewall_web.id
  protocol          = "tcp"
  ip_type           = "v4"
  subnet            = vultr_vpc.vpc.v4_subnet
  subnet_size       = 8
  port              = "11434"
  notes             = "Allow olama"
}

resource "vultr_instance" "llm" {
  plan                = var.vultr_plan_llm
  region              = var.vultr_region
  os_id               = var.vultr_os_id
  label               = "HackTheBurgh2026VPS"
  tags                = ["HackTheBurgh2026"]
  firewall_group_id   = vultr_firewall_group.firewall_llm.id
  enable_ipv6         = false
  vpc_ids             = [vultr_vpc.vpc.id]
  disable_public_ipv4 = true
  backups             = "disabled"
  user_scheme         = "limited"
  ddos_protection     = false
  activation_email    = false
  ssh_key_ids         = [vultr_ssh_key.ssh_key.id]
}
