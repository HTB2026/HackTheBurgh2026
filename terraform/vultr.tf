terraform {
  required_providers {
    vultr = {
      source  = "vultr/vultr"
      version = "2.33.2"
    }
  }
}

provider "vultr" {
  api_key     = var.vultr_api_key
  rate_limit  = 100
  retry_limit = 3
}

resource "vultr_vpc" "vpc" {
  description    = "${var.vultr_tag}VPC"
  region         = var.vultr_region
  v4_subnet      = "10.0.0.0"
  v4_subnet_mask = 24
}

resource "vultr_ssh_key" "ssh_key" {
  name    = "${var.vultr_tag}Key"
  ssh_key = var.vultr_ssh_pub_key
}

resource "vultr_nat_gateway" "nat_gateway" {
  vpc_id = vultr_vpc.vpc.id
  label  = "${var.vultr_tag}NatGateway"
  tag    = var.vultr_tag
}

resource "vultr_nat_gateway_port_forwarding_rule" "port_forwarding_rule" {
  vpc_id         = vultr_nat_gateway.nat_gateway.vpc_id
  nat_gateway_id = vultr_nat_gateway.nat_gateway.id
  name           = "${var.vultr_tag}NatPortForwardingRule"
  protocol       = "tcp"
  internal_port  = "22"
  internal_ip    = vultr_instance.llm.internal_ip
  external_port  = "22"
  enabled        = true
  description    = "Allow to SSH into the LLM VPS"
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

resource "vultr_reserved_ip" "reserved_ip" {
  label       = "${var.vultr_tag}Ipv4"
  region      = var.vultr_region
  ip_type     = "v4"
  instance_id = vultr_instance.web.id
}

resource "vultr_instance" "web" {
  plan                = var.vultr_plan_web
  region              = var.vultr_region
  os_id               = var.vultr_os_id
  label               = "${var.vultr_tag}Web"
  tags                = [var.vultr_tag]
  firewall_group_id   = vultr_firewall_group.firewall_web.id
  enable_ipv6         = false
  disable_public_ipv4 = false
  vpc_ids             = [vultr_vpc.vpc.id]
  backups             = "disabled"
  user_scheme         = "limited"
  ssh_key_ids         = [vultr_ssh_key.ssh_key.id]
  ddos_protection     = true
  activation_email    = false
  vpc_only            = false
}

resource "vultr_firewall_group" "firewall_llm" {
  description = "Firewall for the LLM instance"
}

resource "vultr_firewall_rule" "allow_11434" {
  firewall_group_id = vultr_firewall_group.firewall_llm.id
  protocol          = "tcp"
  ip_type           = "v4"
  subnet            = vultr_vpc.vpc.v4_subnet
  subnet_size       = 8
  port              = "11434"
  notes             = "Allow olama"
}

resource "vultr_firewall_rule" "allow_22_llm" {
  firewall_group_id = vultr_firewall_group.firewall_llm.id
  protocol          = "tcp"
  ip_type           = "v4"
  subnet            = vultr_vpc.vpc.v4_subnet
  subnet_size       = 8
  port              = "22"
  notes             = "Allow SSH"
}

resource "vultr_instance" "llm" {
  plan                = var.vultr_plan_llm
  region              = var.vultr_region
  os_id               = var.vultr_os_id
  label               = "${var.vultr_tag}LLM"
  tags                = [var.vultr_tag]
  firewall_group_id   = vultr_firewall_group.firewall_llm.id
  enable_ipv6         = false
  vpc_ids             = [vultr_vpc.vpc.id]
  disable_public_ipv4 = true
  backups             = "disabled"
  user_scheme         = "limited"
  ddos_protection     = false
  activation_email    = false
  ssh_key_ids         = [vultr_ssh_key.ssh_key.id]
  vpc_only            = true
}

output "web_ip" {
  value = vultr_instance.web.main_ip
}

output "llm_ip" {
  value = "${vultr_nat_gateway.nat_gateway.public_ips[0]}:${vultr_nat_gateway_port_forwarding_rule.port_forwarding_rule.external_port}"
}
