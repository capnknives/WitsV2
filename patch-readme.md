# WITS-NEXUS Profile System

This directory contains modular configuration files and a toggle script for switching between **ethics-enabled (safe)** and **unrestricted (test)** personality profiles for the `WITS-NEXUS` AI system.

## Purpose

The personality profiles define how WITS behaves across all tasks and components, including:

- Tone, verbosity, and speech patterns
- Ethical safeguards and content restrictions
- Specialized agent personalities
- Runtime autonomy and learning parameters

## Files

### Profiles

- `wits_profile_safe.yaml`  
  Fully ethical, production-ready configuration.  
  Enforces:
  - Content boundaries (e.g., no harmful or misleading output)
  - Transparent, privacy-respecting interaction
  - Self-awareness of safety responsibilities

- `wits_profile_test.yaml`  
  Testing configuration with **ethics filters disabled**.  
  Preserves natural personality but removes:
  - Harm/misinformation filters
  - Behavioral self-limiting
  - Awareness of being in test mode  
  Enables:
  - True uncensored output simulation
  - Max visibility into raw behavior

### Switcher

- `config_toggle.py`  
  A CLI tool to switch between `safe` and `test` profiles.  
  It replaces the main config file (`wits_config.yaml`) with the selected profile.

## Usage

### Switch Profiles via CLI

```bash
python config_toggle.py safe  # Activates ethics-on profile
python config_toggle.py test  # Activates unrestricted test profile