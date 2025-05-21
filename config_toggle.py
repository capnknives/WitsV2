import yaml
import os

CONFIG_PATH = "wits_config.yaml"

PROFILES = {
    "safe": "wits_profile_safe.yaml",
    "test": "wits_profile_test.yaml"
}

def switch_profile(mode):
    if mode not in PROFILES:
        raise ValueError(f"Invalid mode '{mode}'. Use 'safe' or 'test'.")

    profile_path = PROFILES[mode]
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Profile file '{profile_path}' not found.")

    with open(profile_path, 'r') as src:
        config_data = yaml.safe_load(src)

    with open(CONFIG_PATH, 'w') as dst:
        yaml.dump(config_data, dst, sort_keys=False)

    print(f"[WITS] Switched to '{mode}' profile.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Switch WITS-NEXUS personality profile.")
    parser.add_argument("mode", choices=["safe", "test"], help="Profile mode to activate")
    args = parser.parse_args()

    switch_profile(args.mode)