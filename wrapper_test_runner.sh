import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

import yaml

def parse_overrides(overrides):
    parsed = {}
    for item in overrides:
        if '=' not in item:
            raise ValueError(f"Invalid override: {item}. Must be in key=value format.")
        key, value = item.split('=', 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass  # Keep as string if not valid JSON
        parsed[key] = value
    return parsed

def apply_overrides(obj, overrides):
    for dotted_key, value in overrides.items():
        keys = dotted_key.strip('.').split('.')
        ref = obj
        for k in keys[:-1]:
            ref = ref.setdefault(k, {})
        ref[keys[-1]] = value

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(path, data):
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Dynamic sys_run launcher with config patching")
    parser.add_argument("-m", dest="original_json", help="Path to the original JSON config")
    parser.add_argument("--json-override", action="append", default=[], help="Override JSON key=value")
    parser.add_argument("--yaml-override", action="append", default=[], help="Override YAML key=value")
    parser.add_argument("--save-dir", help="Directory prefix to save the patched configs")
    parser.add_argument("--load-dir", help="Directory with existing config.json and config.yaml")
    parser.add_argument("--debug", action="store_true", help="Print configs before execution")
    parser.add_argument("--dry-run", action="store_true", help="Show the final command but do not execute it")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments to forward to sys_run.sh")

    args = parser.parse_args()

    if args.load_dir:
        config_json_path = os.path.join(args.load_dir, "config.json")
        config_yaml_path = os.path.join(args.load_dir, "config.yaml")
        if not os.path.isfile(config_json_path) or not os.path.isfile(config_yaml_path):
            sys.exit(f"Error: {args.load_dir} must contain config.json and config.yaml")
        config = load_json(config_json_path)
        yaml_data = load_yaml(config_yaml_path)
    else:
        if not os.path.isfile(args.original_json):
            sys.exit(f"Error: {args.original_json} does not exist")

        config = load_json(args.original_json)
        yaml_path = config.get("yaml_path")
        if not yaml_path or not os.path.isfile(yaml_path):
            sys.exit("Error: yaml_path not found or file missing in JSON config")

        yaml_data = load_yaml(yaml_path)

        # Apply YAML overrides
        yaml_overrides = parse_overrides(args.yaml_override)
        apply_overrides(yaml_data, yaml_overrides)

        # Save to temp YAML
        tmp_yaml = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        save_yaml(tmp_yaml.name, yaml_data)
        config["yaml_path"] = tmp_yaml.name

        # Apply JSON overrides
        json_overrides = parse_overrides(args.json_override)
        apply_overrides(config, json_overrides)

        # Save to temp JSON
        tmp_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        save_json(tmp_json.name, config)

        config_json_path = tmp_json.name
        config_yaml_path = tmp_yaml.name

        # Optionally save
        if args.save_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir = f"{args.save_dir}_{ts}"
            os.makedirs(outdir, exist_ok=True)
            shutil.copy(config_json_path, os.path.join(outdir, "config.json"))
            shutil.copy(config_yaml_path, os.path.join(outdir, "config.yaml"))
            config_json_path = os.path.join(outdir, "config.json")
            config_yaml_path = os.path.join(outdir, "config.yaml")
            print(f"Saved config files to: {outdir}")

    if args.debug:
        print("\nJSON Config:")
        print(json.dumps(load_json(config_json_path), indent=2))
        print("\nYAML Config:")
        print(yaml.dump(load_yaml(config_yaml_path), sort_keys=False))

    command = ["./sys_run.sh", "-m", config_json_path] + args.args

    if args.dry_run:
        print("\nDry run - command to be executed:")
        print(" ".join(map(shlex.quote, command)))
    else:
        subprocess.run(command)

if __name__ == "__main__":
    import shlex
    main()
