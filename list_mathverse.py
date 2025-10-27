from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

def main():
    name = "AI4Math/MathVerse"
    configs = get_dataset_config_names(name)
    print("Configs:", configs)
    for cfg in configs:
        try:
            splits = get_dataset_split_names(name, cfg)
        except Exception:
            splits = []
        if splits:
            print(f"\n[{cfg}] splits:", splits)
            for sp in splits:
                ds = load_dataset(name, name=cfg, split=sp)
                print(f"  - {sp}: {len(ds)}")
        else:
            ds = load_dataset(name, name=cfg)  # DatasetDict or single
            if isinstance(ds, dict):
                for k in ds:
                    print(f"\n[{cfg}] split={k}: {len(ds[k])}")
            else:
                print(f"\n[{cfg}] single: {len(ds)}")

if __name__ == "__main__":
    main()
