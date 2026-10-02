from pathlib import Path

root = Path(__file__).resolve().parent
for path in root.iterdir():
    if path.suffix in {".sh", ".sbatch", ".yaml", ".py"}:
        path.write_bytes(path.read_bytes().replace(b"\r", b""))
        print("lf", path.name)
