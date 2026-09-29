# src/data/extract_zip.py
import zipfile, typer
from pathlib import Path
from rich import print
from src.config import ensure_dir

app = typer.Typer(add_completion=False)

@app.command()
def main(zip: str = typer.Option(..., help="원천 zip 경로"),
         out_dir: str = typer.Option(..., help="압축 해제 폴더")):
    zip_path = Path(zip)
    out = ensure_dir(Path(out_dir))
    assert zip_path.exists(), f"zip 없음: {zip_path}"
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out)
    print(f"[green]✅ Extracted[/green] → {out.resolve()}")

if __name__ == "__main__":
    app()
