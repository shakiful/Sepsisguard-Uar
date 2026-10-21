"""Explicit local fallback runner; never represents this as Colab execution."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
import tempfile
import contextlib
import io
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("notebook")
    p.add_argument("--engine", choices=("python", "jupyter"), default="python",
                   help="Python executes ordinary code cells in process; Jupyter requires kernel sockets")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ["SEPSISGUARD_LOCAL_VALIDATION"] = "1"
    os.environ["SEPSISGUARD_PROJECT"] = str(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "runs" / "local" / (stamp + "-" + Path(args.notebook).name)
    output.parent.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(root / args.notebook, as_version=4)
    if args.engine == "python":
        # Explicitly identified fallback when this host cannot launch a kernel.
        # These notebooks use ordinary Python only, without magics or shell cells.
        namespace = {"__name__": "__main__"}
        nb.metadata["execution_environment"] = "local_in_process_python_cells; not Colab or Jupyter"
        os.chdir(root)
        count = 0
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            print(f"Executing Python cell {index}", flush=True)
            buffer = io.StringIO()
            error = None
            count += 1
            cell.execution_count = count
            try:
                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                    exec(compile(cell.source, f"{args.notebook}:cell-{index}", "exec"), namespace)
            except Exception:
                error = traceback.format_exc()
                buffer.write(error)
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=buffer.getvalue())]
            nbformat.write(nb, output)
            print(buffer.getvalue()[-2000:], flush=True)
            if error:
                raise RuntimeError(f"Python cell {index} failed; see {output}")
        print("LOCAL in-process Python cell execution saved:", output, flush=True)
        return
    def completed(cell, cell_index, execute_reply):
        nbformat.write(nb, output)
        print(f"Executed cell {cell_index}", flush=True)
        for item in cell.get("outputs", []):
            if item.output_type == "stream":
                print(item.text[-700:], flush=True)
    # IPC avoids opening a TCP listener in restricted local environments.
    manager = AsyncKernelManager(kernel_name="python3", transport="ipc",
                                 ip=str(Path(tempfile.gettempdir()) / ("sg-" + stamp)))
    client = NotebookClient(nb, timeout=1800, km=manager, kernel_name="python3",
                            resources={"metadata": {"path": str(root)}},
                            on_cell_executed=completed)
    try:
        client.execute()
    finally:
        nbformat.write(nb, output)
        print("LOCAL executed notebook:", output, flush=True)


if __name__ == "__main__":
    main()
