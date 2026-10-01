import subprocess
import sys

from export_runtime import export_runtime


def test_runtime_export_contains_only_consumer_modules(tmp_path):
    output = export_runtime(tmp_path / "runtime")
    assert (output / "clippy_core" / "chat.py").exists()
    assert (output / "clippy_core" / "vectordb" / "sqlite_hybrid.py").exists()
    assert not (output / "clippy_core" / "ingest").exists()
    assert not (output / "clippy_core" / "cli.py").exists()
    assert not (output / "clippy_core" / "server.py").exists()
    assert not (output / "clippy_core" / "evaluation.py").exists()

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); "
         "import clippy_core; assert clippy_core.__file__.startswith(sys.argv[1]); "
         "from clippy_core import ChatService, ClippyConfig, SQLiteHybridStore; "
         "print(ChatService.__name__, ClippyConfig.__name__, SQLiteHybridStore.__name__)",
         str(output)],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "ChatService ClippyConfig SQLiteHybridStore"
