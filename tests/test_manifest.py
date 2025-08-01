import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RAGModule.rag import RAGHandler


def test_needs_rebuild_when_new_file_added(tmp_path):
    rag_dir = tmp_path / "rag"
    rag_dir.mkdir()
    file1 = rag_dir / "a.txt"
    file1.write_text("hello")

    persist = tmp_path / "persist"
    handler = RAGHandler(rag_files_path=str(rag_dir), persist_directory=str(persist))

    manifest = handler._scan_manifest()
    handler._save_manifest(manifest)

    # Add new file after manifest saved
    file2 = rag_dir / "b.txt"
    file2.write_text("world")

    assert handler._needs_rebuild() is True

