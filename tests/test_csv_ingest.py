import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RAGModule.rag import RAGHandler


def test_ingest_csv(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("col1,col2\n1,a\n2,b\n")
    handler = RAGHandler(rag_files_path=str(tmp_path), persist_directory=str(tmp_path/"db"))
    docs = handler.ingest()
    assert any("col1:" in d.page_content for d in docs)
