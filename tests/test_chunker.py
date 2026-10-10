from bibo.ingestion.chunker import chunk_pages
from bibo.ingestion.parser import PageText


def test_chunk_order_overlap_and_page_provenance():
    pages = [PageText("one two three four five six seven eight nine ten", 3),
             PageText("alpha beta gamma", 4)]
    chunks = chunk_pages(pages, document_id="doc", filename="paper.pdf",
                         source_path="paper.pdf", file_hash="hash", size_words=5,
                         overlap_words=2)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].content.split()[-2:] == chunks[1].content.split()[:2]
    assert [chunk.page_number for chunk in chunks] == [3, 3, 3, 4]
    assert all(chunk.document_id == "doc" for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    first_page_words = set(" ".join(c.content for c in chunks if c.page_number == 3).split())
    assert first_page_words == set(pages[0].content.split())


def test_prefer_sentence_boundary():
    pages = [PageText("one two three four. five six seven eight nine ten")]
    chunks = chunk_pages(pages, document_id="doc", filename="a.txt", source_path="a.txt",
                         file_hash="hash", size_words=5, overlap_words=1)
    assert chunks[0].content.endswith("four.")
