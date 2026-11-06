import pytest
from src.alto.parser import AltoIngestor
from src.ingest.digital_pdf_ingestor import DigitalPdfIngestor as PDF, ExtractedLine

@pytest.mark.parametrize("sep", [":", "：", "∶", "︰"])
def test_plate_caption_preserves_original_specimen_code(sep):
    code = f"T19⑥{sep}301"
    assert AltoIngestor.extract_artifact_codes(f"5. 青白釉碗B型（{code}）") == [code]

def test_first_page_of_plate_excerpt_needs_positive_evidence():
    header = ExtractedLine("图版2.2.1", 50, 45, 100, 12, 10)
    caption = ExtractedLine("5. 青白釉碗B型（T19⑥︰301）", 50, 450, 100, 12, 10)
    classify = PDF._detect_page_number_and_domain_hint
    assert classify(0, [caption, header], 800) == ("图版2.2.1", "plate")
    assert classify(0, [header], 800) == (None, "cover")
