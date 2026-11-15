"""Evidence labels shared by search results and artifact occurrences."""
import re

SOURCE_LABELS = {
    "body": "正文描述", "caption": "图注", "table": "表格",
    "footnote": "脚注", "heading": "标题", "image": "器物切片", "unknown": "待判定",
}


def numbered_caption(text):
    return re.match(r"^\s*\d+[.．、]\s*[^（()\n]{1,60}[（(]", text or "")


def source_type(block_type=None, role=None, *, caption_link=False, text=""):
    if caption_link or block_type == "caption" or role == "captioned":
        return "caption"
    if block_type == "table" or role == "tabulated":
        return "table"
    if block_type == "footnote":
        return "footnote"
    if (numbered_caption(text) and "。" not in text
            and re.search(r"[（(][^）)\n]*[∶︰：:][^）)\n]*[）)]", text)):
        return "caption"
    if block_type == "heading":
        return "heading"
    if block_type in {"paragraph", "text", "body"}:
        return "body"
    return "unknown"
