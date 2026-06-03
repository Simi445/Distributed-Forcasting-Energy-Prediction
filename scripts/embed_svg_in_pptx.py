from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import shutil
import tempfile
import xml.etree.ElementTree as ET

from pptx import Presentation
from pptx.util import Inches


NS = {
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix if prefix not in {"ct", "rel"} else "", uri)


def make_base_pptx(path: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)
    prs.slides.add_slide(prs.slide_layouts[6])
    prs.save(path)


def add_content_type(root: ET.Element) -> None:
    exists = any(
        child.attrib.get("Extension") == "svg"
        for child in root.findall("{http://schemas.openxmlformats.org/package/2006/content-types}Default")
    )
    if not exists:
        ET.SubElement(
            root,
            "{http://schemas.openxmlformats.org/package/2006/content-types}Default",
            {"Extension": "svg", "ContentType": "image/svg+xml"},
        )


def next_rel_id(root: ET.Element) -> str:
    max_id = 0
    for rel in root:
        rid = rel.attrib.get("Id", "")
        if rid.startswith("rId") and rid[3:].isdigit():
            max_id = max(max_id, int(rid[3:]))
    return f"rId{max_id + 1}"


def add_svg_relationship(root: ET.Element) -> str:
    rid = next_rel_id(root)
    ET.SubElement(
        root,
        "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship",
        {
            "Id": rid,
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            "Target": "../media/forecasting_pipeline.svg",
        },
    )
    return rid


def add_pic(slide_root: ET.Element, rid: str) -> None:
    ns_p = NS["p"]
    ns_a = NS["a"]
    ns_r = NS["r"]
    sp_tree = slide_root.find(f".//{{{ns_p}}}spTree")

    pic = ET.fromstring(
        f"""
        <p:pic xmlns:p="{ns_p}" xmlns:a="{ns_a}" xmlns:r="{ns_r}">
          <p:nvPicPr>
            <p:cNvPr id="4" name="Forecasting Pipeline SVG"/>
            <p:cNvPicPr/>
            <p:nvPr/>
          </p:nvPicPr>
          <p:blipFill>
            <a:blip r:embed="{rid}"/>
            <a:stretch><a:fillRect/></a:stretch>
          </p:blipFill>
          <p:spPr>
            <a:xfrm>
              <a:off x="0" y="0"/>
              <a:ext cx="12192000" cy="6858000"/>
            </a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
          </p:spPr>
        </p:pic>
        """
    )
    sp_tree.append(pic)


def embed_svg(svg_path: Path, output_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        base = tmp / "base.pptx"
        make_base_pptx(base)
        unzip_dir = tmp / "pptx"
        unzip_dir.mkdir()

        with ZipFile(base, "r") as zf:
            zf.extractall(unzip_dir)

        media_dir = unzip_dir / "ppt" / "media"
        media_dir.mkdir(exist_ok=True)
        shutil.copyfile(svg_path, media_dir / "forecasting_pipeline.svg")

        content_types = unzip_dir / "[Content_Types].xml"
        ct_root = ET.parse(content_types)
        add_content_type(ct_root.getroot())
        ct_root.write(content_types, encoding="utf-8", xml_declaration=True)

        rels_path = unzip_dir / "ppt" / "slides" / "_rels" / "slide1.xml.rels"
        rel_tree = ET.parse(rels_path)
        rid = add_svg_relationship(rel_tree.getroot())
        rel_tree.write(rels_path, encoding="utf-8", xml_declaration=True)

        slide_path = unzip_dir / "ppt" / "slides" / "slide1.xml"
        slide_tree = ET.parse(slide_path)
        add_pic(slide_tree.getroot(), rid)
        slide_tree.write(slide_path, encoding="utf-8", xml_declaration=True)

        if output_path.exists():
            output_path.unlink()
        with ZipFile(output_path, "w", ZIP_DEFLATED) as zf:
            for file in unzip_dir.rglob("*"):
                zf.write(file, file.relative_to(unzip_dir))


def main():
    root = Path(__file__).resolve().parents[1]
    embed_svg(
        root / "exports/presentation_assets/Forecasting-Pipeline.svg",
        root / "exports/presentation_assets/Forecasting-Pipeline-embedded-svg.pptx",
    )


if __name__ == "__main__":
    main()
