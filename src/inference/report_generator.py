from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
)


# =========================================================
# SAVE SCREENING REPORT
# =========================================================

def save_screening_report(report, image_path):

    image_path = Path(image_path)

    # -----------------------------------------------------
    # Project root
    # -----------------------------------------------------

    project_root = (
        image_path.resolve()
        .parents[2]
    )

    output_dir = (
        project_root
        / "outputs"
        / "reports"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Report filename
    # -----------------------------------------------------

    report_file = (
        output_dir
        / f"{image_path.stem}_screening_report.pdf"
    )
    # =========================================================
# EXTRACT REPORT DATA
# =========================================================

    image_quality = report.get("image_quality", {})
    original_quality_data = image_quality.get("original", {})
    final_quality_data = image_quality.get("final", {})

    classification = report.get("classification", {})
    explainability = report.get("explainability", {})

    user = report.get("user", {})

    user_name = user.get(
        "name",
        "Unknown User"
    )

    user_email = user.get(
        "email",
        "Not available"
    )
    original_quality = float(
        original_quality_data.get("quality_score", 0)
    )

    final_quality = float(
        final_quality_data.get("quality_score", 0)
    )

    original_status = original_quality_data.get(
        "status",
        "UNKNOWN"
    )

    quality_status = final_quality_data.get(
        "status",
    "UNKNOWN"
    )

    classification_source = classification.get(
        "source",
        "ORIGINAL"
)

    referable_probability = float(
        classification.get(
            "referable_probability",
            0
        )
    )

    decision = classification.get(
        "decision",
        "UNKNOWN"
    )

    gradcam_path = explainability.get(
        "file"
    )

    # =====================================================
    # DOCUMENT
    # =====================================================

    document = SimpleDocTemplate(

        str(report_file),

        pagesize=A4,

        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    # =====================================================
    # STYLES
    # =====================================================

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(

        "ReportTitle",

        parent=styles["Title"],

        fontSize=21,

        leading=25,

        alignment=TA_CENTER,

        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(

        "Subtitle",

        parent=styles["Normal"],

        fontSize=10,

        leading=14,

        alignment=TA_CENTER,

        textColor=colors.HexColor("#64748B"),

        spaceAfter=14,
    )

    section_style = ParagraphStyle(

        "Section",

        parent=styles["Heading2"],

        fontSize=13,

        leading=17,

        spaceBefore=10,

        spaceAfter=8,

        textColor=colors.HexColor("#172033"),
    )

    normal_style = ParagraphStyle(

        "NormalReport",

        parent=styles["Normal"],

        fontSize=9.5,

        leading=14,
    )

    small_style = ParagraphStyle(

        "Small",

        parent=styles["Normal"],

        fontSize=8,

        leading=11,

        textColor=colors.HexColor("#64748B"),
    )

    result_style = ParagraphStyle(

        "Result",

        parent=styles["Heading1"],

        fontSize=22,

        leading=26,

        alignment=TA_CENTER,

        textColor=colors.HexColor("#172033"),

        spaceAfter=6,
    )

    probability_style = ParagraphStyle(

        "Probability",

        parent=styles["Normal"],

        fontSize=13,

        leading=18,

        alignment=TA_CENTER,

        textColor=colors.HexColor("#475569"),
    )

    # =====================================================
    # STORY
    # =====================================================

    story = []

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "DIABETIC RETINOPATHY AI",
            title_style
        )
    )
    story.append(
    Spacer(1, 12)
    )

    story.append(
        Paragraph(
            "USER INFORMATION",
            section_style  
        )
)

    user_data = [
        ["Name", user_name],
        ["Email", user_email],
        ["Screening ID", image_path.stem],
        [
            "Screening Date",
            datetime.now().strftime(
                "%d %B %Y, %H:%M"
                )
        ],
    ]

    user_table = Table(
        user_data,
        colWidths=[130, 350]
    )

    user_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F5F1")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#087F5B")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5E5DF")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ])
    )

    story.append(user_table)
    story.append(
        Paragraph(
            "Explainable AI-assisted Fundus Screening Report",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Screening ID:</b> "
            f"{image_path.stem}",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Date:</b> "
            f"{datetime.now().strftime('%d %B %Y, %H:%M')}",
            normal_style
        )
    )

    story.append(
        Spacer(
            1,
            8
        )
    )

    # =====================================================
    # SCREENING RESULT
    # =====================================================

    story.append(
        Paragraph(
            "SCREENING RESULT",
            section_style
        )
    )

    result_table = Table(

        [
            [
                Paragraph(
                    f"<b>{decision}</b>",
                    result_style
                )
            ],

            [
                Paragraph(
                    "Referable probability: "
                    f"<b>{referable_probability:.2%}</b>",
                    probability_style
                )
            ],
        ],

        colWidths=[
            170 * mm
        ],
    )

    result_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#F8FAFC")
                ),

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.8,
                    colors.HexColor("#CBD5E1")
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    12
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    12
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
            ]
        )
    )

    story.append(
        result_table
    )

    # =====================================================
    # IMAGE QUALITY
    # =====================================================

    story.append(
        Paragraph(
            "IMAGE QUALITY",
            section_style
        )
    )

    quality_data = [

        [
            Paragraph(
                "<b>Original Quality</b>",
                normal_style
            ),

            Paragraph(
                "<b>Final Quality</b>",
                normal_style
            ),

            Paragraph(
                "<b>Status</b>",
                normal_style
            ),
        ],

        [
            f"{original_quality:.2f}/100",

            f"{final_quality:.2f}/100",

            quality_status,
        ],

        [
            Paragraph(
                "<b>Classification Source</b>",
                normal_style
            ),

            classification_source,

            "",
        ],
    ]

    quality_table = Table(

        quality_data,

        colWidths=[
            56 * mm,
            56 * mm,
            56 * mm,
        ],
    )

    quality_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#F1F5F9")
                ),

                (
                    "BACKGROUND",
                    (0, 1),
                    (-1, -1),
                    colors.HexColor("#F8FAFC")
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#CBD5E1")
                ),

                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    9
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    9
                ),
            ]
        )
    )

    story.append(
        quality_table
    )

    # =====================================================
    # ORIGINAL FUNDUS IMAGE
    # =====================================================

    story.append(
        Paragraph(
            "FUNDUS IMAGE",
            section_style
        )
    )

    if image_path.exists():

        fundus_image = Image(
            str(image_path)
        )

        fundus_image._restrictSize(
            150 * mm,
            100 * mm
        )

        story.append(
            fundus_image
        )

        story.append(
            Spacer(
                1,
                6
            )
        )

    # =====================================================
    # GRAD-CAM
    # =====================================================

    story.append(
        Paragraph(
            "AI VISUAL EXPLANATION",
            section_style
        )

    )

    if gradcam_path:

        gradcam_file = Path(
            gradcam_path
        )

        if gradcam_file.exists():

            gradcam_image = Image(
                str(gradcam_file)
            )

            gradcam_image._restrictSize(
                170 * mm,
                100 * mm
            )

            story.append(
                gradcam_image
            )

            story.append(
                Spacer(
                    1,
                    6
                )
            )

            story.append(
                Paragraph(
                    "The highlighted regions represent "
                    "areas that influenced the AI model's "
                    "classification.",
                    small_style
                )
            )

        else:

            story.append(
                Paragraph(
                    "Grad-CAM image was not available.",
                    small_style
                )
            )

    else:

        story.append(
            Paragraph(
                "Grad-CAM image was not available.",
                small_style
            )
        )

    # =====================================================
    # INTERPRETATION
    # =====================================================

    story.append(
        Paragraph(
            "SCREENING INTERPRETATION",
            section_style
        )
    )

    interpretation = (

        f"The AI screening model assigned a "
        f"{referable_probability:.2%} probability "
        f"of referable diabetic retinopathy. "

        f"The final image quality was "
        f"{final_quality:.2f}/100 and was classified "
        f"as {quality_status}. "

        f"The classification was performed using the "
        f"{classification_source.lower()} image."
    )

    story.append(
        Paragraph(
            interpretation,
            normal_style
        )
    )

    # =====================================================
    # DISCLAIMER
    # =====================================================

    story.append(
        Spacer(
            1,
            12
        )
    )

    disclaimer_table = Table(

        [
            [
                Paragraph(
                    "<b>IMPORTANT DISCLAIMER</b><br/>"
                    "This is an AI-assisted screening "
                    "prototype and not a clinical diagnosis. "
                    "Results should not replace evaluation "
                    "by a qualified eye-care professional.",
                    small_style
                )
            ]
        ],

        colWidths=[
            170 * mm
        ],
    )

    disclaimer_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#FFF7E6")
                ),

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#E2C46D")
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    10
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    10
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    9
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    9
                ),
            ]
        )
    )

    story.append(
        disclaimer_table
    )

    # =====================================================
    # BUILD PDF
    # =====================================================

    document.build(
        story
    )

    print(
        "\nScreening PDF generated:"
    )

    print(
        report_file
    )

    return report_file