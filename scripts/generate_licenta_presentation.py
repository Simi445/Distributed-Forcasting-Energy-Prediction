import subprocess
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "Template Licenta 7.pptx"
OUTPUT = ROOT / "Template Licenta 7 - Time-Aware Energy Forecasting.pptx"
FIGS = ROOT / "thesis" / "figs"
ASSETS = ROOT / "exports" / "presentation_assets"


TITLE = "TIME-AWARE ENERGY FORECASTING"
AUTHOR_LINE = "Student        Simion-Callian Oltean\nSupervisor   S.L. Dr. Ing. Marcel Antal"


SLIDES = [
    {"title": TITLE, "kind": "title"},
    {
        "title": "Contents",
        "body": [
            "Context and motivation",
            "Objective",
            "Related work",
            "Solution",
            "Implementation details",
            "Experimental results",
            "Conclusion",
            "Bibliography",
        ],
    },
    {
        "title": "Context and Motivation",
        "sections": [
            ("Context", [
                "Buildings are major electricity consumers.",
                "Hourly demand depends on schedules, weather and past load.",
                "ASHRAE GEPIII 2019 provides meter, weather and metadata tables.",
            ]),
            ("Motivation", [
                "Forecasting must preserve chronological order.",
                "Feature engineering must avoid leakage from future observations.",
                "Global and seasonal training regimes must be compared on the same folds.",
            ]),
        ],
    },
    {
        "title": "Objective",
        "sections": [
            ("Main Objective", [
                "Design and validate a time-aware workflow for hourly building electricity forecasting.",
            ]),
            ("Secondary Objectives", [
                "Build leakage-safe feature tables from ASHRAE GEPIII 2019.",
                "Select representative buildings using lag-24 autocorrelation and data quality.",
                "Compare baselines, Random Forest, SVR and XGBoost.",
                "Interpret Random Forest predictions through feature importance.",
            ]),
        ],
    },
    {
        "title": "Related Work",
        "sections": [
            ("Forecasting Models", [
                "Tree ensembles are effective for nonlinear building load patterns.",
                "SVR can model nonlinear relationships when scaling and tuning are used.",
            ]),
            ("Validation Context", [
                "Time-series validation is required to avoid future leakage.",
                "The work forecasts aggregate building meter readings, not appliance-level NILM.",
            ]),
        ],
    },
    {
        "title": "Solution - Data and Features",
        "body": [
            "Filter ASHRAE meter readings to electricity only.",
            "Merge train, weather and building metadata by timestamp and site.",
            "Generate hourly per-building time series.",
            "Create lag, rolling, calendar, weather and nonlinear features.",
        ],
    },
    {
        "title": "Solution - Building Selection",
        "body": [
            "Compute lag-24 autocorrelation on January-June data.",
            "Choose category representatives close to median daily regularity.",
            "Rank candidates around the global median autocorrelation.",
            "Apply missing-ratio, maximum-gap and duration filters.",
        ],
        "image": "validation_acf_selected.png",
    },
    {
        "title": "Solution - Forecasting Models",
        "sections": [
            ("Baselines", ["Naive daily persistence.", "Weekly same-hour mean."]),
            ("Learning Models", [
                "Random Forest with TimeSeriesSplit grid search.",
                "SVR with RBF kernel and scaling.",
                "XGBoost gradient boosted trees.",
            ]),
        ],
    },
    {
        "title": "Solution - Time-Aware Validation",
        "body": [
            "Use expanding chronological training windows.",
            "Evaluate on fixed autumn and winter validation months.",
            "Global mode trains on all preceding observations.",
            "Seasonal mode filters only the training window to matching months.",
        ],
    },
    {
        "title": "Solution - Interpretability",
        "body": [
            "Random Forest feature importance identifies dominant predictors.",
            "Grouped permutation importance measures sensitivity by feature family.",
            "Lag and recent-load features dominate the exported experiments.",
            "Time and nonlinear hour encodings provide secondary explanatory value.",
        ],
    },
    {
        "title": "Implementation - Architecture",
        "body": [
            "The system is deployed with Docker Compose.",
            "Backend components communicate through REST APIs.",
            "Cloud storage provides dataset and grid-search artifact access.",
        ],
        "image": "deployment.pdf",
    },
    {
        "title": "Implementation - Services",
        "sections": [
            ("Services", [
                "data-cloud: dataset and grid artifact access.",
                "data-preprocess: loading, merging, features, ACF and selection.",
                "model-training: baselines, RF, SVR, XGBoost and metrics.",
                "frontend: pipeline execution and visualization.",
            ]),
        ],
    },
    {
        "title": "Implementation Details",
        "body": [
            "Pipeline state is cached inside preprocessing and training services.",
            "GridSearchCV results are reused from local or cloud storage.",
            "Prediction and preprocessing plots are returned as HTML visualizations.",
            "The frontend compares global and seasonal results per model and building.",
        ],
    },
    {
        "title": "Experimental Setup",
        "body": [
            "Dataset: ASHRAE Great Energy Predictor III 2019.",
            "Target: electricity meter readings, meter = 0.",
            "Selected buildings: Healthcare 1209, Education 1088, Public services 460.",
            "Metrics: RMSE, MAE, normalized MAE and R2.",
        ],
    },
    {
        "title": "Results - Selected Buildings",
        "body": [
            "Lag-24 autocorrelation values are around 0.72-0.74.",
            "This indicates strong same-hour daily regularity.",
            "The selected buildings provide representative validation cases.",
        ],
        "image": "validation_acf_selected.png",
    },
    {
        "title": "Results - Model Comparison",
        "body": [
            "Global Random Forest gives the strongest exported results.",
            "Autumn average: MAE 16.758, R2 = 0.8388.",
            "Winter average: MAE 13.749, R2 = 0.9403.",
            "Seasonal filtering did not generally improve accuracy.",
        ],
        "image": "validation_mae_comparison.png",
    },
    {
        "title": "Results - Forecast Trace",
        "body": [
            "Building 1209 winter predictions follow the observed daily pattern.",
            "The global Random Forest captures trend and short-term fluctuations.",
        ],
        "image": "validation_rf_1209_winter_global.png",
    },
    {
        "title": "Conclusions",
        "body": [
            "The thesis demonstrates an end-to-end, leakage-safe forecasting workflow.",
            "Learning-based models generally outperform daily persistence and weekly mean baselines.",
            "Global chronological training is more stable than seasonal filtering in the exported experiments.",
            "Main limitations: three selected buildings, autumn/winter folds, RF-centered interpretation.",
        ],
    },
    {
        "title": "Bibliography",
        "body": [
            "ASHRAE Great Energy Predictor III 2019 dataset.",
            "Hyndman and Athanasopoulos, Forecasting: Principles and Practice.",
            "Hastie, Tibshirani and Friedman, The Elements of Statistical Learning.",
            "Vapnik, The Nature of Statistical Learning Theory.",
            "Wang et al., Random Forest and ensemble learning for building energy forecasting.",
        ],
    },
]


def set_shape_text(shape, text, size=None, bold=None):
    shape.text_frame.clear()
    paragraph = shape.text_frame.paragraphs[0]
    run = paragraph.add_run()
    run.text = text
    run.font.name = "Arial"
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold


def set_title(slide, title):
    if slide.shapes and getattr(slide.shapes[0], "has_text_frame", False):
        set_shape_text(slide.shapes[0], title, size=28, bold=True)
        return
    add_text(slide, 0.5, 0.25, 8.6, 0.7, title, 28, True)


def set_slide_number(slide, number):
    found = False
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False) and shape.text_frame.text.strip().isdigit():
            set_shape_text(shape, str(number), size=12)
            found = True
    if not found:
        add_text(slide, 8.9, 6.48, 0.35, 0.25, str(number), size=12)


def clear_body(slide):
    for idx, shape in enumerate(slide.shapes):
        if not getattr(shape, "has_text_frame", False):
            continue
        text = shape.text_frame.text.strip()
        if idx == 0 or text.isdigit():
            continue
        if "Distributed System Research Laboratory" in text:
            continue
        if "Technical University" in text or "Faculty" in text:
            continue
        shape.text_frame.clear()


def add_text(slide, left, top, width, height, text, size=22, bold=False):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    run = paragraph.add_run()
    run.text = text
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)


def add_bullets(slide, bullets, left=0.65, top=1.55, width=5.9, height=4.8, size=22):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    for idx, bullet in enumerate(bullets):
        paragraph = frame.paragraphs[0] if idx == 0 else frame.add_paragraph()
        paragraph.text = bullet
        paragraph.level = 0
        paragraph.font.name = "Arial"
        paragraph.font.size = Pt(size)


def add_sections(slide, sections):
    y = 1.35
    for heading, bullets in sections:
        add_text(slide, 0.65, y, 6.4, 0.35, heading, size=22, bold=True)
        y += 0.42
        add_bullets(slide, bullets, left=0.85, top=y, width=6.2, height=1.5, size=18)
        y += 0.45 + 0.42 * max(1, len(bullets)) + 0.25


def add_image(slide, image_name):
    path = FIGS / image_name
    if path.suffix.lower() == ".pdf":
        ASSETS.mkdir(parents=True, exist_ok=True)
        converted = ASSETS / f"{path.stem}.png"
        if not converted.exists():
            subprocess.run(
                ["convert", "-density", "180", str(path), "-quality", "95", str(converted)],
                check=True,
            )
        path = converted
    if path.exists():
        slide.shapes.add_picture(str(path), Inches(4.0), Inches(2.05), width=Inches(5.0))


def populate_regular_slide(slide, spec, number):
    set_title(slide, spec["title"])
    set_slide_number(slide, number)
    clear_body(slide)
    if "sections" in spec:
        add_sections(slide, spec["sections"])
    elif spec.get("image"):
        add_bullets(slide, spec["body"], left=0.65, top=1.55, width=3.15, height=4.8, size=18)
        add_image(slide, spec["image"])
    else:
        add_bullets(slide, spec["body"])


def populate_title_slide(slide):
    for shape in slide.shapes:
        if not getattr(shape, "has_text_frame", False):
            continue
        text = shape.text_frame.text.strip()
        if text == "Title":
            set_shape_text(shape, TITLE, size=30, bold=True)
        elif text.startswith("Student"):
            set_shape_text(shape, AUTHOR_LINE, size=18)
        elif "Technical University" in text:
            set_shape_text(
                shape,
                "Technical University of Cluj-Napoca\n"
                "Faculty of Automation and Computer Science\n"
                "Department of Computer Science",
                size=14,
            )


def build():
    prs = Presentation(TEMPLATE)
    content_layout = prs.slides[3].slide_layout

    while len(prs.slides) < len(SLIDES):
        prs.slides.add_slide(content_layout)

    populate_title_slide(prs.slides[0])
    for idx, spec in enumerate(SLIDES[1:], start=2):
        populate_regular_slide(prs.slides[idx - 1], spec, idx)

    if OUTPUT.exists():
        OUTPUT.unlink()
    prs.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
