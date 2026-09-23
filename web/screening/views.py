from pathlib import Path
import sys
import shutil

from django.conf import settings
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Screening

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = (
    Path(settings.BASE_DIR).resolve().parent
)

SRC_PATH = PROJECT_ROOT / "src"


# =========================================================
# IMPORT AI PIPELINE
# =========================================================

if str(SRC_PATH) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_PATH)
    )


from inference.predict import run_pipeline


# =========================================================
# UPLOAD + SCREENING
# =========================================================

@login_required(login_url="/login/")
def upload_image(request):
    # =====================================================
    # GET REQUEST
    # =====================================================

    if request.method != "POST":

        return render(
            request,
            "screening/upload.html"
        )


    # =====================================================
    # GET UPLOADED IMAGE
    # =====================================================

    image = request.FILES.get(
        "image"
    )


    if not image:

        return render(
            request,
            "screening/upload.html",
            {
                "error":
                    "Please select a fundus image."
            }
        )


    # =====================================================
    # CHECK FILE TYPE
    # =====================================================

    allowed_extensions = [
        ".jpg",
        ".jpeg",
        ".png"
    ]


    extension = (
        Path(image.name)
        .suffix
        .lower()
    )


    if extension not in allowed_extensions:

        return render(
            request,
            "screening/upload.html",
            {
                "error":
                    "Please upload a JPG, JPEG or PNG image."
            }
        )


    # =====================================================
    # SAVE UPLOADED IMAGE
    # =====================================================

    upload_dir = (
        PROJECT_ROOT
        / "web"
        / "media"
        / "uploads"
    )


    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # Prevent problematic filenames
    # -----------------------------------------------------

    original_filename = Path(
        image.name
    ).name


    image_path = (
        upload_dir
        / original_filename
    )


    # -----------------------------------------------------
    # Save file
    # -----------------------------------------------------

    with open(
        image_path,
        "wb+"
    ) as destination:

        for chunk in image.chunks():

            destination.write(
                chunk
            )


    print(
        "\n" + "=" * 70
    )

    print(
        "FUNDUS IMAGE UPLOADED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nImage:"
        f"\n{image_path}"
    )


    # =====================================================
    # RUN AI PIPELINE
    # =====================================================

    try:

        print(
            "\nRunning AI screening pipeline..."
        )
        result = run_pipeline(
    image_path,
    user_info={
        "name": request.user.get_full_name() or request.user.username,
        "email": request.user.email,
    }
)


        # =================================================
        # SAFETY CHECK
        # =================================================

        if result is None:

            print(
                "\n" + "=" * 70
            )

            print(
                "AI PIPELINE RETURNED NO RESULT"
            )

            print(
                "=" * 70
            )


            return render(
                request,
                "screening/upload.html",
                {
                    "error":
                        "The screening pipeline did not return a result."
                }
            )


        # =================================================
        # ADD ORIGINAL IMAGE URL
        # =================================================

        image_url = (
            settings.MEDIA_URL
            + "uploads/"
            + original_filename
        )


        result["image_url"] = image_url


        # =================================================
        # HANDLE UNGRADEABLE IMAGE
        # =================================================

        if (
            result.get("quality_status")
            == "UNGRADEABLE"
        ):

            print(
                "\n" + "-" * 70
            )

            print(
                "IMAGE UNGRADEABLE"
            )

            print(
                "-" * 70
            )


            print(
                f"\nQuality:"
                f" {result.get('final_quality', 0):.2f}/100"
            )


            print(
                "\nRecommendation:"
            )


            print(
                result.get(
                    "recommendation",
                    "Recapture fundus image."
                )
            )


            # -------------------------------------------------
            # No classification
            # -------------------------------------------------

            result[
                "referable_probability_percent"
            ] = None


            result[
                "gradcam_url"
            ] = None


            result[
                "report_url"
            ] = None
            # -------------------------------------------------
            # Make sure these values exist
            # -------------------------------------------------

            result[
                "classification_source"
            ] = None


            result[
                "referable_probability"
            ] = None


            result[
                "gradcam_generated"
            ] = False

                        # -------------------------------------------------
            # SAVE SCREENING TO DATABASE
            # -------------------------------------------------

            Screening.objects.create(
                user=request.user,
                image=f"uploads/{original_filename}",
                quality_score=result.get("final_quality"),
                quality_status=result.get("quality_status"),
                referable_probability=None,
                decision=result.get("decision"),
                classification_source=None,
                recommendation=result.get(
                    "recommendation",
                    "Recapture fundus image."
                ),
                gradcam=None,
                report=None
            )
            # -------------------------------------------------
            # Show ungradeable result page
            # -------------------------------------------------

            return render(
                request,
                "screening/result.html",
                {
                    "result": result
                }
            )


        # =================================================
        # CONVERT PROBABILITY TO PERCENTAGE
        # =================================================

        probability = result.get(
            "referable_probability"
        )


        if probability is not None:

            result[
                "referable_probability_percent"
            ] = (
                float(probability)
                * 100
            )

        else:

            result[
                "referable_probability_percent"
            ] = None


        print(
            "\nAI pipeline completed successfully."
        )


    # =====================================================
    # AI PIPELINE ERROR
    # =====================================================

    except Exception as error:

        print(
            "\n" + "=" * 70
        )

        print(
            "AI PIPELINE ERROR"
        )

        print(
            "=" * 70
        )

        print(
            error
        )


        return render(
            request,
            "screening/upload.html",
            {
                "error":
                    f"AI screening failed: {error}"
            }
        )


    # =====================================================
    # ADD GRAD-CAM URL
    # =====================================================

    gradcam_path = result.get(
        "gradcam_path"
    )


    if gradcam_path:

        gradcam_file = Path(
            gradcam_path
        )


        if gradcam_file.exists():

            # ---------------------------------------------
            # Create Django media directory
            # ---------------------------------------------

            gradcam_dir = (
                PROJECT_ROOT
                / "web"
                / "media"
                / "gradcam"
            )


            gradcam_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            # ---------------------------------------------
            # Copy Grad-CAM
            # ---------------------------------------------

            gradcam_destination = (
                gradcam_dir
                / gradcam_file.name
            )


            shutil.copy2(
                gradcam_file,
                gradcam_destination
            )


            result[
                "gradcam_url"
            ] = (
                settings.MEDIA_URL
                + "gradcam/"
                + gradcam_file.name
            )


        else:

            result[
                "gradcam_url"
            ] = None


    else:

        result[
            "gradcam_url"
        ] = None


    # =====================================================
    # ADD SCREENING REPORT URL
    # =====================================================

    report_path = result.get(
        "report_path"
    )


    if report_path:

        report_file = Path(
            report_path
        )


        if report_file.exists():

            # ---------------------------------------------
            # Create report directory
            # ---------------------------------------------

            report_dir = (
                PROJECT_ROOT
                / "web"
                / "media"
                / "reports"
            )


            report_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            # ---------------------------------------------
            # Copy report
            # ---------------------------------------------

            report_destination = (
                report_dir
                / report_file.name
            )


            shutil.copy2(
                report_file,
                report_destination
            )


            result[
                "report_url"
            ] = (
                settings.MEDIA_URL
                + "reports/"
                + report_file.name
            )


        else:

            result[
                "report_url"
            ] = None


    else:

        result[
            "report_url"
        ] = None

    # =====================================================
    # SAVE SCREENING TO DATABASE
    # =====================================================

    gradcam_db_path = None

    if result.get("gradcam_url"):
        gradcam_db_path = result["gradcam_url"].replace(
            settings.MEDIA_URL,
            "",
            1
        )


    report_db_path = None

    if result.get("report_url"):
        report_db_path = result["report_url"].replace(
            settings.MEDIA_URL,
            "",
            1
        )


    Screening.objects.create(

        user=request.user,

        image=f"uploads/{original_filename}",

        quality_score=result.get(
            "final_quality"
        ),

        quality_status=result.get(
            "quality_status"
        ),

        referable_probability=result.get(
            "referable_probability"
        ),

        decision=result.get(
            "decision"
        ),

        classification_source=result.get(
            "classification_source"
        ),

        recommendation=result.get(
            "recommendation"
        ),

        gradcam=gradcam_db_path,

        report=report_db_path
    )
    # =====================================================
    # PRINT WEB RESULT
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "DJANGO SCREENING RESULT"
    )

    print(
        "=" * 70
    )


    print(
        f"\nImage quality:"
        f" {result.get('quality_status')}"
    )


    print(
        f"Quality score:"
        f" {result.get('final_quality', 0):.2f}/100"
    )


    # -----------------------------------------------------
    # Print probability safely
    # -----------------------------------------------------

    probability = result.get(
        "referable_probability"
    )


    if probability is not None:

        print(
            f"Referable probability:"
            f" {float(probability):.2%}"
        )

    else:

        print(
            "Referable probability: N/A"
        )


    print(
        f"Decision:"
        f" {result.get('decision')}"
    )


    print(
        f"Grad-CAM:"
        f" {result.get('gradcam_url')}"
    )


    print(
        f"Report:"
        f" {result.get('report_url')}"
    )


    # =====================================================
    # SHOW RESULT PAGE
    # =====================================================

    return render(
        request,
        "screening/result.html",
        {
            "result": result
        }
    )