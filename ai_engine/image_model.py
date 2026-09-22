"""
Advanced Deepfake Detection Engine.

Pipeline:

Image/Video
    ↓
Face Detection
    ↓
Face Crop
    ↓
EfficientNet
    ↓
Xception
    ↓
Vision Transformer
    ↓
Ensemble Fusion
    ↓
Deepfake Probability
    ↓
Risk Classification

Video additionally:
    ↓
Frame Sampling
    ↓
Frame-level AI inference
    ↓
Temporal consistency analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ExifTags

from ai_engine.deepfake_models import (
    get_model_ensemble,
)

from ai_engine.media_pipeline import (
    prepare_image,
    prepare_video,
)


MAX_FILE_SIZE = 20 * 1024 * 1024

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}


def _clamp(
    value: float,
) -> float:

    return round(
        max(
            0.0,
            min(
                100.0,
                float(value),
            ),
        ),
        2,
    )


def _severity(
    score: float,
) -> str:

    if score >= 80:
        return "CRITICAL"

    if score >= 60:
        return "HIGH"

    if score >= 40:
        return "MEDIUM"

    if score >= 20:
        return "LOW"

    return "SAFE"


def _prediction(
    score: float,
) -> str:

    if score >= 80:
        return (
            "LIKELY_DEEPFAKE_OR_MANIPULATED"
        )

    if score >= 60:
        return (
            "HIGH_MANIPULATION_RISK"
        )

    if score >= 40:
        return "SUSPICIOUS_MEDIA"

    if score >= 20:
        return "LOW_MANIPULATION_RISK"

    return (
        "NO_MAJOR_MANIPULATION_INDICATOR"
    )


def _confidence(
    score: float,
) -> float:

    if score >= 80:
        return 0.90

    if score >= 60:
        return 0.82

    if score >= 40:
        return 0.72

    if score >= 20:
        return 0.62

    return 0.55


def _model_score(
    result: dict[str, Any],
) -> float:

    scores = result[
        "ensemble_scores"
    ]

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def _temporal_score(
    frame_scores: list[float],
) -> float:

    if len(frame_scores) < 2:
        return 0.0

    average = sum(
        frame_scores
    ) / len(frame_scores)

    variance = sum(
        (
            score - average
        ) ** 2
        for score in frame_scores
    ) / len(frame_scores)

    # Large frame-to-frame changes
    # can be a temporal warning signal.
    volatility = min(
        30.0,
        variance ** 0.5,
    )

    return round(
        volatility,
        2,
    )


def _recommendation(
    severity: str,
) -> str:

    if severity == "CRITICAL":

        return (
            "Quarantine the media, flag the incident "
            "and perform identity verification."
        )

    if severity == "HIGH":

        return (
            "Treat the media as highly suspicious and "
            "perform additional forensic verification."
        )

    if severity == "MEDIUM":

        return (
            "Review the AI model evidence before trusting "
            "the media."
        )

    if severity == "LOW":

        return (
            "Continue monitoring; manipulation probability "
            "is currently low."
        )

    return (
        "No major manipulation indicator was detected."
    )


def analyze_image(
    file_name: str,
    file_size: int,
) -> dict[str, Any]:

    extension = Path(
        file_name
    ).suffix.lower()

    if not file_name:

        return {
            "is_valid": False,
            "prediction": "INVALID_INPUT",
            "risk_score": 0.0,
            "severity": "SAFE",
            "error": "File name is required.",
        }

    if file_size <= 0:

        return {
            "is_valid": False,
            "prediction": "INVALID_INPUT",
            "risk_score": 0.0,
            "severity": "SAFE",
            "error": "File is empty.",
        }

    if file_size > MAX_FILE_SIZE:

        return {
            "is_valid": False,
            "prediction": "INVALID_INPUT",
            "risk_score": 0.0,
            "severity": "SAFE",
            "error": "File size exceeds 20 MB.",
        }

    if extension not in IMAGE_EXTENSIONS:

        return {
            "is_valid": False,
            "prediction": "INVALID_INPUT",
            "risk_score": 0.0,
            "severity": "SAFE",
            "error": "Unsupported image format.",
        }

    # The API will provide a temporary path through
    # analyze_image_file() below.
    return {
        "is_valid": True,
        "prediction": "READY_FOR_AI_ANALYSIS",
        "risk_score": 0.0,
        "severity": "SAFE",
    }


def _extract_metadata(file_path: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "available": False,
        "status": "UNAVAILABLE",
        "format": None,
        "width": None,
        "height": None,
        "mode": None,
        "has_exif": False,
        "exif_fields": [],
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "datetime_original": None,
    }

    try:
        with Image.open(file_path) as image:
            metadata["format"] = image.format
            metadata["width"], metadata["height"] = image.size
            metadata["mode"] = image.mode

            exif = image.getexif()
            if exif:
                metadata["has_exif"] = True
                named = {}
                for key, value in exif.items():
                    name = ExifTags.TAGS.get(key, str(key))
                    named[name] = str(value)

                metadata["exif_fields"] = sorted(named.keys())[:30]
                metadata["camera_make"] = named.get("Make")
                metadata["camera_model"] = named.get("Model")
                metadata["software"] = named.get("Software")
                metadata["datetime_original"] = (
                    named.get("DateTimeOriginal")
                    or named.get("DateTime")
                )
                metadata["available"] = True
                metadata["status"] = "AVAILABLE"
            else:
                metadata["status"] = "NO_EXIF"

    except Exception as error:
        metadata["status"] = "READ_ERROR"
        metadata["error"] = str(error)

    return metadata


def _analyze_visual_artifacts(file_path: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "artifact_score": 0.0,
        "signals": [],
        "metrics": {},
    }

    image = cv2.imread(str(file_path))
    if image is None:
        result["error"] = "Unable to decode image for visual analysis."
        return result

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).mean())

    # These are forensic heuristics only. They are supporting signals,
    # not proof that an image is synthetic or manipulated.
    signals: list[str] = []
    artifact_score = 0.0

    if laplacian_variance < 35:
        artifact_score += 20
        signals.append("Image has unusually smooth/low-detail regions.")
    elif laplacian_variance > 2500:
        artifact_score += 10
        signals.append("Image contains unusually strong high-frequency edges.")

    if edge_density > 0.22:
        artifact_score += 10
        signals.append("High edge density detected.")

    if brightness_std < 25:
        artifact_score += 10
        signals.append("Low local lighting variation detected.")

    if width < 256 or height < 256:
        signals.append("Low-resolution input can reduce detector reliability.")

    result.update(
        {
            "available": True,
            "artifact_score": round(min(100.0, artifact_score), 2),
            "signals": signals,
            "metrics": {
                "width": width,
                "height": height,
                "laplacian_variance": round(laplacian_variance, 2),
                "brightness_mean": round(brightness_mean, 2),
                "brightness_std": round(brightness_std, 2),
                "edge_density": round(edge_density, 4),
            },
            "note": (
                "Visual artifact metrics are supporting forensic signals "
                "and are not used as a standalone deepfake verdict."
            ),
        }
    )
    return result


def _build_face_detection(
    pipeline: dict[str, Any],
) -> dict[str, Any]:
    faces = pipeline.get("faces", [])
    return {
        "detected": bool(faces),
        "face_count": len(faces),
        "faces": [
            {
                "x": int(face[0]),
                "y": int(face[1]),
                "width": int(face[2]),
                "height": int(face[3]),
            }
            for face in faces
        ],
        "method": "OpenCV Haar Cascade",
        "note": (
            "Face detection identifies visible faces; it does not by itself "
            "determine whether a face is genuine or manipulated."
        ),
    }


def _analyze_no_face_result(
    face_detection: dict[str, Any],
    visual_artifacts: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "analysis_type": "trained_efficientnet_b0_deepfake_image",
        "is_valid": True,
        "prediction": "NO_FACE_DETECTED",
        "risk_score": 0.0,
        "severity": "SAFE",
        "confidence": 0.50,
        "indicators": [
            "No detectable face was found for face-based deepfake analysis."
        ],
        "face_detection": face_detection,
        "visual_artifacts": visual_artifacts,
        "metadata": metadata,
        "features": {
            "face_detected": False,
            "face_count": 0,
        },
        "recommendation": (
            "No face was available for the trained face-based deepfake model. "
            "Review visual and metadata signals separately."
        ),
    }


def analyze_image_file(
    file_path: str,
    file_name: str,
    file_size: int,
) -> dict[str, Any]:

    basic = analyze_image(
        file_name,
        file_size,
    )

    if not basic["is_valid"]:
        return basic

    pipeline = prepare_image(file_path)

    if not pipeline["success"]:
        return {
            "is_valid": False,
            "prediction": "MEDIA_READ_ERROR",
            "risk_score": 0.0,
            "severity": "SAFE",
            "error": pipeline["error"],
        }

    face_detection = _build_face_detection(pipeline)
    visual_artifacts = _analyze_visual_artifacts(file_path)
    metadata = _extract_metadata(file_path)

    if not pipeline["tensors"]:
        return _analyze_no_face_result(
            face_detection,
            visual_artifacts,
            metadata,
        )

    import torch

    batch = torch.stack(pipeline["tensors"])
    ensemble = get_model_ensemble()
    model_result = ensemble.predict(batch)

    score = _clamp(_model_score(model_result))
    severity = _severity(score)

    indicators = [
        f"Trained EfficientNet-B0 fake probability: {score:.2f}/100."
    ]

    if face_detection["face_count"] > 1:
        indicators.append(
            f"{face_detection['face_count']} faces detected and analyzed."
        )

    if visual_artifacts.get("signals"):
        indicators.extend(
            f"Visual heuristic: {item}"
            for item in visual_artifacts["signals"]
        )

    if metadata.get("status") == "NO_EXIF":
        indicators.append(
            "No EXIF metadata was present in the uploaded image."
        )

    if score >= 60:
        indicators.append(
            "The trained EfficientNet-B0 model detected elevated fake-image probability."
        )
    elif score >= 40:
        indicators.append(
            "The trained EfficientNet-B0 model produced a suspicious/intermediate score; "
            "this is not a definitive deepfake verdict."
        )

    return {
        "analysis_type": "trained_efficientnet_b0_deepfake_image",
        "is_valid": True,
        "risk_score": score,
        "severity": severity,
        "prediction": _prediction(score),
        "confidence": _confidence(score),
        "indicators": indicators,
        "face_detection": face_detection,
        "visual_artifacts": visual_artifacts,
        "metadata": metadata,
        "features": {
            "face_detected": True,
            "face_count": pipeline["face_count"],
            "models_used": model_result["models_used"],
            "fine_tuned_models": model_result["fine_tuned_models"],
            "model_predictions": model_result["model_predictions"],
            "ensemble_score": score,
            "device": model_result["device"],
        },
        "recommendation": _recommendation(severity),
    }


def analyze_video_file(
    file_path: str,
    file_name: str,
    file_size: int,
    max_frames: int = 16,
) -> dict[str, Any]:
    return {
        "is_valid": False,
        "prediction": "VIDEO_MODEL_NOT_AVAILABLE",
        "risk_score": 0.0,
        "severity": "SAFE",
        "error": "Deepfake video detection is disabled because no video-specific trained model is currently included in Cyber Guard.",
    }


    extension = Path(
        file_name
    ).suffix.lower()

    if file_size <= 0:
        return {
            "is_valid": False,
            "prediction": "INVALID_INPUT",
            "risk_score": 0.0,
            "severity": "SAFE",
            "error": "Video is empty.",
        }

    if file_size > MAX_FILE_SIZE:
        return {
            "is_valid": False,
            "prediction": "INVALID_INPUT",
            "risk_score": 0.0,
            "severity": "SAFE",
            "error": "Video exceeds 20 MB.",
        }

    if extension not in VIDEO_EXTENSIONS:
        return {
            "is_valid": False,
            "prediction": "INVALID_INPUT",
            "risk_score": 0.0,
            "severity": "SAFE",
            "error": "Unsupported video format.",
        }

    pipeline = prepare_video(
        file_path,
        max_frames=max_frames,
    )

    if not pipeline["tensors"]:

        return {
            "analysis_type": "advanced_deepfake_video",
            "is_valid": True,
            "risk_score": 0.0,
            "severity": "SAFE",
            "prediction": "NO_FACE_DETECTED",
            "confidence": 0.50,
            "indicators": [
                "No detectable face was found in sampled frames."
            ],
            "features": {
                "frames_analyzed": pipeline[
                    "frames_analyzed"
                ],
                "face_frames": pipeline[
                    "face_frames"
                ],
                "face_detected": False,
            },
            "recommendation": (
                "No face was available for deepfake analysis."
            ),
        }

    import torch

    batch = torch.stack(
        pipeline["tensors"]
    )

    ensemble = get_model_ensemble()

    model_result = ensemble.predict(
        batch
    )

    frame_scores = (
        model_result[
            "ensemble_scores"
        ]
    )

    spatial_score = (
        sum(frame_scores)
        / len(frame_scores)
    )

    temporal_score = _temporal_score(
        frame_scores
    )

    # Temporal signal has a lower weight because
    # volatility alone is not proof of a deepfake.
    final_score = _clamp(
        (
            spatial_score * 0.85
            + temporal_score * 0.15
        )
    )

    severity = _severity(
        final_score
    )

    indicators = [
        f"{len(frame_scores)} face frames analyzed by AI ensemble."
    ]

    if temporal_score >= 15:
        indicators.append(
            "Elevated temporal inconsistency signal detected."
        )

    if final_score >= 60:
        indicators.append(
            "AI ensemble indicates elevated deepfake probability."
        )

    if not model_result[
        "is_fine_tuned"
    ]:
        indicators.append(
            "Deepfake-specific model fine-tuning is still required."
        )

    return {
        "analysis_type": "advanced_deepfake_video",
        "is_valid": True,
        "risk_score": final_score,
        "severity": severity,
        "prediction": _prediction(
            final_score
        ),
        "confidence": _confidence(
            final_score
        ),
        "indicators": indicators,
        "features": {
            "frames_analyzed": pipeline[
                "frames_analyzed"
            ],
            "face_frames": pipeline[
                "face_frames"
            ],
            "face_detected": pipeline[
                "face_detected"
            ],
            "multiple_faces": pipeline[
                "multiple_faces"
            ],
            "frame_scores": frame_scores,
            "temporal_score": temporal_score,
            "spatial_score": round(
                spatial_score,
                2,
            ),
            "models_used": model_result[
                "models_used"
            ],
            "fine_tuned_models": model_result[
                "fine_tuned_models"
            ],
            "model_predictions": model_result[
                "model_predictions"
            ],
            "device": model_result[
                "device"
            ],
        },
        "recommendation": _recommendation(
            severity
        ),
    }


# Compatibility function.
def analyze_media(
    file_name: str,
    file_size: int,
    media_type: str = "image",
    **kwargs: Any,
) -> dict[str, Any]:

    if media_type.lower() == "video":

        return analyze_video_file(
            kwargs["file_path"],
            file_name,
            file_size,
            kwargs.get(
                "max_frames",
                16,
            ),
        )

    return analyze_image_file(
        kwargs["file_path"],
        file_name,
        file_size,
    )


def predict(
    file_name: str,
    file_size: int,
    media_type: str = "image",
    **kwargs: Any,
) -> dict[str, Any]:

    return analyze_media(
        file_name,
        file_size,
        media_type,
        **kwargs,
    )