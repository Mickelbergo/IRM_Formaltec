"""
FastAPI REST API for Forensic Wound Segmentation.

Provides production-grade endpoints for:
- Image segmentation
- LLM-powered forensic reports
- Model comparison
- Performance metrics

Usage:
    uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

API Documentation:
    http://localhost:8000/docs (Swagger)
    http://localhost:8000/redoc (ReDoc)
"""

import os
import sys
import io
import json
import time
import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "training"))

from model import UNetWithViT, UNetWithClassification, UNetWithSwinTransformer
from tta_inference import TTAWrapper, load_image
from local_llm import LocalLLMReportGenerator
from torchvision import transforms

# ==================== Configuration ====================

class Config:
    """Application configuration"""
    MODEL_PATH = os.getenv("MODEL_PATH", "E:/projects/Wound_Segmentation_III/Data/models/best_model.pth")
    CONFIG_PATH = os.getenv("CONFIG_PATH", "Code/configs/training_config.json")
    PREPROCESSING_CONFIG_PATH = os.getenv("PREPROCESSING_CONFIG_PATH", "Code/configs/preprocessing_config.json")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    # LLM Configuration
    # IMPORTANT: For sensitive forensic data, use LOCAL_LLM_ONLY=true
    # This ensures NO data is sent to external APIs
    LOCAL_LLM_ONLY = os.getenv("LOCAL_LLM_ONLY", "true").lower() == "true"
    LLM_TYPE = os.getenv("LLM_TYPE", "template")  # 'template', 'llama3', 'mistral', or 'openai'
    LOCAL_LLM_PATH = os.getenv("LOCAL_LLM_PATH", None)  # Path to local Llama/Mistral model
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Only used if LOCAL_LLM_ONLY=false

    # Performance
    ENABLE_TTA = os.getenv("ENABLE_TTA", "true").lower() == "true"
    TTA_AUGMENTATIONS = ["original", "hflip", "vflip", "rot90"]

config = Config()

# ==================== Models ====================

class Pydantic_Models:
    """Pydantic models for request/response validation"""

    class SegmentationRequest(BaseModel):
        use_tta: bool = Field(True, description="Enable test-time augmentation")
        return_confidence: bool = Field(True, description="Return per-pixel confidence")
        generate_report: bool = Field(False, description="Generate LLM forensic report")

    class SegmentationResult(BaseModel):
        success: bool
        message: str
        prediction: Optional[Dict[str, Any]] = None
        confidence_scores: Optional[Dict[str, float]] = None
        processing_time_ms: float
        model_version: str
        report: Optional[str] = None

    class ModelInfo(BaseModel):
        model_name: str
        version: str
        encoder: str
        num_classes: int
        parameters: int
        device: str
        tta_enabled: bool

    class HealthCheck(BaseModel):
        status: str
        timestamp: str
        model_loaded: bool
        device: str
        gpu_available: bool

# ==================== Application Setup ====================

app = FastAPI(
    title="Forensic Wound Segmentation API",
    description="Production-grade API for automated wound analysis using DINOv2",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Model Loading ====================

class ModelManager:
    """Manages model loading, caching, and inference"""

    def __init__(self):
        self.model = None
        self.tta_model = None
        self.config = None
        self.class_names = None
        self.model_info = None
        self.llm_generator = None

    def load_model(self):
        """Load model and configurations"""
        print(f"Loading model from {config.MODEL_PATH}...")
        print(f"Using device: {config.DEVICE}")

        # Load configs
        with open(config.CONFIG_PATH, 'r') as f:
            self.config = json.load(f)

        try:
            with open(config.PREPROCESSING_CONFIG_PATH, 'r') as f:
                preprocessing_config = json.load(f)
                self.class_names = preprocessing_config.get('class_names', {})
        except:
            self.class_names = {}

        # Create model
        encoder = self.config.get('model', {}).get('encoder', 'vit')
        num_classes = self.config.get('model', {}).get('segmentation_classes', 11)

        if encoder == 'vit':
            vit_config = self.config.get('model', {}).get('vit_config', {})
            self.model = UNetWithViT(
                classes=num_classes,
                activation=None,
                model_name=vit_config.get('model_name', 'facebook/dinov2-large'),
                dropout_rate=vit_config.get('dropout_rate', 0.3),
                stochastic_depth_rate=vit_config.get('stochastic_depth_rate', 0.1)
            )
        elif encoder == 'transformer':
            self.model = UNetWithSwinTransformer(
                classes=num_classes,
                activation=None
            )
        else:
            self.model = UNetWithClassification(
                encoder_name=encoder,
                encoder_weights=self.config.get('model', {}).get('encoder_weights', 'imagenet'),
                classes=num_classes,
                activation=None
            )

        # Load weights
        state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(config.DEVICE)
        self.model.eval()

        # Create TTA wrapper if enabled
        if config.ENABLE_TTA:
            self.tta_model = TTAWrapper(
                self.model,
                device=config.DEVICE,
                augmentations=config.TTA_AUGMENTATIONS
            )

        # Store model info
        total_params = sum(p.numel() for p in self.model.parameters())
        self.model_info = {
            "model_name": encoder,
            "version": self.config.get('model', {}).get('version', 'unknown'),
            "encoder": encoder,
            "num_classes": num_classes,
            "parameters": total_params,
            "device": config.DEVICE,
            "tta_enabled": config.ENABLE_TTA
        }

        # Initialize LLM generator (local by default for privacy)
        if config.LOCAL_LLM_ONLY:
            print("🔒 PRIVACY MODE: Using local LLM (no external API calls)")
            self.llm_generator = LocalLLMReportGenerator(
                model_type=config.LLM_TYPE,
                model_path=config.LOCAL_LLM_PATH
            )
        else:
            print("⚠️  WARNING: External API mode enabled (OpenAI)")
            self.llm_generator = None  # Will use OpenAI in generate_llm_report

        print(f"✓ Model loaded successfully!")
        print(f"  - Encoder: {encoder}")
        print(f"  - Parameters: {total_params:,}")
        print(f"  - Device: {config.DEVICE}")
        print(f"  - TTA: {config.ENABLE_TTA}")
        print(f"  - LLM Mode: {'LOCAL (Privacy-Preserving)' if config.LOCAL_LLM_ONLY else 'EXTERNAL (OpenAI)'}")

    def predict(self, image_tensor: torch.Tensor, use_tta: bool = True) -> tuple:
        """
        Run inference on image tensor.

        Returns:
            (pred_mask, confidence_map, class_confidences)
        """
        with torch.no_grad():
            if use_tta and self.tta_model is not None:
                logits = self.tta_model(image_tensor)
            else:
                logits = self.model(image_tensor)

            # Get predicted classes
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

            # Get confidence scores (softmax probabilities)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            confidence_map = probs.max(dim=0)[0].cpu().numpy()

            # Per-class confidence (mean confidence for predicted pixels of each class)
            class_confidences = {}
            for class_id in range(logits.shape[1]):
                mask = pred_mask == class_id
                if mask.sum() > 0:
                    class_conf = probs[class_id][mask].mean().item()
                    class_name = self.class_names.get(str(class_id), f"Class {class_id}")
                    class_confidences[class_name] = float(class_conf)

            return pred_mask, confidence_map, class_confidences

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    model_manager.load_model()

# ==================== Helper Functions ====================

def validate_image(file: UploadFile) -> None:
    """Validate uploaded image"""
    # Check file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {config.ALLOWED_EXTENSIONS}"
        )

    # Check file size (if possible)
    # Note: file.size might not be available in all cases
    if hasattr(file, 'size') and file.size and file.size > config.MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {config.MAX_IMAGE_SIZE / 1024 / 1024}MB"
        )

def preprocess_image(image: Image.Image, target_size=(384, 384)) -> torch.Tensor:
    """Preprocess PIL image for model input"""
    # Resize
    image = image.resize(target_size, Image.BILINEAR)

    # To tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(config.DEVICE)

def mask_to_image(mask: np.ndarray) -> Image.Image:
    """Convert segmentation mask to PIL image"""
    # Scale for visibility (0-10 -> 0-255)
    mask_scaled = (mask * 23).astype(np.uint8)  # 11 classes * 23 ≈ 253
    return Image.fromarray(mask_scaled)

def generate_llm_report(pred_mask: np.ndarray, class_confidences: Dict[str, float]) -> str:
    """
    Generate forensic report using LOCAL LLM (privacy-preserving by default).

    IMPORTANT: By default, NO data is sent to external APIs.
    All processing happens locally on your infrastructure.

    To enable external APIs (OpenAI), set LOCAL_LLM_ONLY=false in environment.
    """
    # Use local LLM generator (default, privacy-preserving)
    if config.LOCAL_LLM_ONLY and model_manager.llm_generator:
        return model_manager.llm_generator.generate_report(
            pred_mask=pred_mask,
            class_confidences=class_confidences,
            class_names=model_manager.class_names,
            model_info=model_manager.model_info
        )

    # External API mode (OpenAI) - only if explicitly enabled
    elif not config.LOCAL_LLM_ONLY and config.OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = config.OPENAI_API_KEY

            # Prepare analysis data
            unique_classes = np.unique(pred_mask)
            class_distribution = {
                model_manager.class_names.get(str(c), f"Class {c}"): int((pred_mask == c).sum())
                for c in unique_classes if c != 0  # Exclude background
            }

            # Create prompt (ONLY metadata, not actual image data)
            prompt = f"""You are a forensic pathologist analyzing wound segmentation results from an AI model.

SEGMENTATION DATA:
- Detected wound classes: {list(class_distribution.keys())}
- Class distribution (pixels): {class_distribution}
- Confidence scores: {class_confidences}

Generate a professional forensic analysis report with:
1. WOUND CLASSIFICATION: Primary and secondary findings
2. FORENSIC INTERPRETATION: What this suggests about the injury mechanism
3. CONFIDENCE ASSESSMENT: Reliability of the analysis
4. RECOMMENDATIONS: Next steps for forensic investigation

Keep the tone professional and suitable for legal/medical documentation."""

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert forensic pathologist."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"⚠️  OpenAI API failed: {e}")
            print("Falling back to local template")
            return generate_template_report(pred_mask, class_confidences)

    # Fallback to template if no LLM available
    else:
        return generate_template_report(pred_mask, class_confidences)

def generate_template_report(pred_mask: np.ndarray, class_confidences: Dict[str, float]) -> str:
    """Generate basic template report when LLM unavailable"""
    unique_classes = np.unique(pred_mask)
    class_distribution = {
        model_manager.class_names.get(str(c), f"Class {c}"): int((pred_mask == c).sum())
        for c in unique_classes if c != 0
    }

    # Find primary wound type
    if class_distribution:
        primary_class = max(class_distribution, key=class_distribution.get)
        primary_confidence = class_confidences.get(primary_class, 0.0)
    else:
        return "No significant wound features detected in the image."

    report = f"""FORENSIC WOUND ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SEGMENTATION SUMMARY:
- Primary Wound Type: {primary_class}
- Confidence: {primary_confidence:.1%}
- Total Wound Area: {sum(class_distribution.values())} pixels

DETECTED CLASSES:
"""
    for class_name, pixel_count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True):
        confidence = class_confidences.get(class_name, 0.0)
        report += f"  - {class_name}: {pixel_count} pixels (Confidence: {confidence:.1%})\n"

    report += f"""
CONFIDENCE METRICS:
- Model: {model_manager.model_info['model_name']} v{model_manager.model_info['version']}
- TTA Enabled: {model_manager.model_info['tta_enabled']}
- Overall Confidence: {np.mean(list(class_confidences.values())):.1%}

RECOMMENDATIONS:
- Further manual examination recommended for areas with confidence < 70%
- Consider photographic documentation from multiple angles
- Consult with forensic pathologist for definitive classification

Note: This is an automated analysis. Final interpretation should be made by qualified forensic experts.
"""

    return report

# ==================== API Endpoints ====================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "message": "Forensic Wound Segmentation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=Pydantic_Models.HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return Pydantic_Models.HealthCheck(
        status="healthy" if model_manager.model is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_manager.model is not None,
        device=config.DEVICE,
        gpu_available=torch.cuda.is_available()
    )

@app.get("/api/v1/models", response_model=Pydantic_Models.ModelInfo, tags=["Models"])
async def get_model_info():
    """Get current model information"""
    if model_manager.model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return Pydantic_Models.ModelInfo(**model_manager.model_info)

@app.post("/api/v1/segment", response_model=Pydantic_Models.SegmentationResult, tags=["Inference"])
async def segment_image(
    file: UploadFile = File(...),
    use_tta: bool = Query(True, description="Enable test-time augmentation"),
    return_confidence: bool = Query(True, description="Return per-pixel confidence"),
    generate_report: bool = Query(False, description="Generate LLM forensic report")
):
    """
    Segment wound image and optionally generate forensic report.

    - **file**: Image file (JPG, PNG, BMP)
    - **use_tta**: Enable test-time augmentation for better accuracy
    - **return_confidence**: Include confidence scores in response
    - **generate_report**: Generate LLM-powered forensic report

    Returns segmentation mask, confidence scores, and optional report.
    """
    start_time = time.time()

    try:
        # Validate file
        validate_image(file)

        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        original_size = image.size

        # Preprocess
        image_tensor = preprocess_image(image)

        # Run inference
        pred_mask, confidence_map, class_confidences = model_manager.predict(image_tensor, use_tta)

        # Prepare response
        processing_time = (time.time() - start_time) * 1000  # ms

        result = {
            "success": True,
            "message": "Segmentation completed successfully",
            "prediction": {
                "shape": pred_mask.shape,
                "unique_classes": np.unique(pred_mask).tolist(),
                "class_names": {
                    str(c): model_manager.class_names.get(str(c), f"Class {c}")
                    for c in np.unique(pred_mask)
                }
            },
            "confidence_scores": class_confidences if return_confidence else None,
            "processing_time_ms": processing_time,
            "model_version": model_manager.model_info['version']
        }

        # Generate report if requested
        if generate_report:
            report = generate_llm_report(pred_mask, class_confidences)
            result["report"] = report

        return Pydantic_Models.SegmentationResult(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/api/v1/segment/image", tags=["Inference"])
async def segment_image_with_visualization(
    file: UploadFile = File(...),
    use_tta: bool = Query(True)
):
    """
    Segment image and return visualization as PNG.

    Returns PNG image with segmentation mask overlay.
    """
    try:
        validate_image(file)

        # Read and process
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_tensor = preprocess_image(image)

        # Inference
        pred_mask, _, _ = model_manager.predict(image_tensor, use_tta)

        # Convert to image
        mask_image = mask_to_image(pred_mask)

        # Return as PNG
        img_byte_arr = io.BytesIO()
        mask_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# ==================== Main ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Forensic Wound Segmentation API Server")
    print("=" * 60)
    print(f"Model Path: {config.MODEL_PATH}")
    print(f"Device: {config.DEVICE}")
    print(f"TTA Enabled: {config.ENABLE_TTA}")
    print(f"Privacy Mode: {'LOCAL (No external APIs)' if config.LOCAL_LLM_ONLY else 'EXTERNAL (OpenAI)'}")
    print(f"LLM Type: {config.LLM_TYPE}")
    print("=" * 60)
    print("\nStarting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
