"""
Local LLM Integration for Privacy-Preserving Forensic Report Generation.

NO DATA LEAVES YOUR INFRASTRUCTURE.

Supports:
- Local Llama 3 (via llama-cpp-python or transformers)
- Local Mistral models
- Template-based reports (no LLM)

For sensitive forensic/medical data that cannot be sent to external APIs.

Usage:
    from local_llm import LocalLLMReportGenerator

    generator = LocalLLMReportGenerator(model_type='llama3')
    report = generator.generate_report(pred_mask, class_confidences, class_names)
"""

import os
import json
import numpy as np
from typing import Dict, Optional
from datetime import datetime

class LocalLLMReportGenerator:
    """
    Generate forensic reports using local LLM - NO external API calls.

    All processing happens on your infrastructure.
    """

    def __init__(self, model_type: str = 'template', model_path: Optional[str] = None):
        """
        Initialize local LLM report generator.

        Args:
            model_type: 'llama3', 'mistral', or 'template' (no LLM)
            model_path: Path to local model files (for llama3/mistral)
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model = None

        if model_type in ['llama3', 'mistral'] and model_path:
            self._load_local_model()

    def _load_local_model(self):
        """Load local LLM model."""
        print(f"Loading local {self.model_type} model from {self.model_path}...")

        try:
            if self.model_type == 'llama3':
                # Option 1: llama-cpp-python (faster, quantized)
                try:
                    from llama_cpp import Llama
                    self.model = Llama(
                        model_path=self.model_path,
                        n_ctx=2048,
                        n_threads=8,
                        n_gpu_layers=35  # Adjust based on GPU
                    )
                    print("✓ Loaded Llama 3 via llama-cpp-python")
                except ImportError:
                    # Option 2: transformers (slower, full precision)
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        device_map="auto",
                        torch_dtype="auto"
                    )
                    print("✓ Loaded Llama 3 via transformers")

            elif self.model_type == 'mistral':
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    torch_dtype="auto"
                )
                print("✓ Loaded Mistral via transformers")

        except Exception as e:
            print(f"⚠ Failed to load local LLM: {e}")
            print("Falling back to template-based reports")
            self.model = None
            self.model_type = 'template'

    def generate_report(
        self,
        pred_mask: np.ndarray,
        class_confidences: Dict[str, float],
        class_names: Dict[str, str],
        model_info: Optional[Dict] = None
    ) -> str:
        """
        Generate forensic report using local LLM or template.

        Args:
            pred_mask: Predicted segmentation mask [H, W]
            class_confidences: Per-class confidence scores
            class_names: Class ID to name mapping
            model_info: Optional model metadata

        Returns:
            Generated forensic report (string)
        """
        # Prepare data (same for all methods)
        unique_classes = np.unique(pred_mask)
        class_distribution = {
            class_names.get(str(c), f"Class {c}"): int((pred_mask == c).sum())
            for c in unique_classes if c != 0  # Exclude background
        }

        if self.model_type == 'template' or self.model is None:
            return self._generate_template_report(
                class_distribution, class_confidences, model_info
            )

        elif self.model_type in ['llama3', 'mistral']:
            return self._generate_llm_report(
                class_distribution, class_confidences, model_info
            )

        else:
            return self._generate_template_report(
                class_distribution, class_confidences, model_info
            )

    def _generate_llm_report(
        self,
        class_distribution: Dict[str, int],
        class_confidences: Dict[str, float],
        model_info: Optional[Dict]
    ) -> str:
        """Generate report using local LLM (Llama 3 or Mistral)."""

        # Create prompt
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

Keep the tone professional and suitable for legal/medical documentation.

FORENSIC WOUND ANALYSIS REPORT:"""

        try:
            # Generate using llama-cpp-python
            if hasattr(self.model, 'create_completion'):
                response = self.model.create_completion(
                    prompt,
                    max_tokens=800,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["USER:", "ASSISTANT:"]
                )
                report = response['choices'][0]['text'].strip()

            # Generate using transformers
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                report = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract only the generated part (after prompt)
                report = report.split("FORENSIC WOUND ANALYSIS REPORT:")[-1].strip()

            # Add metadata
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            header = f"FORENSIC WOUND ANALYSIS REPORT\nGenerated: {timestamp} (Local LLM: {self.model_type})\n\n"

            return header + report

        except Exception as e:
            print(f"⚠ LLM generation failed: {e}")
            print("Falling back to template report")
            return self._generate_template_report(class_distribution, class_confidences, model_info)

    def _generate_template_report(
        self,
        class_distribution: Dict[str, int],
        class_confidences: Dict[str, float],
        model_info: Optional[Dict]
    ) -> str:
        """Generate template-based report (no LLM, fully deterministic)."""

        # Find primary wound type
        if class_distribution:
            primary_class = max(class_distribution, key=class_distribution.get)
            primary_confidence = class_confidences.get(primary_class, 0.0)
            primary_pixels = class_distribution[primary_class]
        else:
            return "No significant wound features detected in the image."

        # Determine wound interpretation based on class
        interpretations = {
            "cut": "Sharp force injury consistent with a cutting implement. Clean wound edges suggest a sharp blade.",
            "schnitt": "Sharp force injury consistent with a cutting implement. Clean wound edges suggest a sharp blade.",
            "stich": "Puncture wound consistent with a stabbing implement. Deep penetration with minimal surface trauma.",
            "dermatorrhagia": "Subcutaneous hemorrhaging without distinct injury pattern. May indicate blunt force trauma.",
            "ungeformter_bluterguss": "Diffuse bruising pattern without clear boundaries. Consistent with blunt force impact.",
            "geformter_bluterguss": "Well-defined bruising pattern. May retain shape of impact object.",
            "hematoma": "Localized blood collection. Indicates blunt force trauma with vascular damage.",
            "skin_abrasion": "Superficial skin damage. Indicates scraping or friction injury.",
            "hautabschürfung": "Superficial skin damage. Indicates scraping or friction injury.",
            "quetsch_riss_wunden": "Compression-tear wound. Indicates crushing force with tissue tearing.",
            "thermal": "Thermal injury pattern. Requires burn depth assessment.",
            "thermische_gewalt": "Thermal injury pattern. Requires burn depth assessment.",
            "puncture_gun_shot": "Penetrating trauma consistent with projectile injury. Requires entry/exit wound analysis.",
            "halbscharfe_gewalt": "Semi-sharp force injury. Characteristics between sharp and blunt trauma.",
            "risswunden": "Laceration pattern. Indicates tearing force on tissue.",
        }

        # Get interpretation
        primary_lower = primary_class.lower()
        interpretation = interpretations.get(primary_lower, "Further analysis required for definitive classification.")

        # Build report
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        report = f"""FORENSIC WOUND ANALYSIS REPORT
Generated: {timestamp} (Template-based, no external APIs)

========================================
WOUND CLASSIFICATION
========================================
Primary Wound Type: {primary_class}
Confidence: {primary_confidence:.1%}
Affected Area: {primary_pixels} pixels

"""

        # Add secondary findings if present
        if len(class_distribution) > 1:
            report += "Secondary Findings:\n"
            for class_name, pixel_count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)[1:]:
                confidence = class_confidences.get(class_name, 0.0)
                report += f"  - {class_name}: {pixel_count} pixels ({confidence:.1%} confidence)\n"
            report += "\n"

        report += f"""========================================
FORENSIC INTERPRETATION
========================================
{interpretation}

"""

        # Add detailed class distribution
        report += """========================================
DETECTED CLASSES (DETAILED)
========================================
"""
        for class_name, pixel_count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True):
            confidence = class_confidences.get(class_name, 0.0)
            percentage = (pixel_count / sum(class_distribution.values())) * 100
            report += f"  {class_name}:\n"
            report += f"    - Area: {pixel_count} pixels ({percentage:.1f}% of wound)\n"
            report += f"    - Confidence: {confidence:.1%}\n"

        # Add confidence assessment
        avg_confidence = np.mean(list(class_confidences.values()))
        confidence_level = "high" if avg_confidence > 0.8 else "moderate" if avg_confidence > 0.6 else "low"

        report += f"""
========================================
CONFIDENCE ASSESSMENT
========================================
Overall Confidence: {avg_confidence:.1%} ({confidence_level})
Model Version: {model_info.get('version', 'unknown') if model_info else 'unknown'}
TTA Enabled: {model_info.get('tta_enabled', False) if model_info else False}

Reliability Notes:
"""
        if avg_confidence > 0.8:
            report += "  ✓ High confidence predictions across all detected classes\n"
        elif avg_confidence > 0.6:
            report += "  ⚠ Moderate confidence - manual review recommended for low-confidence regions\n"
        else:
            report += "  ⚠ Low confidence - expert review strongly recommended\n"

        # Add recommendations
        report += """
========================================
RECOMMENDATIONS
========================================
1. Photographic Documentation:
   - Document wound from multiple angles (0°, 45°, 90°)
   - Include measurement scale in all photographs
   - Capture both overall and close-up views

2. Physical Examination:
   - Measure wound dimensions (length, width, depth)
   - Assess wound edges (clean-cut vs. irregular)
   - Document any foreign material or debris

3. Further Analysis:
"""

        if primary_confidence < 0.7:
            report += "   - Expert pathologist review required (low confidence)\n"
        if "cut" in primary_lower or "schnitt" in primary_lower:
            report += "   - Blade width estimation from wound morphology\n"
            report += "   - Examination for material transfer from implement\n"
        if "thermal" in primary_lower:
            report += "   - Burn depth classification required\n"
            report += "   - Temperature and duration estimation\n"
        if "puncture" in primary_lower or "stich" in primary_lower:
            report += "   - Penetration depth assessment\n"
            report += "   - Internal organ damage evaluation\n"

        report += """
4. Legal Documentation:
   - Preserve all photographic evidence in original resolution
   - Maintain chain of custody for all physical evidence
   - Document date, time, and personnel involved in examination

========================================
IMPORTANT NOTICE
========================================
This is an AUTOMATED analysis generated by an AI system.
FINAL INTERPRETATION must be made by a qualified forensic pathologist.

All image data was processed LOCALLY on your infrastructure.
NO data was sent to external servers or cloud services.

For legal proceedings, this report should be used as a
preliminary screening tool, NOT as definitive evidence.
========================================

Report generated by: Forensic Wound Segmentation System v1.0
Processing mode: Local inference (privacy-preserving)
"""

        return report


# Example usage
if __name__ == "__main__":
    # Example: Template-based report (no LLM, completely local)
    generator = LocalLLMReportGenerator(model_type='template')

    # Dummy data
    pred_mask = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 4, 4]])
    class_confidences = {"cut": 0.87, "skin_abrasion": 0.12}
    class_names = {"0": "background", "1": "dermatorrhagia", "4": "cut"}

    report = generator.generate_report(pred_mask, class_confidences, class_names)
    print(report)

    # Example: Local Llama 3 (requires model downloaded)
    # generator = LocalLLMReportGenerator(
    #     model_type='llama3',
    #     model_path='./models/llama-3-8b-instruct.gguf'
    # )
    # report = generator.generate_report(pred_mask, class_confidences, class_names)
