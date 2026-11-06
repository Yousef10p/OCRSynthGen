"""
Visual Verification Tools for NER Dataset
Creates visual overlays to verify that coordinates match the text content
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg


class VisualVerification:
    """Tools to visually verify NER dataset coordinates"""

    def __init__(self, dataset_dir: str = "ner_dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"

    def draw_bounding_boxes_on_image(self, image_path: str, label_path: str, output_path: str = None, color_coding: bool = True) -> str:
        """
        Draw bounding boxes from label file onto the image for visual verification.

        Args:
            image_path: Path to the image file
            label_path: Path to the label JSON file
            output_path: Path to save the annotated image (defaults to adding '_annotated')
            color_coding: Whether to use color coding based on entity type (default True)

        Returns:
            Path to the annotated image
        """
        # Load the image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Load the labels
        with open(label_path, 'r') as f:
            label_data = json.load(f)

        # Get image dimensions for coordinate conversion
        img_width, img_height = img.size
        scale_x = img_width / 1000.0  # Scale from normalized coordinates (0-1000) to actual pixel size
        scale_y = img_height / 1000.0

        annotations = label_data.get('annotations', [])

        # Define color mapping for different entity types
        color_map = {
            # Legal Documents
            'COURT_NAME': 'red',
            'JURISDICTION_STATE': 'darkred',
            'PLAINTIFF_NAME': 'orange',
            'DEFENDANT_NAME': 'darkorange',
            'CASE_NUMBER': 'purple',
            'CLAIM_TYPE': 'blue',
            'ATTORNEY_NAME': 'green',
            'LAW_FIRM': 'darkgreen',
            'STATE_BAR_NUMBER': 'brown',
            'DATE_FILED': 'teal',
            'DAMAGES_AMOUNT': 'crimson',
            'SIGNATURE_DATE': 'olive',

            # Financial Documents
            'BANK_NAME': 'blue',
            'ACCOUNT_NUMBER': 'cyan',
            'BALANCE_AMOUNT': 'navy',
            'DATE_PERIOD': 'teal',
            'TRANSACTION_DATE': 'blue',
            'TRANSACTION_DESCRIPTION': 'darkcyan',
            'TRANSACTION_AMOUNT': 'navy',
            'INVOICE_NUMBER': 'blue',
            'INVOICE_DATE': 'darkblue',
            'DUE_DATE': 'mediumblue',
            'TOTAL_AMOUNT': 'darkslateblue',
            'COMPANY_NAME': 'royalblue',
            'CLIENT_NAME': 'lightblue',
            'ITEM_DESCRIPTION': 'steelblue',

            # Identity Documents
            'HOLDER_NAME': 'purple',
            'HOLDER_BIRTH_DATE': 'magenta',
            'HOLDER_ADDRESS': 'pink',
            'HOLDER_HEIGHT': 'violet',
            'HOLDER_WEIGHT': 'plum',
            'HOLDER_EYE_COLOR': 'indigo',
            'HOLDER_SEX': 'orchid',
            'LICENSE_NUMBER': 'darkviolet',
            'LICENSE_CLASS': 'blueviolet',
            'ISSUE_DATE': 'mediumpurple',
            'EXPIRY_DATE': 'rebeccapurple',

            # Healthcare Documents
            'HOSPITAL_NAME': 'red',
            'PATIENT_NAME': 'magenta',
            'PATIENT_DOB': 'deeppink',
            'PATIENT_SSN': 'hotpink',
            'PHYSICIAN_NAME': 'crimson',
            'DIAGNOSIS_CODE': 'orangered',
            'DIAGNOSIS_DESCRIPTION': 'tomato',
            'PRESCRIPTION_DRUG': 'coral',
            'PRESCRIPTION_DOSAGE': 'salmon',
            'PRESCRIPTION_QUANTITY': 'lightsalmon',
            'SYMPTOM': 'orange',
            'TREATMENT_PLAN': 'darkorange',
            'VITAL_SIGNS': 'gold',

            # Government Forms
            'LAST_NAME': 'red',
            'FIRST_NAME': 'crimson',
            'MIDDLE_INITIAL': 'mediumvioletred',
            'ADDRESS': 'sienna',
            'CITY': 'chocolate',
            'STATE': 'peru',
            'ZIP_CODE': 'saddlebrown',
            'DATE_OF_BIRTH': 'maroon',
            'EMPLOYEE_SIGNATURE': 'firebrick',
            'DATE_SIGNED': 'darkred',
            'EMPLOYER_NAME': 'darkblue',
            'EMPLOYER_ADDRESS': 'mediumblue',
            'EXAMINER_SIGNATURE': 'indianred',
            'EXAMINER_DATE': 'darkslategray',

            # Education Documents
            'STUDENT_NAME': 'blue',
            'STUDENT_ID': 'navy',
            'SCHOOL_NAME': 'darkblue',
            'GRADUATION_DATE': 'darkviolet',
            'DEGREE_EARNED': 'purple',
            'MAJOR': 'mediumblue',
            'GPA': 'blueviolet',
            'COURSE_NAME': 'slateblue',
            'COURSE_NUMBER': 'mediumslateblue',

            # Human Resources
            'EMPLOYEE_NAME': 'orange',
            'EMPLOYEE_ADDRESS': 'darkorange',
            'EMPLOYEE_PHONE': 'goldenrod',
            'EMPLOYEE_EMAIL': 'yellow',
            'EMPLOYEE_SSN': 'gold',
            'DATE_OF_HIRE': 'olive',
            'POSITION_TITLE': 'yellowgreen',
            'DEPARTMENT': 'green',
            'SALARY': 'darkgreen',
            'BENEFITS_ELIGIBILITY': 'lightgreen',

            # Real Estate
            'PROPERTY_ADDRESS': 'forestgreen',
            'PROPERTY_CITY': 'green',
            'PROPERTY_STATE': 'darkgreen',
            'LEGAL_DESCRIPTION': 'darkolivegreen',
            'APN_NUMBER': 'olivedrab',
            'BUYER': 'darkseagreen',
            'SELLER': 'seagreen',
            'CONTRACT_SALES_PRICE': 'mediumseagreen',
            'SETTLEMENT_DATE': 'palegreen',

            # Generic Form Elements
            'FORM_TITLE': 'black',
            'FORM_SUBTITLE': 'gray',
            'FORM_ID': 'dimgray',
            'GENERIC_FORM_FIELD': 'gray',
            'SIGNATURE_DATE': 'darkgray',

            # Default
            'default': 'red'
        }

        # Draw each bounding box
        for annotation in annotations:
            box_2d = annotation.get('box_2d', [])
            label = annotation.get('label', 'unknown')
            text_content = annotation.get('text_content', '')
            entity_type = annotation.get('entity_type', 'default')

            if len(box_2d) != 4:
                continue

            x_min, y_min, x_max, y_max = box_2d

            # Convert normalized coordinates (0-1000) to actual image coordinates
            x1 = int(x_min * scale_x)
            y1 = int(y_min * scale_y)
            x2 = int(x_max * scale_x)
            y2 = int(y_max * scale_y)

            # Determine color based on entity type if color coding is enabled
            if color_coding and entity_type != 'default':
                color = color_map.get(entity_type, color_map['default'])
            else:
                color = color_map['default']  # Use default (red) if color coding disabled

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Build annotation label string with all metadata
            # Format: label, entity_type, [line_index/total_lines], "truncated_text..."

            # Only include entity_type if it's different from label (avoid "caption, CAPTION")
            label_parts = [label]
            if entity_type and entity_type.upper() != label.upper().replace('_', ' '):
                label_parts.append(entity_type)

            # Add line metadata if this is a multi-line element
            if annotation.get('line_index') is not None:
                line_info = f"[{annotation['line_index']}/{annotation['total_lines']}]"
                label_parts.append(line_info)

            # Calculate available width for text truncation
            box_width = x2 - x1
            # Estimate characters that fit: box_width / 6 (rough char width) - space for label parts
            base_label = ", ".join(label_parts)
            available_chars = max(10, int((box_width - len(base_label) * 6 - 20) / 6))

            # Truncate text content to fit available space
            truncated_text = text_content[:available_chars].replace('\n', ' ').strip()
            if len(text_content) > available_chars:
                truncated_text += "..."

            # Only add text if there's room (box width > 100px)
            if box_width > 100:
                label_parts.append(f'"{truncated_text}"')

            # Join all parts with commas
            label_text = ", ".join(label_parts)

            # Get font for rendering
            try:
                # Try to use a default font
                font = ImageFont.load_default()
            except:
                font = None

            # Calculate text bounding box size
            try:
                text_bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                # Fallback if textbbox fails
                text_width = len(label_text) * 6  # Rough estimate
                text_height = 10

            # Position label INSIDE box at top-left corner
            label_x = x1 + 2
            label_y = y1 + 2

            # Draw background rectangle for label (semi-transparent white)
            bg_padding = 2
            bg_box = [
                label_x - bg_padding,
                label_y - bg_padding,
                label_x + text_width + bg_padding,
                label_y + text_height + bg_padding
            ]

            # Ensure background doesn't exceed bounding box and maintains valid coordinates
            bg_box[2] = min(bg_box[2], x2 - 2)
            bg_box[3] = min(bg_box[3], y2 - 2)

            # Ensure x2 > x1 and y2 > y1 (valid rectangle)
            if bg_box[2] > bg_box[0] and bg_box[3] > bg_box[1]:
                # Draw opaque white background
                draw.rectangle(bg_box, fill=(255, 255, 255, 255))

            # Draw label text on top of background
            draw.text((label_x, label_y), label_text, fill=color, font=font)

        # Save the annotated image
        if output_path is None:
            # Generate unique output path in same directory as the original image
            original_path = Path(image_path)
            output_path = str(original_path.parent / f"annotated_{original_path.name}")

        img.save(output_path)
        return output_path

    def create_verification_report(self, sample_count: int = 10, output_dir: str = None) -> str:
        """
        Create a visual verification report showing annotated samples.

        Args:
            sample_count: Number of samples to include in the report
            output_dir: Output directory for the report (defaults to verification_report)

        Returns:
            Path to the generated report directory
        """
        if output_dir is None:
            report_dir = self.dataset_dir / "verification_report"
        else:
            report_dir = Path(output_dir)

        report_dir.mkdir(parents=True, exist_ok=True)

        # Get sample label files
        label_files = list(self.labels_dir.glob("*.json"))
        if not label_files:
            raise ValueError("No label files found in the dataset")

        # Take a random sample of label files
        import random
        sample_labels = random.sample(label_files, min(sample_count, len(label_files)))

        verification_results = []

        for label_file in sample_labels:
            # Find corresponding image file
            image_file = self.images_dir / label_file.name.replace('.json', '.jpg')
            if not image_file.exists():
                image_file = self.images_dir / label_file.name.replace('.json', '.jpeg')

            if not image_file.exists():
                print(f"Warning: Image file not found for {label_file}")
                continue

            # Create annotated image
            annotated_image_path = self.draw_bounding_boxes_on_image(
                str(image_file), str(label_file)
            )

            # Move annotated image to report directory
            # Use the original stem plus a unique identifier to avoid conflicts
            annotated_name = f"annotated_{label_file.stem}.jpg"
            final_annotated_path = report_dir / annotated_name

            # Check if file already exists and use a unique name if needed
            if final_annotated_path.exists():
                import uuid
                # Add unique identifier to avoid conflicts between different samples
                unique_name = f"annotated_{label_file.stem}_{uuid.uuid4().hex[:8]}.jpg"
                final_annotated_path = report_dir / unique_name

            # Use shutil.move instead of os.rename to handle potential cross-device issues
            import shutil
            shutil.move(annotated_image_path, final_annotated_path)

            # Load label data for verification info
            with open(label_file, 'r') as f:
                label_data = json.load(f)

            verification_results.append({
                "image": image_file.name,
                "annotated_image": annotated_name,
                "total_annotations": len(label_data.get('annotations', [])),
                "label_path": str(label_file)
            })

        # Create summary report
        summary = {
            "total_samples_verified": len(verification_results),
            "samples": verification_results,
            "dataset_directory": str(self.dataset_dir)
        }

        summary_path = report_dir / "verification_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return str(report_dir)

    def visualize_single_sample(self, sample_id: str, show_text_content: bool = True):
        """
        Create a detailed visualization of a single sample for inspection.

        Args:
            sample_id: ID of the sample (without extension)
            show_text_content: Whether to show the extracted text content
        """
        # Find the corresponding files
        image_file = self.images_dir / f"sample_{sample_id}.jpg"
        label_file = self.labels_dir / f"sample_{sample_id}.json"

        if not image_file.exists() or not label_file.exists():
            raise FileNotFoundError(f"Image or label file not found for sample {sample_id}")

        # Load image
        img = Image.open(image_file)
        img_width, img_height = img.size
        scale_x = img_width / 1000.0
        scale_y = img_height / 1000.0

        # Load labels
        with open(label_file, 'r') as f:
            label_data = json.load(f)

        annotations = label_data.get('annotations', [])

        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 16))
        ax.imshow(img)

        # Draw bounding boxes
        for annotation in annotations:
            box_2d = annotation.get('box_2d', [])
            label = annotation.get('label', 'unknown')
            text_content = annotation.get('text_content', '') if show_text_content else ''

            if len(box_2d) != 4:
                continue

            x_min, y_min, x_max, y_max = box_2d
            x1 = x_min * scale_x
            y1 = y_min * scale_y
            x2 = x_max * scale_x
            y2 = y_max * scale_y

            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

            # Add label text
            ax.text(x1, y1-5, f"{label}", fontsize=8, color='red',
                   bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="red", lw=1))

            # Add text content if requested
            if show_text_content and text_content:
                ax.text(x1, y2+15, f"{text_content}", fontsize=7, color='blue',
                       bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="blue", lw=1))

        ax.set_title(f"Sample {sample_id} - {len(annotations)} annotations")
        ax.axis('off')

        # Save the visualization
        viz_path = self.dataset_dir / "visualizations"
        viz_path.mkdir(exist_ok=True)
        output_path = viz_path / f"sample_{sample_id}_viz.jpg"
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.show()

        return str(output_path)

    def batch_visual_verification(self, max_samples: int = 100):
        """
        Perform batch visual verification on a set of samples.

        Args:
            max_samples: Maximum number of samples to verify
        """
        label_files = list(self.labels_dir.glob("*.json"))
        if not label_files:
            print("No label files found to verify")
            return

        # Limit the number of samples to verify
        label_files = label_files[:max_samples]

        print(f"Starting batch visual verification for {len(label_files)} samples...")

        success_count = 0
        error_count = 0

        for i, label_file in enumerate(label_files):
            try:
                # Find corresponding image
                image_file = self.images_dir / label_file.name.replace('.json', '.jpg')
                if not image_file.exists():
                    image_file = self.images_dir / label_file.name.replace('.json', '.jpeg')

                if not image_file.exists():
                    print(f"Missing image for {label_file.name}")
                    error_count += 1
                    continue

                # Create annotated image
                annotated_path = self.draw_bounding_boxes_on_image(
                    str(image_file), str(label_file)
                )

                success_count += 1

                if (i + 1) % 20 == 0:
                    print(f"Processed {i + 1}/{len(label_files)} samples...")

            except Exception as e:
                print(f"Error processing {label_file.name}: {str(e)}")
                error_count += 1

        print(f"\nBatch verification completed!")
        print(f"Successful: {success_count}")
        print(f"Errors: {error_count}")
        print(f"Total processed: {success_count + error_count}")