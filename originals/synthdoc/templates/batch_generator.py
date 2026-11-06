"""
Batch generation system for creating multiple document variations.
"""
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from .base import BaseTemplate
from .registry import registry
from .config_manager import config_manager


class BatchGenerator:
    """Handles batch generation of documents with multiple variations"""
    
    def __init__(self, output_dir: str = "generated_documents"):
        self.output_dir = Path(output_dir)
    
    def generate_batch(self, 
                      template_name: str, 
                      data_generator_func, 
                      count: int, 
                      config_names: List[str] = None,
                      output_prefix: str = "") -> List[str]:
        """
        Generate a batch of documents using specified template and configurations.
        
        Args:
            template_name: Name of the template to use
            data_generator_func: Function that returns data for each document
            count: Number of documents to generate
            config_names: List of configuration names to use (defaults to all available)
            output_prefix: Prefix for output filenames
        
        Returns:
            List of generated file paths
        """
        if config_names is None:
            # Use default configs for the template category
            category = self._get_template_category(template_name)
            config_names = [name for name in config_manager.list_configs() 
                           if name.startswith(category or template_name)]
            if not config_names:
                config_names = [f"{template_name}_default"]
        
        generated_files = []
        
        for i in range(count):
            # Get data for this document
            doc_data = data_generator_func(i)
            
            # Cycle through configurations
            config_name = config_names[i % len(config_names)]
            config = config_manager.get_config(config_name)
            
            # Get template instance
            template = registry.get_template_instance(template_name, config)
            
            # Render document
            html_content = template.render(doc_data)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_prefix}{template_name}_{i:03d}_{config_name}_{timestamp}.html"
            filepath = self.output_dir / filename
            
            # Save document
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            generated_files.append(str(filepath))
            
            # Also save the data used to generate this document
            data_filename = f"{output_prefix}{template_name}_{i:03d}_{config_name}_{timestamp}_data.json"
            data_filepath = self.output_dir / data_filename
            
            with open(data_filepath, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, default=str)
        
        return generated_files
    
    def generate_template_variations(self, 
                                   base_template_name: str,
                                   variation_params: Dict[str, List[Any]],
                                   data_generator_func,
                                   count_per_variation: int = 1) -> Dict[str, List[str]]:
        """
        Generate documents using template variations based on parameter options.
        
        Args:
            base_template_name: Name of the base template to vary
            variation_params: Dictionary mapping config fields to lists of possible values
            data_generator_func: Function that returns data for each document
            count_per_variation: Number of documents to generate per variation
        
        Returns:
            Dictionary mapping variation names to lists of generated file paths
        """
        # Generate configuration variations
        base_config_name = f"{base_template_name}_default"
        variation_names = config_manager.generate_variations(base_config_name, variation_params)
        
        results = {}
        
        for config_name in variation_names:
            # Create data for each document in this variation
            all_files = []
            for i in range(count_per_variation):
                doc_data = data_generator_func(i)
                
                # Get template instance with this config
                config = config_manager.get_config(config_name)
                template = registry.get_template_instance(base_template_name, config)
                
                # Render document
                html_content = template.render(doc_data)
                
                # Generate filename specific to this variation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Create a much shorter and safer filename
                safe_config_name = f"var_{len(config_name) % 10000:04d}"  # Use hash-like code based on name length
                filename = f"{base_template_name}_{safe_config_name}_{i:03d}_{timestamp}.html"
                filepath = self.output_dir / filename
                
                # Save document
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                all_files.append(str(filepath))
                
                # Save the data used to generate this document
                data_filename = f"{base_template_name}_{safe_config_name}_{i:03d}_{timestamp}_data.json"
                data_filepath = self.output_dir / data_filename
                
                with open(data_filepath, 'w', encoding='utf-8') as f:
                    json.dump(doc_data, f, indent=2, default=str)
            
            results[config_name] = all_files
        
        return results
    
    def generate_layout_segmentation_dataset(self, 
                                           template_categories: List[str],
                                           variations_per_template: int = 5,
                                           docs_per_variation: int = 10) -> Dict[str, List[str]]:
        """
        Generate a comprehensive dataset for layout segmentation training.
        
        Args:
            template_categories: List of template categories to include
            variations_per_template: Number of style/layout variations per template
            docs_per_variation: Number of documents per variation
        
        Returns:
            Dictionary mapping template names to lists of generated file paths
        """
        dataset_results = {}
        
        for category in template_categories:
            # Get all templates in this category
            template_names = registry.list_templates(category)
            
            for template_name in template_names:
                print(f"Generating dataset for {template_name}...")
                
                # Define parameter variations for this category
                variation_params = self._get_default_variations(category)
                
                # Generate variations for this template
                results = self.generate_template_variations(
                    template_name,
                    variation_params,
                    self._get_default_data_generator(category),
                    count_per_variation=docs_per_variation
                )
                
                dataset_results[template_name] = []
                for variation_name, files in results.items():
                    dataset_results[template_name].extend(files)
        
        return dataset_results
    
    def _get_template_category(self, template_name: str) -> Optional[str]:
        """Determine the category of a template based on its name"""
        # Simple heuristic - look for known category keywords
        category_keywords = {
            'legal': ['legal', 'pleading', 'court', 'complaint', 'motion', 'order'],
            'healthcare': ['healthcare', 'medical', 'cms', 'lab', 'eob', 'insurance'],
            'real_estate': ['real_estate', 'deed', 'closing', 'disclosure', 'property'],
            'financial': ['financial', 'bank', 'statement', 'transaction']
        }
        
        template_lower = template_name.lower()
        for category, keywords in category_keywords.items():
            if any(keyword in template_lower for keyword in keywords):
                return category
        
        return None
    
    def _get_default_variations(self, category: str) -> Dict[str, List[Any]]:
        """Get default variations for a category"""
        variations = {
            'font_family': [
                "'Courier New', Courier, monospace",
                "Arial, sans-serif", 
                "'Times New Roman', serif",
                "Helvetica, sans-serif",
                "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
            ],
            'font_size': ["10px", "11px", "12px", "14px", "16px"],
            'background': ["white", "#fafafa", "#f0f0f0", "#f9f9f9", "#ffffff"],
            'border': ["none", "1px solid #000", "1px solid #ccc", "2px solid #333", "1px dashed #999"]
        }
        
        if category == 'legal':
            variations['font_size'] = ["10pt", "11pt", "12pt", "14pt"]
            variations['line_height'] = ["20pt", "22pt", "24pt", "26pt", "28pt"]
            variations['components'] = [
                ["line_numbers", "header", "caption_box", "footer"],
                ["header", "caption_box", "content", "footer"],
                ["line_numbers", "content", "footer"],
                ["header", "content", "signature"],
                ["header", "caption_box", "line_numbers", "signature", "footer"]
            ]
        elif category == 'healthcare':
            variations['font_size'] = ["8px", "9px", "10px", "11px"]
            variations['layout_type'] = ["grid", "form", "table"]
            variations['color'] = ["#000", "#222", "#333", "#444"]
        elif category == 'real_estate':
            variations['font_family'] = ["'Times New Roman', serif", "Georgia, serif", "Arial, sans-serif"]
            variations['font_size'] = ["10px", "11px", "12px"]
            variations['layout_type'] = ["form", "narrative"]
        
        return variations
    
    def _get_default_data_generator(self, category: str):
        """Get a default data generator function for a category"""
        from faker import Faker
        fake = Faker()
        
        def legal_data_generator(index):
            return {
                "plaintiff": fake.name(),
                "defendant": fake.company(),
                "jurisdiction": fake.city(),
                "case_number": f"CIV-{fake.random_int(10000, 99999)}",
                "court": f"{fake.state_abbr()} Court",
                "content": fake.paragraphs(nb=5)
            }
        
        def healthcare_data_generator(index):
            return {
                "patient_name": fake.name(),
                "dob": str(fake.date_of_birth()),
                "insurance_id": f"X{fake.random_int(100000, 999999)}",
                "provider": fake.company(),
                "date_service": str(fake.date_this_year()),
                "cpt_code": f"{fake.random_int(10000, 99999)}",
                "charge_amount": f"${fake.random_int(50, 1000):,}",
                "table_rows": [
                    {"service_date": str(fake.date_this_year()), "procedure": fake.word(), "amount": f"${fake.random_int(50, 500)}"},
                    {"service_date": str(fake.date_this_year()), "procedure": fake.word(), "amount": f"${fake.random_int(50, 500)}"},
                    {"service_date": str(fake.date_this_year()), "procedure": fake.word(), "amount": f"${fake.random_int(50, 500)}"}
                ]
            }
        
        def real_estate_data_generator(index):
            return {
                "property_address": fake.address().replace('\n', ', '),
                "grantor": fake.name(),
                "grantee": fake.name(),
                "sale_price": f"${fake.random_int(100000, 1000000):,}",
                "apn_number": f"{fake.random_int(100, 999)}-{fake.random_int(100, 999)}-{fake.random_int(10, 99)}",
                "county": fake.city(),
                "legal_description": f"LOT {fake.random_int(1, 50)} IN BLOCK {fake.random_int(1, 10)} OF {fake.last_name()} SUBDIVISION"
            }
        
        def financial_data_generator(index):
            return {
                "account_number": f"****{fake.random_int(1000, 9999)}",
                "start_date": str(fake.date_this_month()),
                "end_date": str(fake.date_this_month()),
                "balance": f"${fake.random_int(1000, 100000):,}",
                "transactions": [
                    {"date": str(fake.date_this_month()), "description": fake.sentence(nb_words=3), "amount": f"${fake.random_int(10, 200)}"},
                    {"date": str(fake.date_this_month()), "description": fake.sentence(nb_words=3), "amount": f"${fake.random_int(10, 200)}"},
                    {"date": str(fake.date_this_month()), "description": fake.sentence(nb_words=3), "amount": f"${fake.random_int(10, 200)}"}
                ]
            }
        
        generators = {
            'legal': legal_data_generator,
            'healthcare': healthcare_data_generator,
            'real_estate': real_estate_data_generator,
            'financial': financial_data_generator
        }
        
        return generators.get(category, lambda i: {"index": i, "data": fake.paragraph()})