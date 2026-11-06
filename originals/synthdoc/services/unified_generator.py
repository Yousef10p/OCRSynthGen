"""
Unified Document Generator that leverages the advanced template system as the primary approach
while maintaining backward compatibility with legacy generators.
"""
import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from faker import Faker

from jinja2 import Template
from synthdoc.templates import registry, config_manager, BatchGenerator
from .pdf_exporter import PDFExporter


class UnifiedDocumentGenerator:
    """
    Unified generator that primarily uses the advanced template system while maintaining
    compatibility with legacy generators for backward compatibility.
    """
    def __init__(self):
        self.fake = Faker()

        # Initialize batch generator
        self.batch_generator = BatchGenerator()
    
    
    def generate_document_with_template(self, 
                                       template_name: str, 
                                       config_name: str = None, 
                                       include_pdf: bool = False,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a document using the advanced template system.
        
        Args:
            template_name: Name of the template to use (e.g., 'legal_pleading', 'cms_1500')
            config_name: Configuration name to use (defaults to template_default)
            include_pdf: Whether to also generate a PDF version
            **kwargs: Data to pass to the template
            
        Returns:
            Dictionary containing paths to generated files and metadata
        """
        # Get the appropriate config
        if config_name is None:
            config_name = f"{template_name}_default"
        
        config = config_manager.get_config(config_name)
        
        # Create template instance
        template_class = registry.get_template(template_name)
        template_instance = template_class(config)

        # If no data provided, check if template has a generate_sample_data method
        if not kwargs and hasattr(template_instance, 'generate_sample_data'):
            kwargs = template_instance.generate_sample_data()

        # Render document
        html_content = template_instance.render(kwargs)

        result = {
            "html_content": html_content,
            "template_name": template_name,
            "config_name": config_name,
            "template_data": kwargs
        }

        return result
    
    def generate_batch_with_templates(self, 
                                     template_name: str, 
                                     count: int, 
                                     config_names: List[str] = None,
                                     include_pdf: bool = False,
                                     **kwargs) -> List[Dict[str, Any]]:
        """
        Generate a batch of documents using the template system with different configurations.
        
        Args:
            template_name: Name of the template to use
            count: Number of documents to generate
            config_names: List of configuration names to use (cycled if fewer than count)
            include_pdf: Whether to also generate PDF versions
            **kwargs: Base data to pass to all documents (will be varied per document)
            
        Returns:
            List of dictionaries containing paths to generated files and metadata
        """
        results = []
        
        if config_names is None:
            # Use default config for all documents
            config_names = [f"{template_name}_default"] * count
        else:
            # Cycle through provided configs
            config_names = [config_names[i % len(config_names)] for i in range(count)]
        
        for i in range(count):
            config_name = config_names[i]

            # Get template class and instance
            template_class = registry.get_template(template_name)
            config = config_manager.get_config(config_name)
            template_instance = template_class(config)

            # If base kwargs is empty, check if template has a generate_sample_data method
            if not kwargs and hasattr(template_instance, 'generate_sample_data'):
                doc_kwargs = template_instance.generate_sample_data()
            else:
                doc_kwargs = kwargs.copy()

            # Add index-specific data if needed
            doc_kwargs['index'] = i

            # Add document-specific randomness
            doc_kwargs['id'] = self.fake.random_int(10000, 99999)
            
            doc = self.generate_document_with_template(
                template_name, 
                config_name, 
                include_pdf=include_pdf,
                **doc_kwargs
            )
            results.append(doc)
        
        return results
    
    def generate_layout_segmentation_dataset(self,
                                           template_categories: List[str],
                                           variations_per_template: int = 5,
                                           docs_per_variation: int = 3,
                                           output_dir: str = None) -> Dict[str, List[str]]:
        """
        Generate a layout segmentation dataset using the advanced batch generator.
        
        Args:
            template_categories: List of template categories to include
            variations_per_template: Number of style/layout variations per template
            docs_per_variation: Number of documents per variation
            output_dir: Output directory (defaults to generated_documents)
            
        Returns:
            Dictionary mapping template names to lists of generated file paths
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        # Use the batch generator to create a comprehensive dataset
        return self.batch_generator.generate_layout_segmentation_dataset(
            template_categories=template_categories,
            variations_per_template=variations_per_template,
            docs_per_variation=docs_per_variation
        )
    
    def generate_template_variations(self,
                                   base_template_name: str,
                                   variation_params: Dict[str, List[Any]],
                                   doc_data_func: callable = None,
                                   count_per_variation: int = 1) -> Dict[str, List[str]]:
        """
        Generate documents using template variations based on parameter options.
        
        Args:
            base_template_name: Name of the base template to vary
            variation_params: Dictionary mapping config fields to lists of possible values
            doc_data_func: Function to generate document data for each variation
            count_per_variation: Number of documents to generate per variation
            
        Returns:
            Dictionary mapping configuration names to lists of generated file paths
        """
        if doc_data_func is None:
            # Default data function that creates basic document data
            def doc_data_func(i):
                return {
                    "id": self.fake.random_int(10000, 99999),
                    "index": i
                }
        
        return self.batch_generator.generate_template_variations(
            base_template_name,
            variation_params,
            doc_data_func,
            count_per_variation=count_per_variation
        )
    
    def get_available_templates(self) -> Dict[str, List[str]]:
        """Get all available templates by category."""
        result = {}
        for category in registry.get_categories():
            templates = registry.list_templates(category)
            if templates:
                result[category] = templates
        return result
    
    def get_available_configs(self, category: str = None) -> List[str]:
        """Get all available configurations."""
        return config_manager.list_configs(category)
    
    def create_config_from_params(self, config_name: str, **params) -> str:
        """Create a new configuration with specified parameters."""
        config = config_manager.create_config_from_params(config_name, **params)
        return config_name


# Legacy compatibility wrapper
class LegacyDocumentGenerator:
    """
    Legacy generator classes that can be called when specific document types are requested
    but implemented using the template system for consistency.
    """
    def __init__(self):
        self.generator = UnifiedDocumentGenerator()
    
    def generate_pleading(self, include_pdf: bool = False, **kwargs):
        """Generate a legal pleading document."""
        return self.generator.generate_document_with_template(
            "legal_pleading", 
            include_pdf=include_pdf,
            **kwargs
        )
    
    def generate_healthcare_document(self, doc_type: str = "BOTH", include_pdf: bool = False, **kwargs):
        """Generate healthcare documents (CMS-1500 form and/or lab report)."""
        results = []
        
        if doc_type in ["BOTH", "FORM"]:
            cms_result = self.generator.generate_document_with_template(
                "cms_1500",
                config_name="healthcare_default",
                include_pdf=include_pdf,
                **kwargs
            )
            results.append(cms_result)
        
        if doc_type in ["BOTH", "REPORT"]:
            lab_result = self.generator.generate_document_with_template(
                "lab_report",
                config_name="healthcare_default",
                include_pdf=include_pdf,
                **kwargs
            )
            results.append(lab_result)
        
        return results if len(results) > 1 else results[0]
    
    def generate_real_estate_document(self, doc_type: str = "BOTH", include_pdf: bool = False, **kwargs):
        """Generate real estate documents (closing disclosure and/or warranty deed)."""
        results = []
        
        if doc_type in ["BOTH", "CLOSING"]:
            closing_result = self.generator.generate_document_with_template(
                "closing_disclosure",
                config_name="real_estate_default",
                include_pdf=include_pdf,
                **kwargs
            )
            results.append(closing_result)
        
        if doc_type in ["BOTH", "DEED"]:
            deed_result = self.generator.generate_document_with_template(
                "warranty_deed",
                config_name="real_estate_default",
                include_pdf=include_pdf,
                **kwargs
            )
            results.append(deed_result)
        
        return results if len(results) > 1 else results[0]


class IntegratedDocumentGenerator:
    """
    Main integrated generator that prioritizes the template system but provides
    backward compatibility with legacy generators.
    """
    def __init__(self):
        self.unified_gen = UnifiedDocumentGenerator()
        self.legacy_gen = LegacyDocumentGenerator()
    
    def generate_document(self, doc_type: str, **kwargs):
        """
        Generate a document of specified type, using template system when possible.
        
        Args:
            doc_type: Document type (e.g., 'legal', 'healthcare', 'real_estate', or template names)
            **kwargs: Additional parameters for document generation
            
        Returns:
            Dictionary with file paths and metadata
        """
        # Check if doc_type matches a specific template name first
        try:
            # Try to find an exact template match
            available_templates = self.unified_gen.get_available_templates()
            all_templates = []
            for templates in available_templates.values():
                all_templates.extend(templates)
            
            if doc_type in all_templates:
                return self.unified_gen.generate_document_with_template(doc_type, **kwargs)
        except ValueError:
            # Template not found, continue to legacy approach
            pass
        
        # Map high-level types to their templates
        type_to_template = {
            'legal': 'legal_pleading',
            'healthcare': 'cms_1500',  # Default to CMS-1500 for healthcare
            'real_estate': 'closing_disclosure',  # Default to closing disclosure for real estate
            'cms1500': 'cms_1500',
            'lab_report': 'lab_report',
            'closing_disclosure': 'closing_disclosure',
            'warranty_deed': 'warranty_deed',
            'bank_statement': 'bank_statement'
        }
        
        if doc_type.lower() in type_to_template:
            template_name = type_to_template[doc_type.lower()]
            return self.unified_gen.generate_document_with_template(template_name, **kwargs)
        
        # Fall back to legacy generators for backward compatibility
        if doc_type.lower() == "legal":
            return self.legacy_gen.generate_pleading(**kwargs)
        elif doc_type.lower() == "healthcare":
            doc_type_param = kwargs.get('doc_type', 'BOTH')
            return self.legacy_gen.generate_healthcare_document(doc_type=doc_type_param, **kwargs)
        elif doc_type.lower() == "real_estate":
            doc_type_param = kwargs.get('doc_type', 'BOTH')
            return self.legacy_gen.generate_real_estate_document(doc_type=doc_type_param, **kwargs)
        else:
            # If no match found, try the template name directly (might raise ValueError)
            return self.unified_gen.generate_document_with_template(doc_type, **kwargs)
    
    def generate_document_with_pdf(self, doc_type: str, include_pdf: bool = True, **kwargs):
        """
        Generate a document and optionally convert it to PDF.
        """
        return self.generate_document(doc_type, include_pdf=include_pdf, **kwargs)
    
    def generate_batch(self, doc_types: List[str], count: int = 1, include_pdf: bool = False, **kwargs):
        """
        Generate a batch of documents of specified types.
        """
        results = []
        for doc_type in doc_types:
            for i in range(count):
                doc_kwargs = kwargs.copy()
                doc_kwargs['index'] = i
                doc = self.generate_document(doc_type, include_pdf=include_pdf, **doc_kwargs)
                results.append(doc)
        return results


if __name__ == "__main__":
    # Example usage
    generator = IntegratedDocumentGenerator()
    
    # Generate documents using the unified approach
    print("=== Testing Unified Document Generator ===")
    
    # Generate a legal document using template system
    legal_doc = generator.generate_document_with_pdf("legal_pleading", include_pdf=True)
    print(f"Generated legal document: {legal_doc['html_file']}")
    if legal_doc.get('pdf_file'):
        print(f"PDF version: {legal_doc['pdf_file']}")
    
    # Generate using high-level type (backward compatibility)
    healthcare_doc = generator.generate_document_with_pdf("healthcare", include_pdf=True)
    print(f"Generated healthcare document: {healthcare_doc['html_file']}")
    
    # Generate batch using advanced features
    print("\n=== Testing Advanced Batch Generation ===")
    batch_gen = generator.unified_gen
    
    # Generate variations
    variations = batch_gen.generate_template_variations(
        "legal_pleading",
        {
            "font_family": ["Arial, sans-serif", "'Times New Roman', serif"],
            "font_size": ["10pt", "12pt"],
        },
        lambda i: {
            "plaintiff": batch_gen.fake.name(),
            "defendant": batch_gen.fake.company(),
            "jurisdiction": batch_gen.fake.city(),
            "case_number": f"CIV-{batch_gen.fake.random_int(10000, 99999)}",
            "court": f"SUPERIOR COURT OF THE STATE OF CALIFORNIA, COUNTY OF {batch_gen.fake.city().upper()}",
            "doc_type": "COMPLAINT",
            "content": batch_gen.fake.paragraphs(nb=2)
        },
        count_per_variation=2
    )
    
    print(f"Generated {len(variations)} variation sets")
    for config_name, files in variations.items():
        print(f"  {config_name}: {len(files)} documents")