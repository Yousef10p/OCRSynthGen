"""
Ontology-based type mapper for RDFa extraction

Dynamically loads Schema.org and DOCO ontologies to map RDFa types and properties
to entity types, eliminating hard-coded mappings.
"""
from typing import Dict, Optional, Set
from rdflib import Graph, Namespace, URIRef, RDF, RDFS
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


# Define namespaces
SCHEMA = Namespace("http://schema.org/")
SCHEMA_HTTPS = Namespace("https://schema.org/")
DOCO = Namespace("http://purl.org/spar/doco/")
DOCO_FD = Namespace("http://purl.org/doco-fd/")


class OntologyMapper:
    """
    Maps RDF types and properties to NER entity types using actual ontology definitions.
    
    This replaces hard-coded mappings with dynamic ontology-based inference.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize ontology mapper.
        
        Args:
            cache_dir: Directory to cache downloaded ontologies (optional)
        """
        self.cache_dir = cache_dir or Path(__file__).parent / "ontology_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # RDF graphs for ontologies
        self.schema_graph: Optional[Graph] = None
        self.doco_graph: Optional[Graph] = None
        
        # Mapping caches
        self._type_to_entity_cache: Dict[str, str] = {}
        self._property_to_entity_cache: Dict[str, str] = {}
        
        # Default mappings (fallback if ontology not loaded)
        self._default_type_map = {
            'Person': 'PERSON',
            'Patient': 'PERSON',
            'Physician': 'PERSON',
            'Organization': 'ORG',
            'EducationalOrganization': 'ORG',
            'GovernmentOrganization': 'ORG',
            'MedicalOrganization': 'ORG',
            'BankOrCreditUnion': 'ORG',
            'Airline': 'ORG',
            'Hotel': 'ORG',
            'Hospital': 'ORG',
            'PostalAddress': 'ADDRESS',
            'Place': 'LOCATION',
            'Airport': 'LOCATION',
            'Country': 'LOCATION',
            'MonetaryAmount': 'AMOUNT',
            'Date': 'DATE',
            'Duration': 'DURATION',
            'Distance': 'QUANTITY',
            'QuantitativeValue': 'QUANTITY',
            'Rating': 'QUANTITY',
            'Vehicle': 'FIELD_LABEL',
            'Car': 'FIELD_LABEL',
            'Product': 'FIELD_LABEL',
            'Drug': 'FIELD_LABEL',
            'MedicalCondition': 'FIELD_LABEL',
            'FinancialProduct': 'FIELD_LABEL',
            'BankAccount': 'FIELD_LABEL',
            'LoanOrCredit': 'FIELD_LABEL',
            'InsurancePolicy': 'FIELD_LABEL',
        }
        
        self._default_property_map = {
            # --- Person ---
            # 'name' is deliberately absent: schema:name names whatever its subject is
            # (a person, a company, a product line), so it resolves through
            # _subject_named_properties instead.
            'givenName': 'PERSON',
            'familyName': 'PERSON',
            'additionalName': 'PERSON',
            'honorificSuffix': 'PERSON',
            'author': 'PERSON',
            'attendee': 'PERSON',
            'performer': 'PERSON',
            # --- Organization ---
            'legalName': 'ORG',
            'issuingAuthority': 'ORG',
            'manufacturer': 'ORG',
            'brand': 'ORG',
            'provider': 'ORG',
            'department': 'ORG',
            'worksFor': 'ORG',
            'publisher': 'ORG',
            'creditor': 'ORG',
            'broker': 'ORG',
            'legislationPassedBy': 'ORG',
            # --- Date (calendar dates) ---
            'date': 'DATE',
            'birthDate': 'DATE',
            'deathDate': 'DATE',
            'startDate': 'DATE',
            'endDate': 'DATE',
            'dateIssued': 'DATE',
            'dateCreated': 'DATE',
            'datePublished': 'DATE',
            'dateModified': 'DATE',
            'dateSent': 'DATE',
            'dateReceived': 'DATE',
            'dateSigned': 'DATE',
            'dateEffective': 'DATE',
            'dateRange': 'DATE',
            'expires': 'DATE',
            'issueDate': 'DATE',
            'validFrom': 'DATE',
            'validThrough': 'DATE',
            'validUntil': 'DATE',
            'validFor': 'DURATION',
            'paymentDueDate': 'DATE',
            'expectedArrivalFrom': 'DATE',
            'foundingDate': 'DATE',
            'productionDate': 'DATE',
            'orderDate': 'DATE',
            'scheduledDate': 'DATE',
            'temporalCoverage': 'DATE',
            'temporal': 'DATE',
            'vehicleModelDate': 'DATE',
            'modelDate': 'DATE',
            # --- Time (clock times) ---
            'time': 'TIME',
            'startTime': 'TIME',
            'endTime': 'TIME',
            'scheduledTime': 'TIME',
            'departureTime': 'TIME',
            'arrivalTime': 'TIME',
            'checkinTime': 'TIME',
            'checkoutTime': 'TIME',
            'bookingTime': 'TIME',
            'doorTime': 'TIME',
            # --- Duration (time spans) ---
            'duration': 'DURATION',
            'loanTerm': 'DURATION',
            # --- ID ---
            'identifier': 'ID',
            'trackingNumber': 'ID',
            'orderNumber': 'ID',
            'postalCode': 'ID',
            'taxID': 'ID',
            'serialNumber': 'ID',
            'vehicleIdentificationNumber': 'ID',
            'legislationIdentifier': 'ID',
            'reservationId': 'ID',
            'reservationNumber': 'ID',
            'productID': 'ID',
            'courseCode': 'ID',
            'code': 'ID',
            'codeValue': 'ID',
            'version': 'ID',
            'pagination': 'ID',
            'flightNumber': 'ID',
            'iataCode': 'ID',
            'sku': 'ID',
            'tickerSymbol': 'ID',
            'accountId': 'ID',
            'barcode': 'ID',
            # --- Address/Location ---
            'streetAddress': 'ADDRESS',
            'address': 'ADDRESS',
            'addressLocality': 'LOCATION',
            'addressRegion': 'LOCATION',
            'addressCountry': 'LOCATION',
            'location': 'LOCATION',
            'birthPlace': 'LOCATION',
            'nationality': 'LOCATION',
            'containedInPlace': 'LOCATION',
            'workLocation': 'LOCATION',
            'areaServed': 'LOCATION',
            'departureAirport': 'LOCATION',
            'arrivalAirport': 'LOCATION',
            'legislationJurisdiction': 'LOCATION',
            # --- Phone ---
            'telephone': 'PHONE',
            'faxNumber': 'PHONE',
            'contactPoint': 'PHONE',
            # --- Email ---
            'email': 'EMAIL',
            # --- URL ---
            'url': 'URL',
            'sameAs': 'URL',
            'image': 'URL',
            'logo': 'URL',
            # --- Amount/Money ---
            'value': 'AMOUNT',
            'price': 'AMOUNT',
            'amount': 'AMOUNT',
            'discount': 'AMOUNT',
            'totalPrice': 'AMOUNT',
            'baseSalary': 'AMOUNT',
            'totalPaymentDue': 'AMOUNT',
            'interestRate': 'AMOUNT',
            'taxRate': 'AMOUNT',
            'starRating': 'AMOUNT',
            # --- Quantity/Measurement ---
            'weight': 'QUANTITY',
            'width': 'QUANTITY',
            'height': 'QUANTITY',
            'depth': 'QUANTITY',
            'distance': 'QUANTITY',
            'floorSize': 'QUANTITY',
            'mileageFromOdometer': 'QUANTITY',
            'numberOfCredits': 'QUANTITY',
            'numberOfItems': 'QUANTITY',
            'numberOfPages': 'QUANTITY',
            'numberOfEmployees': 'QUANTITY',
            'numberOfRooms': 'QUANTITY',
            'numberOfDays': 'QUANTITY',
            'orderQuantity': 'QUANTITY',
            'ratingValue': 'QUANTITY',
            'age': 'QUANTITY',
            'frequency': 'QUANTITY',
            'percentage': 'QUANTITY',
            'doseValue': 'QUANTITY',
            'occupancy': 'QUANTITY',
            'wordCount': 'QUANTITY',
            # --- Boolean ---
            'boolean': 'BOOLEAN',
            # --- Text (field values and generic content) ---
            'description': 'TEXT',
            'text': 'TEXT',
            'abstract': 'TEXT',
            'about': 'TEXT',
            'comment': 'TEXT',
            'instructions': 'TEXT',
            'termsOfService': 'TEXT',
            'jobTitle': 'TEXT',
            'hasCredential': 'TEXT',
            'slogan': 'TEXT',
            'deliveryMethod': 'TEXT',
            'roleName': 'TEXT',
            'educationalCredentialAwarded': 'TEXT',
            'educationalLevel': 'TEXT',
            'educationalProgramMode': 'TEXT',
            'credentialCategory': 'TEXT',
            'unitText': 'TEXT',
            'model': 'TEXT',
            'color': 'TEXT',
            'paymentMethod': 'TEXT',
            'gender': 'TEXT',
            'maritalStatus': 'TEXT',
            'employmentType': 'TEXT',
            'employmentStatus': 'TEXT',
            'accountType': 'TEXT',
            'category': 'TEXT',
            'additionalType': 'TEXT',
            'termOffered': 'TEXT',
            'openingHours': 'TEXT',
            'hoursAvailable': 'TEXT',
            'workHours': 'TEXT',
            'vehicleConfiguration': 'TEXT',
            'medicalSpecialty': 'TEXT',
            'dosageForm': 'TEXT',
            'dosageInstructions': 'TEXT',
            'activeIngredient': 'TEXT',
            'strength': 'TEXT',
            'diagnosis': 'TEXT',
            'healthCondition': 'TEXT',
            'causeOfDeath': 'TEXT',
            'roomType': 'TEXT',
            'seatNumber': 'TEXT',
            'itemCondition': 'TEXT',
            'orderStatus': 'TEXT',
            'reservationStatus': 'TEXT',
            'actionStatus': 'TEXT',
            'eventType': 'TEXT',
            'occupationalCategory': 'TEXT',
            'position': 'TEXT',
            'responsibilities': 'TEXT',
            'benefits': 'TEXT',
            'knowsAbout': 'TEXT',
            'award': 'TEXT',
            'amenityFeature': 'TEXT',
            'relationship': 'TEXT',
            'hasOccupation': 'TEXT',
            'aircraft': 'TEXT',
            'alternateName': 'TEXT',
            'taxYear': 'ID',
        }

        # Properties whose entity type is that of their subject: the nearest enclosing
        # schema.org typeof. Only named-entity subjects count, so the name of an
        # Invoice line or a FinancialProduct stays TEXT.
        self._subject_named_properties = ['name']
        self._named_entity_types = ['PERSON', 'ORG', 'LOCATION']

        # Structural DOCO properties (not semantic entities)
        self._structural_properties = {
            'fieldLabel': 'FIELD_LABEL',
            'columnHeader': 'TABLE_HEADER',
            'rowHeader': 'TABLE_HEADER',
            'sectionTitle': 'SECTION_TITLE',
            'title': 'TITLE',
            'caption': 'CAPTION',
            'cellValue': 'TABLE_CELL',
            'tableCell': 'TABLE_CELL',
        }
    
    def load_schema_org(self, force_reload: bool = False) -> bool:
        """
        Load Schema.org ontology.
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            True if successfully loaded
        """
        if self.schema_graph and not force_reload:
            return True
        
        cache_file = self.cache_dir / "schema_org.ttl"
        
        try:
            self.schema_graph = Graph()
            
            if cache_file.exists() and not force_reload:
                logger.info(f"Loading Schema.org from cache: {cache_file}")
                self.schema_graph.parse(cache_file, format='turtle')
            else:
                logger.info("Downloading Schema.org ontology...")
                # Load from schema.org
                self.schema_graph.parse("https://schema.org/version/latest/schemaorg-current-https.ttl", format='turtle')
                
                # Cache it
                self.schema_graph.serialize(destination=str(cache_file), format='turtle')
                logger.info(f"Cached Schema.org to: {cache_file}")
            
            logger.info(f"Loaded Schema.org: {len(self.schema_graph)} triples")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load Schema.org ontology: {e}")
            logger.info("Will use default type mappings as fallback")
            return False
    
    def load_doco(self, force_reload: bool = False) -> bool:
        """
        Load DOCO ontology.
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            True if successfully loaded
        """
        if self.doco_graph and not force_reload:
            return True
        
        cache_file = self.cache_dir / "doco.ttl"
        
        try:
            self.doco_graph = Graph()
            
            if cache_file.exists() and not force_reload:
                logger.info(f"Loading DOCO from cache: {cache_file}")
                self.doco_graph.parse(cache_file, format='turtle')
            else:
                logger.info("Downloading DOCO ontology...")
                # Load from SPAR ontologies
                self.doco_graph.parse("http://purl.org/spar/doco", format='xml')
                
                # Cache it
                self.doco_graph.serialize(destination=str(cache_file), format='turtle')
                logger.info(f"Cached DOCO to: {cache_file}")
            
            logger.info(f"Loaded DOCO: {len(self.doco_graph)} triples")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load DOCO ontology: {e}")
            logger.info("Will use default structural mappings as fallback")
            return False
    
    def map_type_to_entity(self, rdf_type: str, namespace: str = "schema") -> str:
        """
        Map an RDF type to an entity type.
        
        Args:
            rdf_type: The type name (e.g., "Person", "Organization")
            namespace: The namespace ("schema", "doco", "doco-fd")
            
        Returns:
            Entity type string (e.g., "PERSON", "ORG")
        """
        # Check cache first
        cache_key = f"{namespace}:{rdf_type}"
        if cache_key in self._type_to_entity_cache:
            return self._type_to_entity_cache[cache_key]
        
        # DOCO/DOCO-FD types are structural, not semantic entities
        if namespace in ("doco", "doco-fd"):
            # These don't map to entity types
            return "TEXT"
        
        # Schema.org types
        if namespace == "schema":
            entity_type = self._map_schema_type(rdf_type)
            self._type_to_entity_cache[cache_key] = entity_type
            return entity_type
        
        return "TEXT"
    
    def _map_schema_type(self, type_name: str) -> str:
        """
        Map a Schema.org type to entity type using ontology or defaults.
        
        Args:
            type_name: Schema.org type name
            
        Returns:
            Entity type
        """
        # Try default mapping first (fastest)
        if type_name in self._default_type_map:
            return self._default_type_map[type_name]
        
        # If ontology loaded, check for subclass relationships
        if self.schema_graph:
            type_uri = SCHEMA_HTTPS[type_name]
            
            # Check if it's a subclass of known types
            for parent_type, entity_type in self._default_type_map.items():
                parent_uri = SCHEMA_HTTPS[parent_type]
                
                # Check direct subclass or transitive
                if self._is_subclass_of(type_uri, parent_uri):
                    return entity_type
        
        return "TEXT"
    
    def _is_subclass_of(self, type_uri: URIRef, parent_uri: URIRef) -> bool:
        """
        Check if type_uri is a subclass of parent_uri.
        
        Args:
            type_uri: The type to check
            parent_uri: The potential parent type
            
        Returns:
            True if type_uri is subclass of parent_uri
        """
        if not self.schema_graph:
            return False
        
        # Check direct subclass
        if (type_uri, RDFS.subClassOf, parent_uri) in self.schema_graph:
            return True
        
        # Check transitive (up to 3 levels to avoid infinite recursion)
        for _ in range(3):
            for parent in self.schema_graph.objects(type_uri, RDFS.subClassOf):
                if parent == parent_uri:
                    return True
                type_uri = parent
        
        return False
    
    def map_property_to_entity(self, property_name: str, namespace: str = "schema") -> str:
        """
        Map an RDF property to an entity type.
        
        Args:
            property_name: The property name (e.g., "name", "birthDate")
            namespace: The namespace ("schema", "doco", "doco-fd")
            
        Returns:
            Entity type string (e.g., "PERSON", "DATE")
        """
        # Check cache first
        cache_key = f"{namespace}:{property_name}"
        if cache_key in self._property_to_entity_cache:
            return self._property_to_entity_cache[cache_key]
        
        # DOCO/DOCO-FD structural properties
        if namespace in ("doco", "doco-fd"):
            entity_type = self._structural_properties.get(property_name, "TEXT")
            self._property_to_entity_cache[cache_key] = entity_type
            return entity_type
        
        # Schema.org properties
        if namespace == "schema":
            entity_type = self._map_schema_property(property_name)
            self._property_to_entity_cache[cache_key] = entity_type
            return entity_type
        
        return "TEXT"
    
    def _map_schema_property(self, property_name: str) -> str:
        """
        Map a Schema.org property to entity type using ontology or defaults.
        
        Args:
            property_name: Schema.org property name
            
        Returns:
            Entity type
        """
        # Try default mapping first
        if property_name in self._default_property_map:
            return self._default_property_map[property_name]
        
        # If ontology loaded, check property range (expected type)
        if self.schema_graph:
            prop_uri = SCHEMA_HTTPS[property_name]
            
            # Get range (expected type) of property
            for range_type in self.schema_graph.objects(prop_uri, SCHEMA_HTTPS.rangeIncludes):
                type_name = range_type.split('/')[-1]
                if type_name in self._default_type_map:
                    return self._default_type_map[type_name]
        
        return "TEXT"
    
    def generate_javascript_config(self) -> str:
        """
        Generate JavaScript configuration object for use in pagedjs_renderer.
        
        Returns:
            JavaScript code string with type and property mappings
        """
        # Load ontologies if not already loaded (non-blocking)
        try:
            self.load_schema_org()
            self.load_doco()
        except:
            pass
        
        config = {
            'typeMap': self._default_type_map,
            'propertyMap': self._default_property_map,
            'structuralProperties': self._structural_properties,
            'subjectNamedProperties': self._subject_named_properties,
            'namedEntityTypes': self._named_entity_types,
        }
        
        # Generate JavaScript
        js_code = f"""
// Auto-generated ontology mappings
const ONTOLOGY_CONFIG = {json.dumps(config, indent=2)};

// Helper: Get Schema.org type from typeof attribute
function getSchemaType(el) {{
    const typeofAttr = el.getAttribute('typeof');
    if (!typeofAttr) return null;

    const types = typeofAttr.split(/\\s+/);

    for (const type of types) {{
        // Skip DOCO types (both doco: and doco-fd:)
        if (type.startsWith('doco:') || type.startsWith('doco-fd:')) continue;

        const cleanType = type.replace(/^schema:/, '');
        
        if (ONTOLOGY_CONFIG.typeMap[cleanType]) {{
            return ONTOLOGY_CONFIG.typeMap[cleanType];
        }}
    }}

    return null;
}}

// Helper: Get property name from property attribute
function getPropertyName(el) {{
    const propertyAttr = el.getAttribute('property');
    if (!propertyAttr) return null;

    const properties = propertyAttr.split(/\\s+/);

    for (const prop of properties) {{
        if (prop.startsWith('schema:')) {{
            return prop.replace(/^schema:/, '');
        }}
    }}

    return null;
}}

// Helper: Infer entity type from Schema.org property name
function inferEntityTypeFromProperty(propertyName) {{
    if (!propertyName) return 'TEXT';
    return ONTOLOGY_CONFIG.propertyMap[propertyName] || 'TEXT';
}}

// Helper: Entity type of a subject-named property (schema:name), taken from the
// nearest schema.org typeof at any depth. Null unless the subject is a named entity,
// so typeof="schema:Text" on a field stops it inheriting a Person from its section.
function inferEntityTypeFromSubject(el) {{
    for (let node = el; node && node.getAttribute; node = node.parentElement) {{
        const typeofAttr = node.getAttribute('typeof') || '';
        if (!/(^|\\s)schema:/.test(typeofAttr)) continue;
        const type = getSchemaType(node);
        return ONTOLOGY_CONFIG.namedEntityTypes.includes(type) ? type : null;
    }}
    return null;
}}

// Helper: Map DOCO structural property to entity type
function mapStructuralProperty(propertyAttr) {{
    for (const [propName, entityType] of Object.entries(ONTOLOGY_CONFIG.structuralProperties)) {{
        if (propertyAttr.includes(`doco:${{propName}}`)) return entityType;
        if (propertyAttr.includes(`doco-fd:${{propName}}`)) return entityType;
    }}
    return null;
}}
"""
        return js_code


# Global singleton instance
_mapper_instance: Optional[OntologyMapper] = None


def get_ontology_mapper() -> OntologyMapper:
    """Get or create global ontology mapper instance."""
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = OntologyMapper()
        # Try to load ontologies in background (non-blocking)
        try:
            _mapper_instance.load_schema_org()
        except:
            pass
    return _mapper_instance
