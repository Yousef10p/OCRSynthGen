"""
RDF Graph Structure Analysis and Recommendations

Analyzes template RDFa patterns and suggests improvements for:
1. Reducing redundant "fieldValue" usage
2. Improving graph connectivity
3. Enhancing visual clarity
"""

# ============================================================
# CURRENT PATTERN ANALYSIS
# ============================================================

CURRENT_PATTERN = """
Your example:
<div typeof="schema:MonetaryAmount doco-fd:KeyValuePair" resource="#tax">
    <p>
        <span property="doco-fd:fieldLabel"><strong>Tax (8.5%):</strong></span>
        <span property="schema:value doco-fd:fieldValue">$0.00</span>
    </p>
</div>

Current Graph:
  #tax (MonetaryAmount, KeyValuePair)
    ├── fieldLabel → "Tax (8.5%):"
    ├── value → "$0.00"
    └── fieldValue → "$0.00"  ← REDUNDANT
"""

# ============================================================
# RECOMMENDATION 1: Property Hierarchy
# ============================================================

PROPERTY_STRATEGY = """
┌─────────────────────────────────────────────────────────────────┐
│ PROPERTY SELECTION STRATEGY (Priority Order)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ 1. SEMANTIC PROPERTY (best)                                     │
│    Use domain-specific property when available:                 │
│    - schema:amount for money                                    │
│    - schema:name for person/org names                          │
│    - schema:streetAddress for addresses                        │
│    - schema:dateCreated for dates                              │
│                                                                  │
│ 2. TYPE-SPECIFIC PROPERTY (good)                               │
│    Use property appropriate to the typeof:                      │
│    - MonetaryAmount → schema:value                             │
│    - Person → schema:givenName, schema:familyName              │
│    - Organization → schema:legalName                           │
│                                                                  │
│ 3. GENERIC PROPERTY (fallback)                                  │
│    Only when no semantic property fits:                         │
│    - rdf:value (standard RDF generic value)                    │
│    - schema:value (schema.org generic)                         │
│    - doco-fd:fieldValue (document-specific, last resort)       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
"""

# ============================================================
# RECOMMENDATION 2: Improved Pattern
# ============================================================

IMPROVED_PATTERN = """
BEFORE (redundant):
─────────────────────────────────────────────────────────────────
<div typeof="schema:MonetaryAmount doco-fd:KeyValuePair" resource="#tax">
    <span property="doco-fd:fieldLabel">Tax:</span>
    <span property="schema:value doco-fd:fieldValue">$0.00</span>
</div>

Graph: #tax → fieldLabel="Tax:" + value="$0.00" + fieldValue="$0.00"


AFTER (clean):
─────────────────────────────────────────────────────────────────
<div typeof="schema:MonetaryAmount" resource="#tax">
    <span property="rdfs:label">Tax</span>
    <span property="schema:value" content="0.00">$0.00</span>
    <span property="schema:currency" content="USD"></span>
</div>

Graph: #tax (MonetaryAmount) → label="Tax" + value="0.00" + currency="USD"


EVEN BETTER (with relationship):
─────────────────────────────────────────────────────────────────
<div typeof="schema:Invoice" resource="#invoice">
    <div rel="schema:totalPaymentDue">
        <div typeof="schema:MonetaryAmount" resource="#total">
            <span property="schema:name">Total</span>
            <span property="schema:value" content="100.00">$100.00</span>
        </div>
    </div>
</div>

Graph: #invoice → totalPaymentDue → #total (MonetaryAmount)
                                            ├── name="Total"
                                            └── value="100.00"
"""

# ============================================================
# RECOMMENDATION 3: Type System
# ============================================================

TYPE_SYSTEM = """
┌────────────────────────────────────────────────────────────────┐
│ TYPE HIERARCHY FOR DOCUMENT FIELDS                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  doco-fd:FormField (abstract base)                             │
│       │                                                         │
│       ├── doco-fd:TextField                                    │
│       │      └── for free-form text                            │
│       │                                                         │
│       ├── doco-fd:NumberField                                  │
│       │      └── for numeric values                            │
│       │                                                         │
│       ├── doco-fd:DateField                                    │
│       │      └── for date inputs                               │
│       │                                                         │
│       ├── doco-fd:MonetaryField  ≈ schema:MonetaryAmount       │
│       │      └── for currency values                           │
│       │                                                         │
│       └── doco-fd:IdentifierField                              │
│              └── for IDs, codes, reference numbers             │
│                                                                 │
│  Instead of: typeof="doco-fd:FormField"                        │
│  Use:      typeof="doco-fd:MonetaryField schema:MonetaryAmount"│
│                                                                 │
└────────────────────────────────────────────────────────────────┘
"""

# ============================================================
# VISUALIZATION APPROACHES
# ============================================================

VISUALIZATION_APPROACHES = """
┌────────────────────────────────────────────────────────────────┐
│ GRAPH VISUALIZATION OPTIONS                                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. STRUCTURAL VIEW (hierarchy-focused)                         │
│    ┌─────────────────┐                                         │
│    │    #invoice     │                                         │
│    │  (Invoice)      │                                         │
│    └────────┬────────┘                                         │
│             │ contains                                          │
│    ┌────────┴────────┐                                         │
│    │ #tax_row        │                                         │
│    │ (TableRow)      │                                         │
│    │ • label: "Tax"  │                                         │
│    │ • value: $8.50  │                                         │
│    └─────────────────┘                                         │
│                                                                 │
│ 2. PROPERTY GRAPH (relationship-focused)                       │
│                                                                 │
│    [invoice] ──totalPaymentDue──▶ [MonetaryAmount]             │
│                                         │                       │
│                              ┌──────────┴──────────┐           │
│                              │                     │           │
│                         value: $100          currency: USD     │
│                                                                 │
│ 3. TABLE VIEW (data-focused)                                   │
│    ┌──────────────┬──────────────────┬───────────────┐        │
│    │ Resource     │ Type             │ Properties    │        │
│    ├──────────────┼──────────────────┼───────────────┤        │
│    │ #tax         │ MonetaryAmount   │ value: $8.50  │        │
│    │              │ KeyValuePair     │ label: "Tax"  │        │
│    └──────────────┴──────────────────┴───────────────┘        │
│                                                                 │
│ 4. RDF TRIPLE LIST (SPARQL-ready)                              │
│    #tax    rdf:type        schema:MonetaryAmount .             │
│    #tax    schema:value    "8.50"^^xsd:decimal .               │
│    #tax    rdfs:label      "Tax" .                             │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
"""

# ============================================================
# QUICK REFERENCE: Property Mappings
# ============================================================

PROPERTY_MAPPINGS = """
┌────────────────────────┬─────────────────────────┬────────────────┐
│ AVOID                  │ PREFER                  │ FOR            │
├────────────────────────┼─────────────────────────┼────────────────┤
│ doco-fd:fieldValue     │ schema:value            │ generic value  │
│ doco-fd:fieldValue     │ schema:amount           │ money amount   │
│ doco-fd:fieldValue     │ schema:price            │ price          │
│ doco-fd:fieldValue     │ schema:identifier       │ IDs/codes      │
│ doco-fd:fieldValue     │ schema:name             │ names          │
│ doco-fd:fieldValue     │ schema:description      │ descriptions   │
│ doco-fd:fieldValue     │ schema:dateCreated      │ dates          │
├────────────────────────┼─────────────────────────┼────────────────┤
│ doco-fd:fieldLabel     │ rdfs:label              │ generic label  │
│ doco-fd:fieldLabel     │ schema:name             │ entity name    │
│ doco-fd:fieldLabel     │ dcterm:title            │ titles         │
├────────────────────────┼─────────────────────────┼────────────────┤
│ Multiple types         │ Primary type +          │ cleaner graph  │
│ on same element        │ rel/rev for relations   │                │
└────────────────────────┴─────────────────────────┴────────────────┘
"""

# ============================================================
# EXAMPLES: Before/After
# ============================================================

EXAMPLES = """
═══════════════════════════════════════════════════════════════════
EXAMPLE 1: Monetary Amount (Tax, Shipping, Total)
═══════════════════════════════════════════════════════════════════

BEFORE:
<div typeof="schema:MonetaryAmount doco-fd:KeyValuePair" resource="#tax">
    <span property="doco-fd:fieldLabel">Tax (8.5%):</span>
    <span property="schema:value doco-fd:fieldValue">$8.50</span>
</div>

AFTER:
<div typeof="schema:MonetaryAmount" resource="#tax" 
     property="schema:name" content="Tax">
    <meta property="schema:currency" content="USD">
    <span property="schema:value" content="8.50">$8.50</span>
</div>

GRAPH:
  #tax a schema:MonetaryAmount ;
       schema:name "Tax" ;
       schema:currency "USD" ;
       schema:value 8.50 .


═══════════════════════════════════════════════════════════════════
EXAMPLE 2: Connected Fields (Invoice → Line Items)
═══════════════════════════════════════════════════════════════════

BEFORE:
<div resource="#invoice">
    <div resource="#invoice_items_table">
        <tr resource="#item_row_1">
            <td property="doco-fd:cellValue">Consulting</td>
        </tr>
    </div>
</div>

AFTER:
<div typeof="schema:Invoice" resource="#invoice">
    <div rel="schema:hasPart" resource="#items">
        <div typeof="schema:ItemList" resource="#items">
            <div rel="schema:itemListElement" resource="#item1">
                <div typeof="schema:InvoiceItem" resource="#item1">
                    <span property="schema:name">Consulting</span>
                    <span property="schema:quantity" content="1">1</span>
                    <span rel="schema:unitPrice" resource="#price1">
                        <span typeof="schema:MonetaryAmount" resource="#price1">
                            <span property="schema:value">500.00</span>
                        </span>
                    </span>
                </div>
            </div>
        </div>
    </div>
</div>

GRAPH:
  #invoice a schema:Invoice ;
           schema:hasPart #items .
  #items a schema:ItemList ;
         schema:itemListElement #item1 .
  #item1 a schema:InvoiceItem ;
         schema:name "Consulting" ;
         schema:quantity 1 ;
         schema:unitPrice #price1 .
  #price1 a schema:MonetaryAmount ;
          schema:value 500.00 .


═══════════════════════════════════════════════════════════════════
EXAMPLE 3: Simple Key-Value (with default property)
═══════════════════════════════════════════════════════════════════

If you want a simple pattern that works everywhere:

<div typeof="doco-fd:KeyValuePair" resource="#field">
    <span property="doco-fd:key">Label</span>
    <span property="doco-fd:value">Content</span>
</div>

Or define a default in your ontology:
  doco-fd:KeyValuePair 
    rdfs:subClassOf [
      a owl:Restriction ;
      owl:onProperty rdf:value ;
      owl:hasValue ?value
    ]
"""

def print_analysis():
    print("=" * 70)
    print("RDF GRAPH STRUCTURE ANALYSIS")
    print("=" * 70)
    print()
    print(CURRENT_PATTERN)
    print()
    print(PROPERTY_STRATEGY)
    print()
    print(IMPROVED_PATTERN)
    print()
    print(VISUALIZATION_APPROACHES)
    print()
    print(PROPERTY_MAPPINGS)
    print()
    print(EXAMPLES)


if __name__ == "__main__":
    print_analysis()
