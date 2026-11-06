"""
RDF Graph Visualizer for Document Templates

Visualizes the RDFa structure embedded in HTML templates.
Generates interactive HTML visualizations and static images.
"""
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field
import json


@dataclass
class RDFTriple:
    subject: str
    predicate: str
    obj: str
    subject_type: List[str] = field(default_factory=list)


@dataclass
class GraphNode:
    id: str
    types: List[str]
    properties: Dict[str, List[str]]
    label: str = ""


class RDFaExtractor:
    """Extracts RDF triples from RDFa-enhanced HTML templates"""

    # Common namespace prefixes
    NAMESPACES = {
        'schema': 'http://schema.org/',
        'doco': 'http://purl.org/spar/doco/',
        'doco-fd': 'http://purl.org/doco-fd/',
        'foaf': 'http://xmlns.com/foaf/0.1/',
        'dc': 'http://purl.org/dc/terms/',
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    }

    def __init__(self):
        self.triples: List[RDFTriple] = []
        self.nodes: Dict[str, GraphNode] = {}
        self.hierarchy: Dict[str, List[str]] = defaultdict(list)

    def extract_from_html(self, html_content: str) -> Tuple[List[RDFTriple], Dict[str, GraphNode]]:
        """Extract RDF triples from HTML with RDFa markup"""
        self.triples = []
        self.nodes = {}
        self.hierarchy = defaultdict(list)

        self._extract_resources(html_content)
        self._extract_types(html_content)
        self._extract_properties(html_content)
        self._build_hierarchy(html_content)

        return self.triples, self.nodes

    def _extract_resources(self, html: str):
        """Find all resource definitions"""
        pattern = r'resource="([^"]+)"'
        for match in re.finditer(pattern, html):
            resource_id = match.group(1)
            if resource_id.startswith('#'):
                resource_id = resource_id[1:]
            if resource_id not in self.nodes:
                self.nodes[resource_id] = GraphNode(
                    id=resource_id,
                    types=[],
                    properties={},
                    label=self._clean_label(resource_id)
                )

    def _extract_types(self, html: str):
        """Extract typeof declarations"""
        pattern = r'typeof="([^"]+)"[^>]*resource="([^"]+)"'
        for match in re.finditer(pattern, html):
            types_str = match.group(1)
            resource_id = match.group(2)
            if resource_id.startswith('#'):
                resource_id = resource_id[1:]

            types = [t.strip() for t in types_str.split()]
            clean_types = [self._shorten_namespace(t) for t in types]

            if resource_id in self.nodes:
                self.nodes[resource_id].types.extend(clean_types)
            else:
                self.nodes[resource_id] = GraphNode(
                    id=resource_id,
                    types=clean_types,
                    properties={},
                    label=self._clean_label(resource_id)
                )

            for t in clean_types:
                self.triples.append(RDFTriple(
                    subject=resource_id,
                    predicate="rdf:type",
                    obj=t,
                    subject_type=clean_types
                ))

    def _extract_properties(self, html: str):
        """Extract property declarations and their values"""
        prop_pattern = r'property="([^"]+)"[^>]*>([^<]*)<'

        current_resource = None
        for line in html.split('\n'):
            resource_match = re.search(r'resource="([^"]+)"', line)
            if resource_match:
                current_resource = resource_match.group(1)
                if current_resource.startswith('#'):
                    current_resource = current_resource[1:]

            for match in re.finditer(prop_pattern, line):
                props_str = match.group(1)
                value = match.group(2).strip()

                props = [p.strip() for p in props_str.split()]
                clean_props = [self._shorten_namespace(p) for p in props]

                for prop in clean_props:
                    if current_resource and current_resource in self.nodes:
                        if prop not in self.nodes[current_resource].properties:
                            self.nodes[current_resource].properties[prop] = []
                        self.nodes[current_resource].properties[prop].append(value)

                        self.triples.append(RDFTriple(
                            subject=current_resource,
                            predicate=prop,
                            obj=value,
                            subject_type=self.nodes[current_resource].types
                        ))

    def _build_hierarchy(self, html: str):
        """Build parent-child relationships based on HTML nesting"""
        resource_stack = []

        for line in html.split('\n'):
            open_match = re.search(r'<[^>]*resource="([^"]+)"', line)
            if open_match:
                resource_id = open_match.group(1)
                if resource_id.startswith('#'):
                    resource_id = resource_id[1:]
                if resource_stack:
                    parent = resource_stack[-1]
                    self.hierarchy[parent].append(resource_id)
                resource_stack.append(resource_id)

            if '</div>' in line or '</section>' in line or '</header>' in line:
                if resource_stack:
                    resource_stack.pop()

    def _shorten_namespace(self, uri: str) -> str:
        """Convert full URIs to prefixed form"""
        for prefix, ns in self.NAMESPACES.items():
            if uri.startswith(ns):
                return uri.replace(ns, f"{prefix}:")
        return uri

    def _clean_label(self, resource_id: str) -> str:
        """Create human-readable label from resource ID"""
        label = resource_id.replace('_', ' ').replace('#', '')
        label = re.sub(r'_field$', '', label)
        label = re.sub(r'_section$', '', label)
        return label.title()


class GraphVisualizer:
    """Generates visualizations from extracted RDF graph"""

    def __init__(self, triples: List[RDFTriple], nodes: Dict[str, GraphNode], hierarchy: Dict[str, List[str]]):
        self.triples = triples
        self.nodes = nodes
        self.hierarchy = hierarchy

    def to_mermaid(self) -> str:
        """Generate Mermaid.js diagram code"""
        lines = ["graph TD"]
        lines.append("    %% Node Types")

        type_colors = {
            'schema:MonetaryAmount': '#90EE90',
            'schema:Person': '#FFB6C1',
            'schema:Organization': '#87CEEB',
            'schema:PostalAddress': '#DDA0DD',
            'schema:Invoice': '#F0E68C',
            'doco-fd:FormField': '#E6E6FA',
            'doco-fd:FormSection': '#F5DEB3',
            'doco-fd:Header': '#FFD700',
            'doco-fd:Footer': '#D3D3D3',
            'doco-fd:KeyValuePair': '#98FB98',
            'doco:Table': '#FFA07A',
        }

        defined_nodes = set()

        for node_id, node in self.nodes.items():
            if not node.types:
                continue

            safe_id = self._safe_id(node_id)
            main_type = node.types[0] if node.types else "Node"
            color = type_colors.get(main_type, '#FFFFFF')

            label_parts = [f"{node.label}"]
            if node.types:
                label_parts.append(f"[{', '.join(node.types[:2])}]")

            label = "\\n".join(label_parts)
            lines.append(f'    {safe_id}["{label}"]:::typedNode')
            lines.append(f'    style {safe_id} fill:{color},stroke:#333')
            defined_nodes.add(safe_id)

        lines.append("")
        lines.append("    %% Relationships")

        for triple in self.triples:
            if triple.predicate == "rdf:type":
                continue

            subj_id = self._safe_id(triple.subject)
            if subj_id not in defined_nodes:
                continue

            pred_label = self._shorten_predicate(triple.predicate)
            obj_label = triple.obj[:30] + "..." if len(triple.obj) > 30 else triple.obj

            if triple.obj in self.nodes:
                obj_id = self._safe_id(triple.obj)
                if obj_id in defined_nodes:
                    lines.append(f'    {subj_id} -->|{pred_label}| {obj_id}')
            else:
                value_id = f"value_{hash(triple.obj) % 10000}"
                lines.append(f'    {value_id}["{obj_label}"]:::valueNode')
                lines.append(f'    {subj_id} -->|{pred_label}| {value_id}')
                lines.append(f'    style {value_id} fill:#f9f,stroke:#666')

        lines.append("")
        lines.append("    %% Hierarchy (containment)")
        for parent, children in self.hierarchy.items():
            parent_id = self._safe_id(parent)
            if parent_id not in defined_nodes:
                continue
            for child in children:
                child_id = self._safe_id(child)
                if child_id in defined_nodes:
                    lines.append(f'    {parent_id} -.->|contains| {child_id}')

        lines.append("")
        lines.append("    classDef typedNode fill:#fff,stroke:#333,stroke-width:2px")
        lines.append("    classDef valueNode fill:#f9f,stroke:#666,stroke-dasharray: 5 5")

        return "\n".join(lines)

    def to_d3_data(self) -> Dict[str, Any]:
        """Generate data for D3.js force-directed graph"""
        nodes_data = []
        links_data = []

        type_colors = {
            'schema:MonetaryAmount': '#90EE90',
            'schema:Person': '#FFB6C1',
            'schema:Organization': '#87CEEB',
            'schema:PostalAddress': '#DDA0DD',
            'schema:Invoice': '#F0E68C',
            'doco-fd:FormField': '#E6E6FA',
            'doco-fd:FormSection': '#F5DEB3',
            'doco-fd:Header': '#FFD700',
            'doco-fd:Footer': '#D3D3D3',
            'doco-fd:KeyValuePair': '#98FB98',
        }

        node_index = {}
        idx = 0

        for node_id, node in self.nodes.items():
            if not node.types:
                continue

            main_type = node.types[0] if node.types else "Node"
            color = type_colors.get(main_type, '#CCCCCC')

            nodes_data.append({
                "id": node_id,
                "label": node.label,
                "types": node.types,
                "color": color,
                "group": main_type.split(':')[-1] if ':' in main_type else main_type
            })
            node_index[node_id] = idx
            idx += 1

        added_links = set()
        for triple in self.triples:
            if triple.predicate == "rdf:type":
                continue

            if triple.subject in node_index:
                source_idx = node_index[triple.subject]

                if triple.obj in node_index:
                    target_idx = node_index[triple.obj]
                    link_key = (source_idx, target_idx, triple.predicate)
                    if link_key not in added_links:
                        links_data.append({
                            "source": source_idx,
                            "target": target_idx,
                            "label": self._shorten_predicate(triple.predicate),
                            "type": "node-to-node"
                        })
                        added_links.add(link_key)

        for parent, children in self.hierarchy.items():
            if parent in node_index:
                for child in children:
                    if child in node_index:
                        link_key = (node_index[parent], node_index[child], "contains")
                        if link_key not in added_links:
                            links_data.append({
                                "source": node_index[parent],
                                "target": node_index[child],
                                "label": "contains",
                                "type": "hierarchy",
                                "dashed": True
                            })
                            added_links.add(link_key)

        return {"nodes": nodes_data, "links": links_data}

    def to_html_interactive(self, title: str = "RDF Graph") -> str:
        """Generate self-contained interactive HTML visualization"""
        d3_data = self.to_d3_data()

        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
        }}
        #container {{
            display: flex;
            height: 100vh;
        }}
        #sidebar {{
            width: 300px;
            background: #16213e;
            color: #fff;
            padding: 20px;
            overflow-y: auto;
        }}
        #graph {{
            flex: 1;
        }}
        h1 {{
            font-size: 18px;
            margin-bottom: 20px;
            color: #e94560;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
            font-size: 12px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
            margin-right: 8px;
        }}
        .node {{
            cursor: pointer;
        }}
        .node circle {{
            stroke-width: 2px;
        }}
        .node text {{
            font-size: 11px;
            fill: #fff;
            pointer-events: none;
        }}
        .link {{
            fill: none;
            stroke-opacity: 0.6;
        }}
        .link-hierarchy {{
            stroke-dasharray: 5, 5;
        }}
        .link-label {{
            font-size: 9px;
            fill: #aaa;
        }}
        .tooltip {{
            position: absolute;
            background: #0f3460;
            color: #fff;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            pointer-events: none;
            max-width: 250px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        .stats {{
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #333;
        }}
        .stat-item {{
            margin: 5px 0;
            font-size: 12px;
        }}
        .controls {{
            margin-top: 20px;
        }}
        .controls button {{
            background: #e94560;
            color: #fff;
            border: none;
            padding: 8px 16px;
            margin: 5px;
            border-radius: 4px;
            cursor: pointer;
        }}
        .controls button:hover {{
            background: #ff6b6b;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="sidebar">
            <h1>{title}</h1>
            <div id="legend"></div>
            <div class="controls">
                <button onclick="resetZoom()">Reset Zoom</button>
                <button onclick="toggleHierarchy()">Toggle Hierarchy</button>
            </div>
            <div class="stats">
                <h3 style="color:#e94560;font-size:14px;">Statistics</h3>
                <div class="stat-item">Nodes: <span id="node-count"></span></div>
                <div class="stat-item">Links: <span id="link-count"></span></div>
            </div>
        </div>
        <div id="graph"></div>
    </div>
    <div class="tooltip" id="tooltip" style="display:none;"></div>

    <script>
        const data = {json.dumps(d3_data)};

        const width = document.getElementById('graph').clientWidth;
        const height = window.innerHeight;

        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);

        const g = svg.append("g");

        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => g.attr("transform", event.transform));

        svg.call(zoom);

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(40));

        const link = g.append("g")
            .selectAll("line")
            .data(data.links)
            .enter().append("line")
            .attr("class", d => d.dashed ? "link link-hierarchy" : "link")
            .attr("stroke", d => d.dashed ? "#666" : "#e94560")
            .attr("stroke-width", d => d.dashed ? 1 : 2);

        const linkLabels = g.append("g")
            .selectAll("text")
            .data(data.links)
            .enter().append("text")
            .attr("class", "link-label")
            .text(d => d.label);

        const node = g.append("g")
            .selectAll("g")
            .data(data.nodes)
            .enter().append("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("circle")
            .attr("r", 20)
            .attr("fill", d => d.color)
            .attr("stroke", "#fff");

        node.append("text")
            .attr("dy", 30)
            .attr("text-anchor", "middle")
            .text(d => d.label.length > 15 ? d.label.substring(0, 15) + "..." : d.label);

        node.on("mouseover", showTooltip)
            .on("mouseout", hideTooltip);

        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            linkLabels
                .attr("x", d => (d.source.x + d.target.x) / 2)
                .attr("y", d => (d.source.y + d.target.y) / 2);

            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});

        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}

        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}

        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}

        function showTooltip(event, d) {{
            const tooltip = document.getElementById("tooltip");
            tooltip.innerHTML = `
                <strong>${{d.label}}</strong><br>
                <small>Types: ${{d.types.join(', ')}}</small><br>
                <small>ID: ${{d.id}}</small>
            `;
            tooltip.style.display = "block";
            tooltip.style.left = (event.pageX + 10) + "px";
            tooltip.style.top = (event.pageY - 10) + "px";
        }}

        function hideTooltip() {{
            document.getElementById("tooltip").style.display = "none";
        }}

        function resetZoom() {{
            svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
        }}

        let showHierarchy = true;
        function toggleHierarchy() {{
            showHierarchy = !showHierarchy;
            link.attr("display", d => (d.dashed && !showHierarchy) ? "none" : "block");
            linkLabels.attr("display", d => (d.dashed && !showHierarchy) ? "none" : "block");
        }}

        document.getElementById("node-count").textContent = data.nodes.length;
        document.getElementById("link-count").textContent = data.links.length;

        const groups = [...new Set(data.nodes.map(d => d.group))];
        const legend = d3.select("#legend");
        groups.forEach(g => {{
            const color = data.nodes.find(n => n.group === g).color;
            legend.append("div")
                .attr("class", "legend-item")
                .html(`<div class="legend-color" style="background:${{color}}"></div>${{g}}`);
        }});
    </script>
</body>
</html>'''

    def to_ascii(self) -> str:
        """Generate simple ASCII representation of the graph"""
        lines = []
        lines.append("=" * 60)
        lines.append("RDF GRAPH STRUCTURE")
        lines.append("=" * 60)

        for node_id, node in self.nodes.items():
            if not node.types:
                continue

            lines.append(f"\n[{node.label}]")
            lines.append(f"  ID: {node_id}")
            lines.append(f"  Types: {', '.join(node.types)}")

            if node.properties:
                lines.append("  Properties:")
                for prop, values in node.properties.items():
                    for value in values[:3]:
                        display_val = value[:40] + "..." if len(value) > 40 else value
                        lines.append(f"    {prop} -> {display_val}")

            if node_id in self.hierarchy:
                children = self.hierarchy[node_id]
                lines.append(f"  Contains: {len(children)} children")

        lines.append("\n" + "=" * 60)
        lines.append(f"Summary: {len(self.nodes)} nodes")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _safe_id(self, id_str: str) -> str:
        """Convert ID to safe Mermaid identifier"""
        return re.sub(r'[^a-zA-Z0-9_]', '_', id_str)

    def _shorten_predicate(self, predicate: str) -> str:
        """Shorten predicate for display"""
        short_names = {
            'doco-fd:fieldValue': 'value',
            'doco-fd:fieldLabel': 'label',
            'schema:value': 'amount',
            'schema:name': 'name',
            'schema:legalName': 'name',
            'schema:streetAddress': 'address',
            'schema:telephone': 'phone',
            'schema:email': 'email',
            'schema:dateCreated': 'date',
            'schema:dateIssued': 'date',
            'schema:identifier': 'id',
        }
        return short_names.get(predicate, predicate.split(':')[-1])


def visualize_template(template_path: str, output_dir: str = None):
    """Main function to visualize a template's RDF structure"""
    template_path = Path(template_path)
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = template_path.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)

    with open(template_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    extractor = RDFaExtractor()
    triples, nodes = extractor.extract_from_html(html_content)
    hierarchy = extractor.hierarchy

    visualizer = GraphVisualizer(triples, nodes, hierarchy)

    name = template_path.stem

    ascii_output = visualizer.to_ascii()
    ascii_path = output_dir / f"{name}_graph.txt"
    with open(ascii_path, 'w', encoding='utf-8') as f:
        f.write(ascii_output)
    print(f"ASCII graph: {ascii_path}")

    mermaid_output = visualizer.to_mermaid()
    mermaid_path = output_dir / f"{name}_mermaid.md"
    with open(mermaid_path, 'w', encoding='utf-8') as f:
        f.write(f"```mermaid\n{mermaid_output}\n```\n")
    print(f"Mermaid diagram: {mermaid_path}")

    html_output = visualizer.to_html_interactive(title=f"{name.title()} RDF Graph")
    html_path = output_dir / f"{name}_interactive.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_output)
    print(f"Interactive HTML: {html_path}")

    return {
        "triples": len(triples),
        "nodes": len(nodes),
        "ascii_path": str(ascii_path),
        "mermaid_path": str(mermaid_path),
        "html_path": str(html_path)
    }


def visualize_all_templates(templates_dir: str, output_base: str = None, create_index: bool = True):
    """
    Process all templates and optionally create an index page.
    
    Args:
        templates_dir: Root directory containing template folders
        output_base: Base directory for all visualizations (defaults to templates_dir/rdf_graphs)
        create_index: Whether to create an HTML index page
    
    Returns:
        List of results for each template
    """
    from glob import glob
    
    templates_dir = Path(templates_dir)
    output_base = Path(output_base) if output_base else templates_dir / "rdf_graphs"
    output_base.mkdir(exist_ok=True)
    
    template_files = sorted(templates_dir.glob("**/template.html"))
    
    results = []
    categories = defaultdict(list)
    
    for template_path in template_files:
        template_name = template_path.parent.name
        category = template_path.parent.parent.name
        if category == "templates":
            category = "general"
        
        print(f"\nProcessing: {template_name}")
        
        try:
            output_dir = output_base / template_name
            result = visualize_template(str(template_path), str(output_dir))
            result["name"] = template_name
            result["category"] = category
            result["relative_html"] = f"{template_name}/template_interactive.html"
            results.append(result)
            categories[category].append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "name": template_name,
                "category": category,
                "error": str(e)
            })
    
    if create_index and results:
        index_path = output_base / "index.html"
        index_html = _generate_index_page(results, categories, output_base)
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_html)
        print(f"\n\nIndex page: {index_path}")
    
    return results


def _generate_index_page(results: List[Dict], categories: Dict[str, List], output_base: Path) -> str:
    """Generate an HTML index page with all template visualizations"""
    
    total_nodes = sum(r.get('nodes', 0) for r in results if 'error' not in r)
    total_triples = sum(r.get('triples', 0) for r in results if 'error' not in r)
    success_count = sum(1 for r in results if 'error' not in r)
    
    category_sections = []
    for cat, items in sorted(categories.items()):
        rows = []
        for item in items:
            if 'error' in item:
                rows.append(f'''
                <tr class="error">
                    <td>{item['name']}</td>
                    <td colspan="4">Error: {item['error']}</td>
                </tr>''')
            else:
                rows.append(f'''
                <tr>
                    <td><a href="{item['relative_html']}">{item['name']}</a></td>
                    <td>{item['nodes']}</td>
                    <td>{item['triples']}</td>
                    <td><a href="{item['relative_html']}">Interactive</a></td>
                    <td><a href="{item['name']}/template_mermaid.md">Mermaid</a></td>
                </tr>''')
        
        category_sections.append(f'''
        <div class="category">
            <h2 class="category-header" data-category="{cat}">
                {cat.title()} ({len(items)} templates)
            </h2>
            <table class="template-table">
                <thead>
                    <tr>
                        <th>Template</th>
                        <th>Nodes</th>
                        <th>Triples</th>
                        <th>Graph</th>
                        <th>Mermaid</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>''')
    
    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>RDF Graph Visualizations - All Templates</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }}
        .header {{
            background: #16213e;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        h1 {{
            margin: 0 0 10px 0;
            color: #e94560;
        }}
        .stats {{
            display: flex;
            gap: 30px;
            margin-top: 15px;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #e94560;
        }}
        .stat-label {{
            font-size: 0.8em;
            color: #888;
        }}
        .category {{
            margin-bottom: 20px;
        }}
        .category-header {{
            background: #16213e;
            padding: 15px 20px;
            margin: 0;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            user-select: none;
        }}
        .category-header:hover {{
            background: #1f2b4d;
        }}
        .category-header::before {{
            content: "▼ ";
            color: #e94560;
        }}
        .category.collapsed .category-header::before {{
            content: "▶ ";
        }}
        .category.collapsed table {{
            display: none;
        }}
        .template-table {{
            width: 100%;
            border-collapse: collapse;
            background: #16213e;
            border-radius: 0 0 8px 8px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #2a3a5e;
        }}
        th {{
            background: #0f3460;
            color: #e94560;
            font-weight: 500;
        }}
        tr:hover {{
            background: #1f2b4d;
        }}
        a {{
            color: #4fc3f7;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .error {{
            color: #ff6b6b;
        }}
        .search-box {{
            margin-bottom: 20px;
        }}
        .search-box input {{
            width: 100%;
            max-width: 400px;
            padding: 12px 15px;
            border: none;
            border-radius: 5px;
            background: #16213e;
            color: #fff;
            font-size: 14px;
        }}
        .search-box input:focus {{
            outline: 2px solid #e94560;
        }}
        .search-box input::placeholder {{
            color: #666;
        }}
        .footer {{
            margin-top: 40px;
            padding: 20px;
            background: #16213e;
            border-radius: 8px;
            text-align: center;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RDF Graph Visualizations</h1>
        <p>Interactive visualizations of RDFa structure in document templates</p>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{success_count}</div>
                <div class="stat-label">Templates</div>
            </div>
            <div class="stat">
                <div class="stat-value">{total_nodes:,}</div>
                <div class="stat-label">Total Nodes</div>
            </div>
            <div class="stat">
                <div class="stat-value">{total_triples:,}</div>
                <div class="stat-label">Total Triples</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(categories)}</div>
                <div class="stat-label">Categories</div>
            </div>
        </div>
    </div>
    
    <div class="search-box">
        <input type="text" id="search" placeholder="Search templates..." onkeyup="filterTemplates()">
    </div>
    
    {''.join(category_sections)}
    
    <div class="footer">
        <p>Generated by RDF Graph Visualizer</p>
        <p>Open individual template folders for detailed ASCII and Mermaid exports</p>
    </div>
    
    <script>
        document.querySelectorAll('.category-header').forEach(header => {{
            header.addEventListener('click', () => {{
                header.parentElement.classList.toggle('collapsed');
            }});
        }});
        
        function filterTemplates() {{
            const query = document.getElementById('search').value.toLowerCase();
            document.querySelectorAll('.template-table tbody tr').forEach(row => {{
                const name = row.querySelector('td')?.textContent.toLowerCase() || '';
                row.style.display = name.includes(query) ? '' : 'none';
            }});
        }}
    </script>
</body>
</html>'''


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single template: python rdf_graph_visualizer.py <template.html> [output_dir]")
        print("  All templates:   python rdf_graph_visualizer.py --all <templates_dir> [output_dir]")
        print("\nExamples:")
        print("  python rdf_graph_visualizer.py templates/invoice_template/template.html")
        print("  python rdf_graph_visualizer.py --all templates/")
        sys.exit(1)

    if sys.argv[1] == "--all":
        templates_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        results = visualize_all_templates(templates_dir, output_dir)
        success = sum(1 for r in results if 'error' not in r)
        print(f"\nProcessed {len(results)} templates ({success} successful)")
    else:
        template_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        result = visualize_template(template_path, output_dir)
        print(f"\nExtracted {result['nodes']} nodes, {result['triples']} triples")
        print(f"Open {result['html_path']} in a browser for interactive view")
