// Graph View — D3 force + tree visualizations
// Depends on: utils.js (escapeHtml, buildGraphTooltipHtml)
// Read from inspector.js globals: templatesDataCache, highlightAnnotation, scrollToAnnotationRow, unhighlightAnnotation

let graphViewActive = {};   // { templateName: boolean }
let graphMode = {};         // { templateName: 'force' | 'tree' }
let graphDataCache = {};    // { templateName: { nodes, links } }

const GRAPH_NODE_COLORS = {
    document: '#2563eb',
    section: '#10b981',
    field: '#f59e0b',
    table: '#0891b2',
    header_row: '#64748b',
    row: '#0d9488',
    PERSON: '#ec4899', ORG: '#8b5cf6', DATE: '#06b6d4',
    ADDRESS: '#84cc16', AMOUNT: '#f97316', PHONE: '#14b8a6',
    ID: '#a78bfa', EMAIL: '#f472b6',
};
const GRAPH_NODE_RADIUS = { document: 14, section: 10, field: 8, table: 10, header_row: 7, row: 7, annotation: 5 };

function getNodeColor(node) {
    if (node.type !== 'annotation') return GRAPH_NODE_COLORS[node.type] || '#6b7280';
    return GRAPH_NODE_COLORS[node.entity_type] || '#6b7280';
}
function getNodeRadius(node) { return GRAPH_NODE_RADIUS[node.type] || 5; }

function toggleGraphView(btn, templateName, skipHashUpdate) {
    const section = document.getElementById(templateName);
    if (!section) return;
    const viewer = section.querySelector('.viewer-content');
    const pageSidebar = section.querySelector('.page-sidebar');
    const graphPanel = document.getElementById(`graph-panel-${templateName}`);
    if (!graphPanel) return;

    const isActive = btn.classList.toggle('active');
    graphViewActive[templateName] = isActive;

    if (isActive) {
        viewer.style.display = 'none';
        if (pageSidebar) pageSidebar.style.display = 'none';
        graphPanel.style.display = 'flex';

        if (!graphPanel.dataset.initialized) {
            const tmplData = templatesDataCache.find(t => t.template_name === templateName);
            if (tmplData) {
                const graphData = buildGraphData(tmplData);
                graphDataCache[templateName] = graphData;
                renderForceGraph(templateName, graphData);
                graphPanel.dataset.initialized = 'true';
                graphMode[templateName] = 'force';
                const info = document.getElementById(`graph-info-${templateName}`);
                if (info) info.textContent = `${graphData.nodes.length} nodes, ${graphData.links.length} edges`;
            }
        }
    } else {
        viewer.style.display = '';
        if (pageSidebar) pageSidebar.style.display = '';
        graphPanel.style.display = 'none';
    }

    if (!skipHashUpdate) {
        window.location.hash = isActive ? `${templateName}:graph` : templateName;
    }
}

// Parse hash into { templateName, graph }
function parseHash(hash) {
    const raw = (hash || '').startsWith('#') ? hash.substring(1) : (hash || '');
    const [templateName, view] = raw.split(':');
    return { templateName: templateName || '', graph: view === 'graph' };
}

// Activate graph view for a template (called on hash restore, skipHashUpdate=true)
function activateGraphView(templateName) {
    const section = document.getElementById(templateName);
    if (!section) return;
    const btn = section.querySelector('.graph-toggle');
    if (!btn || btn.classList.contains('active')) return;
    toggleGraphView(btn, templateName, true);
}

function buildGraphData(templateData) {
    const nodes = [];
    const links = [];
    const nodeMap = {};

    // Structural label types — label their parent node, not shown as separate annotation nodes
    const FIELD_LABEL_TYPES = new Set(['FIELD_LABEL', 'SECTION_TITLE', 'TITLE', 'CAPTION']);

    const annotations = templateData.annotations || [];

    // ── Pass 1: Collect label text for fields/sections ──
    const fieldLabels = {};   // field_id → label text
    const sectionLabels = {}; // section_id → label text

    for (const ann of annotations) {
        if (!FIELD_LABEL_TYPES.has(ann.entity_type)) continue;
        const text = (ann.text_content || '').trim();
        if (!text) continue;
        if (ann.field_id) {
            if (!fieldLabels[ann.field_id]) fieldLabels[ann.field_id] = text;
        } else if (ann.section_id) {
            if (!sectionLabels[ann.section_id]) sectionLabels[ann.section_id] = text;
        }
    }

    // ── Pass 2: Discover tables and collect row header labels ──
    const tableAnnotations = {};  // table_id → [ann, ...]
    const rowLabels = {};         // row_id → label text

    for (const ann of annotations) {
        if (!ann.table_id) continue;
        if (!tableAnnotations[ann.table_id]) tableAnnotations[ann.table_id] = [];
        tableAnnotations[ann.table_id].push(ann);

        // Collect row header labels (tfoot: "Subtotal:", "Total Due:", etc.)
        if (ann.entity_type === 'TABLE_HEADER' && ann.row_id) {
            const text = (ann.text_content || '').trim();
            if (text && !rowLabels[ann.row_id]) rowLabels[ann.row_id] = text;
        }
    }

    // Track which annotations are handled by the table builder
    const tableHandledIdx = new Set();

    // ── Build document root ──
    const docId = `doc_${templateData.template_name}`;
    nodes.push({
        id: docId, label: templateData.display_name || templateData.template_name,
        type: 'document', rdfa_typeof: null, box_2d: null, page: null,
    });
    nodeMap[docId] = true;

    // ── Build section nodes ──
    for (const sec of (templateData.sections || [])) {
        const nodeId = `section_${sec.section_id}`;
        if (!nodeMap[nodeId]) {
            const label = sectionLabels[sec.section_id] || sec.section_id.replace(/_/g, ' ');
            nodes.push({
                id: nodeId, label: label,
                resource_id: sec.section_id,
                type: 'section', rdfa_typeof: sec.rdfa_typeof || sec.section_type,
                box_2d: sec.box_2d, page: sec.page,
            });
            nodeMap[nodeId] = true;
        }
        links.push({ source: docId, target: nodeId, relation: 'hasSection' });
    }

    // ── Build explicit field nodes (non-table RDFa FormFields) ──
    for (const fld of (templateData.fields || [])) {
        const nodeId = `field_${fld.field_id}`;
        if (!nodeMap[nodeId]) {
            const label = fieldLabels[fld.field_id] || fld.field_id.replace(/_/g, ' ').replace(' field', '');
            nodes.push({
                id: nodeId, label: label,
                resource_id: fld.field_id,
                type: 'field', rdfa_typeof: fld.rdfa_typeof || fld.field_type,
                box_2d: fld.box_2d, page: fld.page,
            });
            nodeMap[nodeId] = true;
        }
        const parentId = fld.section_id ? `section_${fld.section_id}` : docId;
        links.push({ source: nodeMap[parentId] ? parentId : docId, target: nodeId, relation: 'hasField' });
    }

    // ── Build table structures: Table → Header + Rows → Cells ──
    for (const [tableId, tableAnns] of Object.entries(tableAnnotations)) {
        const tblNodeId = `table_${tableId}`;
        if (!nodeMap[tblNodeId]) {
            const sampleAnn = tableAnns.find(a => a.section_id) || tableAnns[0];
            const parentSectionId = sampleAnn ? sampleAnn.section_id : null;
            const parentId = parentSectionId && nodeMap[`section_${parentSectionId}`]
                ? `section_${parentSectionId}` : docId;

            nodes.push({
                id: tblNodeId, label: tableId.replace(/_/g, ' '),
                resource_id: tableId,
                type: 'table', rdfa_typeof: 'doco-fd:Table',
                box_2d: null, page: sampleAnn?.page || null,
            });
            nodeMap[tblNodeId] = true;
            links.push({ source: parentId, target: tblNodeId, relation: 'hasTable' });
        }

        // Separate column headers from data/summary cells
        const columnHeaders = tableAnns.filter(a =>
            a.entity_type === 'TABLE_HEADER' &&
            (a.rdfa_property || '').includes('columnHeader')
        );
        const nonHeaderAnns = tableAnns.filter(a =>
            !(a.entity_type === 'TABLE_HEADER' && (a.rdfa_property || '').includes('columnHeader'))
        );

        // ── Header row node with column header children ──
        if (columnHeaders.length > 0) {
            const headerNodeId = `${tblNodeId}_header`;
            nodes.push({
                id: headerNodeId, label: 'Header',
                type: 'header_row', rdfa_typeof: 'doco-fd:TableRow',
                box_2d: null, page: columnHeaders[0]?.page || null,
            });
            nodeMap[headerNodeId] = true;
            links.push({ source: tblNodeId, target: headerNodeId, relation: 'hasHeader' });

            for (const hdr of columnHeaders) {
                tableHandledIdx.add(hdr._idx);
                const hNodeId = `ann_${hdr._idx}`;
                nodes.push({
                    id: hNodeId, label: hdr.text_content || '?', type: 'annotation',
                    resource_id: hdr.label, entity_type: hdr.entity_type,
                    text_content: hdr.text_content,
                    rdfa_typeof: hdr.rdfa_typeof, rdfa_property: hdr.rdfa_property,
                    box_2d: hdr.box_2d, page: hdr.page, _idx: hdr._idx,
                });
                nodeMap[hNodeId] = true;
                links.push({ source: headerNodeId, target: hNodeId, relation: 'column' });
            }
        }

        // ── Group remaining annotations by row_id ──
        const rowGroups = {};
        const noRowAnns = [];

        for (const ann of nonHeaderAnns) {
            if (ann.row_id) {
                if (!rowGroups[ann.row_id]) rowGroups[ann.row_id] = [];
                rowGroups[ann.row_id].push(ann);
            } else {
                noRowAnns.push(ann);
            }
        }

        // ── Create row nodes and attach cell annotations ──
        const sortedRowIds = Object.keys(rowGroups).sort((a, b) => {
            const aY = rowGroups[a][0]?.box_2d?.[1] ?? 0;
            const bY = rowGroups[b][0]?.box_2d?.[1] ?? 0;
            return aY - bY;
        });

        for (let ri = 0; ri < sortedRowIds.length; ri++) {
            const rowId = sortedRowIds[ri];
            const rowAnns = rowGroups[rowId];
            const rowNodeId = `row_${rowId}`;

            const rowLabel = rowLabels[rowId] || `Row ${ri + 1}`;

            nodes.push({
                id: rowNodeId, label: rowLabel,
                resource_id: rowId,
                type: 'row', rdfa_typeof: 'doco-fd:TableRow',
                box_2d: null, page: rowAnns[0]?.page || null,
            });
            nodeMap[rowNodeId] = true;
            links.push({ source: tblNodeId, target: rowNodeId, relation: 'hasRow' });

            for (const ann of rowAnns) {
                tableHandledIdx.add(ann._idx);
                if (FIELD_LABEL_TYPES.has(ann.entity_type)) continue;
                if (ann.entity_type === 'TABLE_HEADER') continue;

                const cellNodeId = `ann_${ann._idx}`;
                nodes.push({
                    id: cellNodeId, label: ann.text_content || ann.label, type: 'annotation',
                    resource_id: ann.label, entity_type: ann.entity_type,
                    text_content: ann.text_content,
                    rdfa_typeof: ann.rdfa_typeof, rdfa_property: ann.rdfa_property,
                    box_2d: ann.box_2d, page: ann.page, _idx: ann._idx,
                });
                nodeMap[cellNodeId] = true;

                const edgeLabel = ann.column_header || ann.rdfa_property || 'cell';
                links.push({ source: rowNodeId, target: cellNodeId, relation: edgeLabel });
            }
        }

        // Handle orphan table annotations without row_id
        for (const ann of noRowAnns) {
            tableHandledIdx.add(ann._idx);
            if (FIELD_LABEL_TYPES.has(ann.entity_type) || ann.entity_type === 'TABLE_HEADER') continue;

            const cellNodeId = `ann_${ann._idx}`;
            nodes.push({
                id: cellNodeId, label: ann.text_content || ann.label, type: 'annotation',
                resource_id: ann.label, entity_type: ann.entity_type,
                text_content: ann.text_content,
                rdfa_typeof: ann.rdfa_typeof, rdfa_property: ann.rdfa_property,
                box_2d: ann.box_2d, page: ann.page, _idx: ann._idx,
            });
            nodeMap[cellNodeId] = true;
            links.push({ source: tblNodeId, target: cellNodeId, relation: 'containsElement' });
        }
    }

    // ── Build non-table annotation nodes ──
    for (const ann of annotations) {
        if (tableHandledIdx.has(ann._idx)) continue;
        if (FIELD_LABEL_TYPES.has(ann.entity_type)) continue;
        if (ann.entity_type === 'TABLE_HEADER') continue;

        const nodeId = `ann_${ann._idx}`;
        if (nodeMap[nodeId]) continue;
        nodes.push({
            id: nodeId, label: ann.text_content || ann.label, type: 'annotation',
            resource_id: ann.label,
            entity_type: ann.entity_type, text_content: ann.text_content,
            rdfa_typeof: ann.rdfa_typeof, rdfa_property: ann.rdfa_property,
            box_2d: ann.box_2d, page: ann.page, _idx: ann._idx,
        });
        nodeMap[nodeId] = true;

        let parentId = docId;
        if (ann.field_id) {
            const fid = `field_${ann.field_id}`;
            if (nodeMap[fid]) parentId = fid;
        } else if (ann.section_id) {
            const sid = `section_${ann.section_id}`;
            if (nodeMap[sid]) parentId = sid;
        }
        links.push({ source: parentId, target: nodeId, relation: 'containsElement' });
    }

    return { nodes, links };
}

function renderForceGraph(templateName, graphData) {
    const container = document.getElementById(`graph-container-${templateName}`);
    if (!container) return;
    container.innerHTML = '';

    const rect = container.getBoundingClientRect();
    const width = rect.width || 800;
    const height = rect.height || 600;

    const svg = d3.select(container).append('svg')
        .attr('width', width).attr('height', height);

    const g = svg.append('g');

    svg.call(d3.zoom().scaleExtent([0.1, 4]).on('zoom', (event) => {
        g.attr('transform', event.transform);
    }));

    const simulation = d3.forceSimulation(graphData.nodes)
        .force('link', d3.forceLink(graphData.links).id(d => d.id).distance(d => {
            if (d.relation === 'containsElement') return 50;
            if (d.relation === 'hasField' || d.relation === 'hasRow') return 70;
            if (d.relation === 'hasTable' || d.relation === 'hasHeader') return 80;
            if (d.relation === 'column') return 40;
            return 100;
        }))
        .force('charge', d3.forceManyBody().strength(-150))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => getNodeRadius(d) + 4));

    const linkG = g.append('g');
    const link = linkG.selectAll('line')
        .data(graphData.links).enter().append('line')
        .attr('class', 'graph-link')
        .attr('stroke-width', 1);

    // Edge labels for table row→cell links (show column header name)
    const edgeLabels = linkG.selectAll('.graph-edge-label')
        .data(graphData.links.filter(d =>
            !['containsElement','hasSection','hasField','hasTable','hasHeader','hasRow','column'].includes(d.relation)
        ))
        .enter().append('text')
        .attr('class', 'graph-edge-label')
        .text(d => {
            const r = d.relation || '';
            return r.length > 15 ? r.substring(0, 12) + '...' : r;
        });

    const node = g.append('g').selectAll('g')
        .data(graphData.nodes).enter().append('g')
        .attr('class', d => `graph-node graph-node-${d.type}`)
        .call(d3.drag()
            .on('start', (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
            .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
            .on('end', (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
        );

    node.append('circle')
        .attr('r', d => getNodeRadius(d))
        .attr('fill', d => getNodeColor(d))
        .attr('stroke', d => d3.color(getNodeColor(d)).darker(0.5))
        .attr('stroke-width', 1.5);

    node.append('text')
        .attr('dx', d => getNodeRadius(d) + 4)
        .attr('dy', '0.35em')
        .text(d => {
            const lbl = d.label || '';
            return lbl.length > 25 ? lbl.substring(0, 22) + '...' : lbl;
        });

    // Tooltip (uses shared buildGraphTooltipHtml from utils.js)
    const tooltip = d3.select(container).append('div').attr('class', 'graph-tooltip');

    node.on('mouseenter', (event, d) => {
        tooltip.html(buildGraphTooltipHtml(d)).classed('visible', true);
    })
    .on('mousemove', (event) => {
        tooltip.style('left', (event.clientX + 12) + 'px').style('top', (event.clientY - 10) + 'px');
    })
    .on('mouseleave', () => { tooltip.classed('visible', false); });

    node.on('click', (event, d) => { onGraphNodeClick(templateName, d); });

    simulation.on('tick', () => {
        link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        edgeLabels
            .attr('x', d => (d.source.x + d.target.x) / 2)
            .attr('y', d => (d.source.y + d.target.y) / 2);
        node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    container._simulation = simulation;
}

function renderTreeView(templateName, graphData) {
    const container = document.getElementById(`graph-container-${templateName}`);
    if (!container) return;
    container.innerHTML = '';
    if (container._simulation) { container._simulation.stop(); container._simulation = null; }

    const rect = container.getBoundingClientRect();
    const width = rect.width || 800;
    const height = rect.height || 600;

    // Build parent map and relation map from links
    const parentMap = {};
    const relationMap = {};
    const STRUCTURAL_RELATIONS = new Set(['containsElement','hasSection','hasField','hasTable','hasHeader','hasRow','column']);
    for (const link of graphData.links) {
        const tid = typeof link.target === 'object' ? link.target.id : link.target;
        const sid = typeof link.source === 'object' ? link.source.id : link.source;
        if (!parentMap[tid]) parentMap[tid] = sid;
        if (link.relation && !STRUCTURAL_RELATIONS.has(link.relation)) {
            relationMap[tid] = link.relation;
        }
    }

    // Stratify into hierarchy
    let root;
    try {
        root = d3.stratify()
            .id(d => d.id)
            .parentId(d => parentMap[d.id] || null)
            (graphData.nodes);
    } catch (e) {
        container.innerHTML = `<div style="color:#f87171;padding:20px">Failed to build tree: ${e.message}</div>`;
        return;
    }

    const nodeCount = root.descendants().length;
    const treeHeight = Math.max(height, nodeCount * 18);
    const treeWidth = Math.max(width - 160, 400);

    const treeLayout = d3.tree().size([treeHeight, treeWidth]);
    treeLayout(root);

    const svg = d3.select(container).append('svg')
        .attr('width', width).attr('height', height);

    const g = svg.append('g').attr('transform', 'translate(80, 0)');

    svg.call(d3.zoom().scaleExtent([0.1, 4]).on('zoom', (event) => {
        g.attr('transform', event.transform);
    }));
    svg.call(d3.zoom().transform, d3.zoomIdentity.translate(80, -treeHeight / 2 + height / 2));

    // Links
    g.selectAll('.tree-link')
        .data(root.links()).enter().append('path')
        .attr('class', 'tree-link')
        .attr('d', d3.linkHorizontal().x(d => d.y).y(d => d.x));

    // Edge labels on tree links (column header names for row→cell)
    g.selectAll('.tree-edge-label')
        .data(root.links().filter(d => relationMap[d.target.data.id]))
        .enter().append('text')
        .attr('class', 'tree-edge-label')
        .attr('x', d => (d.source.y + d.target.y) / 2)
        .attr('y', d => (d.source.x + d.target.x) / 2 - 4)
        .text(d => {
            const r = relationMap[d.target.data.id] || '';
            return r.length > 15 ? r.substring(0, 12) + '...' : r;
        });

    // Nodes
    const node = g.selectAll('.tree-node')
        .data(root.descendants()).enter().append('g')
        .attr('class', 'tree-node')
        .attr('transform', d => `translate(${d.y},${d.x})`);

    node.append('circle')
        .attr('r', d => getNodeRadius(d.data))
        .attr('fill', d => getNodeColor(d.data))
        .attr('stroke', d => d3.color(getNodeColor(d.data)).darker(0.5))
        .attr('stroke-width', 1.5);

    node.append('text')
        .attr('dx', d => d.children ? -(getNodeRadius(d.data) + 4) : (getNodeRadius(d.data) + 4))
        .attr('dy', '0.35em')
        .attr('text-anchor', d => d.children ? 'end' : 'start')
        .text(d => {
            const lbl = d.data.label || '';
            return lbl.length > 30 ? lbl.substring(0, 27) + '...' : lbl;
        });

    // Tooltip (uses shared buildGraphTooltipHtml from utils.js)
    const tooltip = d3.select(container).append('div').attr('class', 'graph-tooltip');

    node.on('mouseenter', (event, d) => {
        tooltip.html(buildGraphTooltipHtml(d.data)).classed('visible', true);
    })
    .on('mousemove', (event) => {
        tooltip.style('left', (event.clientX + 12) + 'px').style('top', (event.clientY - 10) + 'px');
    })
    .on('mouseleave', () => { tooltip.classed('visible', false); });

    node.on('click', (event, d) => { onGraphNodeClick(templateName, d.data); });
}

function switchGraphMode(templateName, mode, btn) {
    graphMode[templateName] = mode;
    const toolbar = btn.closest('.graph-toolbar');
    toolbar.querySelectorAll('.graph-mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const container = document.getElementById(`graph-container-${templateName}`);
    if (container && container._simulation) { container._simulation.stop(); container._simulation = null; }

    // Rebuild fresh copy (D3 mutates nodes with x/y)
    const freshData = buildGraphData(templatesDataCache.find(t => t.template_name === templateName));
    graphDataCache[templateName] = freshData;

    if (mode === 'force') {
        renderForceGraph(templateName, freshData);
    } else {
        renderTreeView(templateName, freshData);
    }
}

function onGraphNodeClick(templateName, node) {
    if (node._idx === undefined || !node.box_2d || !node.page) return;

    // Switch back to document view
    const section = document.getElementById(templateName);
    if (!section) return;
    const btn = section.querySelector('.graph-toggle');
    if (btn && graphViewActive[templateName]) toggleGraphView(btn, templateName);

    // Highlight the bbox
    setTimeout(() => {
        highlightAnnotation(node._idx, node.page);
        scrollToAnnotationRow(node._idx, node.page);
        setTimeout(() => unhighlightAnnotation(node._idx, node.page), 2000);
    }, 100);
}
