// Shared utilities — loaded before all other inspector scripts

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Shared tooltip builder for D3 graph nodes (used by force + tree views)
function buildGraphTooltipHtml(node) {
    let html = `<div class="graph-tooltip-label">${escapeHtml(node.label || node.id)}</div>`;
    html += `<div class="graph-tooltip-type">${node.type === 'annotation' ? (node.entity_type || 'TEXT') : node.type}</div>`;
    if (node.resource_id) html += `<div class="graph-tooltip-rdfa">resource: ${escapeHtml(node.resource_id)}</div>`;
    if (node.rdfa_typeof) html += `<div class="graph-tooltip-rdfa">typeof: ${escapeHtml(node.rdfa_typeof)}</div>`;
    if (node.rdfa_property) html += `<div class="graph-tooltip-rdfa">property: ${escapeHtml(node.rdfa_property)}</div>`;
    if (node.box_2d) html += `<div class="graph-tooltip-rdfa">box: [${node.box_2d.join(', ')}]</div>`;
    return html;
}
