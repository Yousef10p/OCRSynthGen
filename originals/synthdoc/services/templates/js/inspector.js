// Dev Server Inspector JS - Simplified for dynamic mode

let templatesDataCache = null;
let currentlyLoadedTemplate = null;
let currentRunId = null;  // tracks active run for image URLs and regeneration
let reviewsCache = {};    // { templateName: { reviewed: true, timestamp: "..." } }
let notesCache = [];      // [ { id, template_name, page, annotation_idx, label, ... } ]
const ROWS_PER_PAGE = 50;
const tablePageState = {};  // { templateName: { annotations: [], currentPage: 0 } }
let annotationsTableCollapsed = localStorage.getItem('inspectorAnnotationsCollapsed') === 'true';
const activePageByTemplate = {};  // { templateName: pageNum } — tracks visible page per template
// ============== Run Management ==============

function showGenerationUI() {
    const toolbar = document.getElementById('generation-toolbar');
    if (toolbar) {
        toolbar.style.display = 'flex';
    }
    loadRuns();
}

async function loadRuns() {
    try {
        const response = await fetch('/api/inspector/runs');
        const data = await response.json();
        
        const selector = document.getElementById('run-selector');
        if (selector && data.runs) {
            selector.innerHTML = '<option value="">Select run...</option>';
            data.runs.forEach(run => {
                selector.innerHTML += `<option value="${run.id}">${run.id}</option>`;
            });
            
            if (data.runs.length > 0) {
                selector.value = data.runs[0].id;
                loadInspectorData(data.runs[0].id);
            } else {
                showEmptyState();
            }
        }
    } catch (e) {
        console.error('Failed to load runs:', e);
        showEmptyState(`Failed to connect to server: ${e.message}`);
    }
}

function updateRunStats(templates) {
    const el = document.getElementById('run-stats');
    if (!el) return;

    if (!templates || templates.length === 0) {
        el.innerHTML = '';
        return;
    }

    const totalTemplates = templates.length;
    const totalPages = templates.reduce((s, t) => s + (t.total_pages || 0), 0);
    const totalAnnotations = templates.reduce((s, t) => s + (t.annotation_count || 0), 0);
    const categories = new Set(templates.map(t => t.category).filter(Boolean));
    const reviewedCount = templates.filter(t => reviewsCache[t.template_name]?.reviewed).length;

    const stats = [
        { value: totalTemplates, label: 'templates' },
        { value: totalPages, label: 'pages' },
        { value: totalAnnotations.toLocaleString(), label: 'annotations' },
    ];
    if (categories.size > 1) {
        stats.push({ value: categories.size, label: 'categories' });
    }
    stats.push({
        value: `${reviewedCount}/${totalTemplates}`,
        label: 'reviewed',
        cls: reviewedCount === totalTemplates && totalTemplates > 0 ? 'stat-complete' : ''
    });

    el.innerHTML = stats.map(s =>
        `<span class="run-stat ${s.cls || ''}"><span class="stat-value">${s.value}</span> <span class="stat-label">${s.label}</span></span>`
    ).join('');
}

async function loadInspectorData(runId) {
    showLoadingState('Loading run data...');
    try {
        const response = await fetch(`/api/inspector/run/${runId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        templatesDataCache = data.templates || data;
        currentRunId = data.run_id || runId;

        // Load review statuses and notes for this run
        try {
            const revResp = await fetch(`/api/inspector/run/${runId}/reviews`);
            reviewsCache = revResp.ok ? await revResp.json() : {};
        } catch (_) { reviewsCache = {}; }
        try {
            const notesResp = await fetch(`/api/inspector/run/${runId}/notes`);
            notesCache = notesResp.ok ? await notesResp.json() : [];
        } catch (_) { notesCache = []; }

        updateRunStats(templatesDataCache);

        if (templatesDataCache && templatesDataCache.length > 0) {
            document.getElementById('empty-state').style.display = 'none';
            document.getElementById('template-container').style.display = '';

            buildSidebarFromTemplates(templatesDataCache);
            renderTemplate(templatesDataCache[0].template_name);
        } else {
            showEmptyState();
        }
    } catch (err) {
        console.error('Failed to load inspector data:', err);
        showEmptyState(`Failed to load run: ${err.message}`);
    }
}

function switchRun() {
    const selector = document.getElementById('run-selector');
    const runId = selector.value;
    if (runId) {
        loadInspectorData(runId);
    }
}

// ============== Generation Modal ==============

async function openGenerateModal() {
    const modal = document.getElementById('generateModal');
    const listEl = document.getElementById('template-list');
    
    try {
        let templates = [];
        
        const templatesResponse = await fetch('/api/inspector/templates');
        if (templatesResponse.ok) {
            const templatesData = await templatesResponse.json();
            templates = templatesData.templates || [];
        }
        
        const categories = {};
        templates.forEach(t => {
            const cat = t.category || 'General';
            if (!categories[cat]) categories[cat] = [];
            categories[cat].push(t);
        });
        
        let html = `
            <div class="select-all-row">
                <label>
                    <input type="checkbox" id="selectAll" onchange="toggleAllSelect(this.checked)">
                    <strong>Select All</strong>
                </label>
            </div>
        `;
        
        Object.keys(categories).sort().forEach(cat => {
            const catId = cat.replace(/[^a-zA-Z0-9]/g, '_');
            html += `
                <div class="category-group">
                    <div class="category-header" onclick="toggleCategoryCollapse('${catId}')" style="cursor: pointer;">
                        <span class="collapse-icon" id="icon-${catId}">▶</span>
                        <input type="checkbox" id="cat-${catId}" onchange="toggleCategorySelect('${catId}')" onclick="event.stopPropagation()">
                        <label for="cat-${catId}" onclick="event.stopPropagation()">${cat}</label>
                        <span class="count">${categories[cat].length}</span>
                    </div>
                    <div class="category-items" id="items-${catId}" style="display: none;">
            `;
            categories[cat].forEach(t => {
                html += `
                    <div class="template-item">
                        <input type="checkbox" class="template-checkbox cat-${catId}" value="${t.template_name}" data-category="${catId}" id="tmpl-${t.template_name}">
                        <label for="tmpl-${t.template_name}">${t.display_name || t.template_name}</label>
                    </div>
                `;
            });
            html += '</div></div>';
        });
        
        listEl.innerHTML = html;
        modal.style.display = 'flex';  // triggers .generate-modal[style*="flex"]
    } catch (err) {
        console.error('Failed to load templates:', err);
    }
}

function toggleCategorySelect(catId) {
    const catCheckbox = document.getElementById(`cat-${catId}`);
    const isChecked = catCheckbox.checked;
    
    document.querySelectorAll(`.cat-${catId}`).forEach(cb => {
        cb.checked = isChecked;
    });
}

function toggleAllSelect(checked) {
    document.querySelectorAll('.template-checkbox').forEach(cb => {
        cb.checked = checked;
    });
    document.querySelectorAll('[id^="cat-"]').forEach(cb => {
        cb.checked = checked;
    });
}

function toggleCategoryCollapse(catId) {
    const items = document.getElementById(`items-${catId}`);
    const icon = document.getElementById(`icon-${catId}`);
    
    if (items.style.display === 'none') {
        items.style.display = 'block';
        icon.textContent = '▼';
    } else {
        items.style.display = 'none';
        icon.textContent = '▶';
    }
}

function closeGenerateModal(event) {
    if (event && event.target !== event.currentTarget) return;
    document.getElementById('generateModal').style.display = '';
}

async function generateSelectedTemplates() {
    const checkboxes = document.querySelectorAll('.template-checkbox:checked');
    const templates = Array.from(checkboxes).map(cb => cb.value);
    
    if (templates.length === 0) {
        alert('Please select at least one template');
        return;
    }
    
    closeGenerateModal();
    
    const statusEl = document.getElementById('gen-status');
    if (statusEl) statusEl.textContent = `Generating ${templates.length} templates (50 workers)...`;
    
    try {
        const response = await fetch('/api/inspector/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ templates: templates, count: 1, workers: 50 })
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const result = await response.json();
        
        if (statusEl) statusEl.textContent = `Generated ${result.template_count} documents`;
        
        await loadRuns();
    } catch (err) {
        console.error('Generation error:', err);
        if (statusEl) statusEl.textContent = `Error: ${err.message}`;
    }
}

// ============== Refresh Single Template ==============

async function refreshTemplate(templateName) {
    if (document.body.getAttribute('data-mode') === 'static') return;
    const btn = event && event.currentTarget;
    if (btn) {
        btn.disabled = true;
        btn.querySelector('.refresh-icon').classList.add('spinning');
    }

    // Dim the current template content while regenerating
    const section = document.getElementById(templateName);
    if (section) section.classList.add('refreshing');

    try {
        if (!currentRunId) throw new Error('No active run');

        const response = await fetch('/api/inspector/regenerate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ run_id: currentRunId, template_name: templateName })
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.error || `HTTP ${response.status}`);
        }

        const result = await response.json();

        // Swap the template data in-place in the cache
        if (result.template_data && templatesDataCache) {
            // Cache-bust image URLs so browser fetches the new images
            const cacheBuster = `_cb=${Date.now()}`;
            const td = result.template_data;
            if (td.page_buttons_data) {
                td.page_buttons_data.forEach(p => {
                    p.thumb_src = p.thumb_src + (p.thumb_src.includes('?') ? '&' : '?') + cacheBuster;
                });
            }
            if (td.page_containers_data) {
                td.page_containers_data.forEach(p => {
                    p.img_src = p.img_src + (p.img_src.includes('?') ? '&' : '?') + cacheBuster;
                });
            }

            const idx = templatesDataCache.findIndex(t => t.template_name === templateName);
            if (idx >= 0) {
                templatesDataCache[idx] = td;
            } else {
                templatesDataCache.push(td);
            }
        }

        // Update header stats (annotation/page counts may have changed)
        updateRunStats(templatesDataCache);

        // Re-render the same template with new data
        renderTemplate(templateName);
    } catch (err) {
        console.error('Refresh error:', err);
        alert(`Refresh failed: ${err.message}`);
        // Remove dim on error (on success, renderTemplate replaces the DOM)
        if (section) section.classList.remove('refreshing');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.querySelector('.refresh-icon').classList.remove('spinning');
        }
    }
}

// ============== UI Helpers ==============

function showEmptyState(errorMsg) {
    const container = document.getElementById('template-container');
    const emptyState = document.getElementById('empty-state');

    updateRunStats(null);
    if (container) container.style.display = 'none';
    if (emptyState) {
        if (errorMsg) {
            emptyState.innerHTML = `<h2>Error</h2><p style="color:#ef4444">${errorMsg}</p><button class="gen-btn" onclick="loadRuns()">Retry</button>`;
        } else {
            emptyState.innerHTML = '<h2>No Documents Generated</h2><p>Click "Generate" to create documents.</p><button class="gen-btn" onclick="openGenerateModal()">Generate</button>';
        }
        emptyState.style.display = 'flex';
    }
}

function showLoadingState(msg) {
    const container = document.getElementById('template-container');
    const emptyState = document.getElementById('empty-state');

    if (container) container.style.display = 'none';
    if (emptyState) {
        emptyState.innerHTML = `<p style="color:#60a5fa">${msg || 'Loading...'}</p>`;
        emptyState.style.display = 'flex';
    }
}

function buildSidebarFromTemplates(templates) {
    const sidebarNav = document.querySelector('.sidebar-nav');
    if (!sidebarNav) return;
    
    const categories = {};
    templates.forEach(t => {
        const cat = t.category || 'General';
        if (!categories[cat]) categories[cat] = [];
        categories[cat].push(t);
    });
    
    let html = '<h2>Templates</h2><ul>';
    Object.keys(categories).sort().forEach(cat => {
        const catId = cat.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase();
        html += `
            <div class="nav-category">
                <div class="nav-category-header" onclick="toggleCategory('${catId}')">
                    <span class="category-icon">▼</span>
                    <span class="category-title">${cat}</span>
                    <span class="category-count">(${categories[cat].length})</span>
                </div>
                <ul class="nav-category-items" id="cat-${catId}">
        `;
        categories[cat].forEach(t => {
            const rev = reviewsCache[t.template_name] && reviewsCache[t.template_name].reviewed;
            html += `<li><a href="#${t.template_name}" data-template="${t.template_name}" class="${rev ? 'reviewed' : ''}">${t.display_name || t.template_name}</a></li>`;
        });
        html += '</ul></div>';
    });
    html += '</ul>';
    
    sidebarNav.innerHTML = html;
    
    // Re-attach event listeners after rebuild
    document.querySelectorAll('.sidebar-nav a[data-template]').forEach(link => {
        link.addEventListener('click', async function(e) {
            e.preventDefault();
            const templateName = this.getAttribute('data-template');
            await renderTemplate(templateName);
            
            document.querySelectorAll('.sidebar-nav a').forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

function toggleCategory(categoryId) {
    const categoryItems = document.getElementById(`cat-${categoryId}`);
    const header = categoryItems?.previousElementSibling;
    
    if (!categoryItems || !header) return;
    
    const icon = header.querySelector('.category-icon');
    const isCollapsed = categoryItems.classList.contains('collapsed');
    
    if (isCollapsed) {
        categoryItems.classList.remove('collapsed');
        icon.textContent = '▼';
    } else {
        categoryItems.classList.add('collapsed');
        icon.textContent = '▶';
    }
}

// ============== Template Rendering ==============

async function renderTemplate(templateName) {
    const templatesData = templatesDataCache;
    const templateData = templatesData.find(t => t.template_name === templateName);
    
    if (!templateData) {
        console.error(`Template ${templateName} not found`);
        return;
    }
    
    const container = document.getElementById('template-container');
    if (!container) {
        console.error('Template container not found');
        return;
    }
    
    container.innerHTML = buildTemplateHTML(templateData);

    // Render first page of annotations table
    if (tablePageState[templateName]) {
        renderTablePage(templateName, 0);
    }

    // Populate notes panel now that DOM exists
    updateNotesPanel(templateName);

    // Track which page is visible for page-level notes
    setupPageScrollTracking(templateName);

    currentlyLoadedTemplate = templateName;
    showTemplate(templateName);

    // Update active sidebar link
    document.querySelectorAll('.sidebar-nav a').forEach(l => l.classList.remove('active'));
    const activeLink = document.querySelector(`.sidebar-nav a[data-template="${templateName}"]`);
    if (activeLink) activeLink.classList.add('active');
}

function showTemplate(templateName) {
    const allSections = document.querySelectorAll('.template-section');
    allSections.forEach(section => section.style.display = 'none');
    
    const targetSection = document.getElementById(templateName);
    if (targetSection) targetSection.style.display = 'flex';
    
    const navLinks = document.querySelectorAll('.sidebar-nav a');
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + templateName) {
            link.classList.add('active');
        }
    });
}

function buildTemplateHTML(data) {
    const {template_name, display_name, doc_id, total_pages, annotation_count,
           orig_width, orig_height, annotations, page_buttons_data,
           page_containers_data, unique_pages, unique_labels, unique_types} = data;
    
    const pageThumbs = page_buttons_data ? page_buttons_data.map((page, idx) => `
        <div class="page-thumb${idx === 0 ? ' active' : ''}" data-page="${page.page_num}" onclick="scrollToPage('${template_name}', ${page.page_num})">
            <img src="${page.thumb_src}" alt="Page ${page.page_num}" loading="lazy">
            <div class="page-thumb-label">Page ${page.page_num}</div>
        </div>
    `).join('') : '';
    
    const pageContainers = page_containers_data ? page_containers_data.map((page, idx) => {
        const sortedAnns = [...page.annotations].sort((a, b) => {
            const aA = (a.box_2d[2]-a.box_2d[0])*(a.box_2d[3]-a.box_2d[1]);
            const aB = (b.box_2d[2]-b.box_2d[0])*(b.box_2d[3]-b.box_2d[1]);
            return aB - aA;
        });
        
        const svgRects = sortedAnns.map((entity, entityIdx) => {
            const bbox = entity.box_2d || [];
            if (bbox.length !== 4) return '';
            
            const x = (bbox[0] / 1000.0) * orig_width;
            const y = (bbox[1] / 1000.0) * orig_height;
            const x2 = (bbox[2] / 1000.0) * orig_width;
            const y2 = (bbox[3] / 1000.0) * orig_height;
            const w = x2 - x;
            const h = y2 - y;
            
            const globalIdx = entity._idx !== undefined ? entity._idx : annotations.indexOf(entity);
            const label = (entity.label || '').replace(/'/g, "\\'").replace(/"/g, '\\"');
            const entityType = (entity.entity_type || '').replace(/'/g, "\\'").replace(/"/g, '\\"');
            const text = ((entity.text_content || '').substring(0, 100)).replace(/'/g, "\\'").replace(/"/g, '\\"');
            const bboxStr = `[${bbox.join(', ')}]`;
            
            const hasNote = notesCache.some(n => n.template_name === template_name && n.annotation_idx === globalIdx && n.page === page.page_num);

            return `<rect class="bbox${hasNote ? ' has-note' : ''}" x="${x}" y="${y}" width="${w}" height="${h}"
                data-idx="${globalIdx}" data-label="${entity.label}" data-type="${entity.entity_type}" data-page="${page.page_num}"
                data-text="${text}" data-bbox="${bboxStr}"
                onmouseenter="showTooltip(event, '${label}', '${entityType}', '${text}', ${bboxStr})"
                onmousemove="updateTooltipPosition(event)"
                onmouseleave="hideTooltip()"
                onclick="scrollToAnnotationRow(${globalIdx}, ${page.page_num})"
                oncontextmenu="openNoteMenu(event, '${template_name}', ${page.page_num}, ${globalIdx}, '${label}', '${entityType}', '${text}', ${bboxStr})">
            </rect>`;
        }).join('');
        
        const loading = idx === 0 ? 'loading="eager"' : 'loading="lazy"';
        
        return `
            <div class="page-container" data-page="${page.page_num}">
                <div class="page-number-label">Page ${page.page_num} of ${total_pages}</div>
                <div class="image-canvas">
                    <div style="position: relative; width: ${orig_width}px; height: ${orig_height}px;">
                        <img src="${page.img_src}" alt="Page ${page.page_num}" ${loading} style="width: 100%; height: 100%; display: block; position: relative; z-index: 1;">
                        <svg class="overlay-layer overlay-${template_name}" viewBox="0 0 ${orig_width} ${orig_height}" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 10; pointer-events: auto;">
                            ${svgRects}
                        </svg>
                    </div>
                </div>
            </div>`;
    }).join('') : '';
    
    const sortedTableAnns = annotations ? [...annotations].sort((a, b) => {
        const pageA = a.page || 1, pageB = b.page || 1;
        if (pageA !== pageB) return pageA - pageB;
        const boxA = a.box_2d || [0,0,0,0], boxB = b.box_2d || [0,0,0,0];
        if (boxA[1] !== boxB[1]) return boxA[1] - boxB[1];
        return boxA[0] - boxB[0];
    }) : [];

    // Store for pagination
    tablePageState[template_name] = { annotations: sortedTableAnns, currentPage: 0 };

    const isReviewed = !!(reviewsCache[template_name] && reviewsCache[template_name].reviewed);

    return `
        <div id="${template_name}" class="template-section active${isReviewed ? ' reviewed' : ''}">
            <div class="template-header">
                <h2>${display_name}${isReviewed ? ' <span class="reviewed-badge">Reviewed</span>' : ''}</h2>
                <div class="template-stats">
                    <span class="bates-id">Id: ${doc_id}</span>
                    <span>Pages: ${total_pages}</span>
                    <span>Annotations: ${annotation_count}</span>
                </div>
            </div>
            <div class="template-toolbar">
                <button class="toolbar-btn refresh-btn" onclick="refreshTemplate('${template_name}')" title="Regenerate this template with new random data">
                    <span class="refresh-icon">↻</span> Refresh
                </button>
                <button class="toolbar-btn overlay-toggle active" onclick="toggleOverlay(this, '${template_name}')" title="Toggle bounding box overlays">
                    <span class="overlay-toggle-icon">▣</span> Overlays
                </button>
                <button class="toolbar-btn graph-toggle" onclick="toggleGraphView(this, '${template_name}')" title="Toggle RDFa graph visualization">
                    <span>◈</span> Graph
                </button>
                <button class="toolbar-btn note-btn" onclick="openPageNoteDialog('${template_name}')" title="Add a note for the current page">
                    <span class="note-btn-icon">+</span> Note
                </button>
                <span class="toolbar-separator"></span>
                <button class="toolbar-btn review-btn${isReviewed ? ' active' : ''}" onclick="toggleReview('${template_name}', this)" title="${isReviewed ? 'Mark as not reviewed' : 'Mark as reviewed'}">
                    <span class="review-icon">${isReviewed ? '✓' : '○'}</span><span class="review-label"> ${isReviewed ? 'Done' : 'Mark Done'}</span>
                </button>
                <span class="toolbar-separator"></span>
                <button class="toolbar-btn" disabled title="Coming soon: Run OCR extraction">
                    OCR
                </button>
                <button class="toolbar-btn" disabled title="Coming soon: Run layout segmentation">
                    Segment
                </button>
                <button class="toolbar-btn" disabled title="Coming soon: Compare GT vs predictions">
                    GT vs Pred
                </button>
            </div>
            
            <div class="template-content">
                ${pageThumbs ? `<div class="page-sidebar" id="page_sidebar_${template_name}">${pageThumbs}</div>` : ''}
                <div class="viewer-content">
                    ${pageContainers}
                </div>
                <div class="graph-panel" id="graph-panel-${template_name}" style="display:none">
                    <div class="graph-toolbar">
                        <button class="graph-mode-btn active" data-mode="force" onclick="switchGraphMode('${template_name}', 'force', this)">Force</button>
                        <button class="graph-mode-btn" data-mode="tree" onclick="switchGraphMode('${template_name}', 'tree', this)">Tree</button>
                        <span class="graph-info" id="graph-info-${template_name}"></span>
                    </div>
                    <div class="graph-container" id="graph-container-${template_name}"></div>
                </div>
            </div>
            
            <div class="bottom-panels" id="table-${template_name}">
                <div class="bottom-panels-header" onclick="toggleBottomPanel('table-${template_name}')">
                    <h3><span class="collapse-icon">${annotationsTableCollapsed ? '▶' : '▼'}</span> ${annotations ? `Annotations (<span class="ann-count">${annotation_count}</span>)` : 'Details'}<span class="notes-header-badge"> · Notes (<span class="notes-count">${getTemplateNotes(template_name).length}</span>)</span></h3>
                    ${annotations ? `
                    <div class="table-filters" onclick="event.stopPropagation()" ${annotationsTableCollapsed ? 'style="display:none"' : ''}>
                        <input type="text" class="table-search" placeholder="Search..." oninput="filterAnnotations('${template_name}')">
                        <select class="table-type-filter" onchange="filterAnnotations('${template_name}')">
                            <option value="">All Types</option>
                            ${(unique_types || []).map(t => `<option value="${t}">${t}</option>`).join('')}
                        </select>
                    </div>` : ''}
                </div>
                <div class="bottom-panels-content" ${annotationsTableCollapsed ? 'style="display:none"' : ''}>
                    ${annotations ? `
                    <div class="annotations-table-container">
                        <div class="table-container">
                            <table class="data-table">
                                <thead>
                                    <tr>
                                        <th>Page</th>
                                        <th>Label</th>
                                        <th>Type</th>
                                        <th>Text Content</th>
                                        <th>Bounding Box</th>
                                    </tr>
                                </thead>
                                <tbody id="tbody-${template_name}"></tbody>
                            </table>
                        </div>
                        <div class="table-pagination" id="pagination-${template_name}"></div>
                    </div>` : ''}
                    <div class="notes-panel" id="notes-panel-${template_name}">
                        <div class="notes-panel-header">
                            <h3>Notes (<span class="notes-count-inner">${getTemplateNotes(template_name).length}</span>)</h3>
                        </div>
                        <div class="notes-panel-body"></div>
                    </div>
                </div>
            </div>
        </div>`;
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.querySelector('.main-content');
    const btn = document.getElementById('sidebar-toggle');
    if (!sidebar) return;
    const collapsed = sidebar.classList.toggle('collapsed');
    if (mainContent) mainContent.classList.toggle('sidebar-collapsed', collapsed);
    if (btn) btn.textContent = collapsed ? '▶' : '◀';
    if (btn) btn.title = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
    localStorage.setItem('inspectorSidebarCollapsed', collapsed);
}

// Graph view code moved to graph.js


function toggleOverlay(btn, templateName) {
    const section = document.getElementById(templateName);
    if (!section) return;
    const overlays = section.querySelectorAll('.overlay-layer');
    const isActive = btn.classList.toggle('active');
    overlays.forEach(svg => svg.style.display = isActive ? '' : 'none');
}

async function toggleReview(templateName, btn) {
    if (!currentRunId) return;

    const wasReviewed = !!(reviewsCache[templateName] && reviewsCache[templateName].reviewed);
    const nowReviewed = !wasReviewed;

    // Optimistic UI update
    const section = document.getElementById(templateName);
    const badge = section?.querySelector('.reviewed-badge');
    const icon = btn.querySelector('.review-icon');
    const label = btn.querySelector('.review-label');

    if (nowReviewed) {
        btn.classList.add('active');
        icon.textContent = '✓';
        if (label) label.textContent = ' Done';
        section?.classList.add('reviewed');
        const h2 = section?.querySelector('.template-header h2');
        if (h2 && !h2.querySelector('.reviewed-badge')) {
            h2.insertAdjacentHTML('beforeend', ' <span class="reviewed-badge">Reviewed</span>');
        }
    } else {
        btn.classList.remove('active');
        icon.textContent = '○';
        if (label) label.textContent = ' Mark Done';
        section?.classList.remove('reviewed');
        badge?.remove();
    }
    btn.title = nowReviewed ? 'Mark as not reviewed' : 'Mark as reviewed';

    // Update sidebar indicator
    const sidebarLink = document.querySelector(`.sidebar-nav a[data-template="${templateName}"]`);
    if (sidebarLink) {
        sidebarLink.classList.toggle('reviewed', nowReviewed);
    }

    // Update header stats
    if (nowReviewed) {
        reviewsCache[templateName] = { reviewed: true, timestamp: new Date().toISOString() };
    } else {
        delete reviewsCache[templateName];
    }
    updateRunStats(templatesDataCache);

    // Persist to server
    try {
        await fetch(`/api/inspector/run/${currentRunId}/review`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ template_name: templateName, reviewed: nowReviewed })
        });
    } catch (e) {
        console.error('Failed to save review state:', e);
    }
}

function toggleBottomPanel(panelId) {
    const container = document.getElementById(panelId);
    if (!container) return;
    const content = container.querySelector('.bottom-panels-content');
    const filters = container.querySelector('.table-filters');
    const icon = container.querySelector('.collapse-icon');
    if (!content) return;
    const collapsed = content.style.display === 'none';
    content.style.display = collapsed ? '' : 'none';
    if (filters) filters.style.display = collapsed ? '' : 'none';
    icon.textContent = collapsed ? '▼' : '▶';
    annotationsTableCollapsed = !collapsed;
    localStorage.setItem('inspectorAnnotationsCollapsed', !collapsed);
}

// Legacy alias in case anything still calls the old name
function toggleAnnotationsTable(tableId) {
    toggleBottomPanel(tableId);
}

function filterAnnotations(templateName) {
    const state = tablePageState[templateName];
    if (!state) return;

    const container = document.getElementById(`table-${templateName}`);
    if (!container) return;
    const searchVal = (container.querySelector('.table-search')?.value || '').toLowerCase();
    const typeVal = container.querySelector('.table-type-filter')?.value || '';

    state.filtered = state.annotations.filter(a => {
        if (typeVal && (a.entity_type || '') !== typeVal) return false;
        if (searchVal) {
            const text = `${a.label || ''} ${a.entity_type || ''} ${a.text_content || ''}`.toLowerCase();
            if (!text.includes(searchVal)) return false;
        }
        return true;
    });

    // Update count display
    const countEl = container.querySelector('.ann-count');
    if (countEl) countEl.textContent = state.filtered.length;

    renderTablePage(templateName, 0);
}

function renderTablePage(templateName, page) {
    const state = tablePageState[templateName];
    if (!state) return;

    const data = state.filtered || state.annotations;
    const totalPages = Math.max(1, Math.ceil(data.length / ROWS_PER_PAGE));
    page = Math.max(0, Math.min(page, totalPages - 1));
    state.currentPage = page;

    const start = page * ROWS_PER_PAGE;
    const slice = data.slice(start, start + ROWS_PER_PAGE);

    const tbody = document.getElementById(`tbody-${templateName}`);
    if (tbody) {
        tbody.innerHTML = slice.map(entity => {
            const p = entity.page || 1;
            const globalIdx = entity._idx !== undefined ? entity._idx : 0;
            const label = entity.label || '';
            const entityType = entity.entity_type || '';
            const text = (entity.text_content || '').substring(0, 100);
            const bbox = entity.box_2d || [];
            const bboxStr = bbox.length ? `[${bbox.join(', ')}]` : '';
            return `<tr class="annotation-row" data-idx="${globalIdx}" data-page="${p}" onmouseenter="highlightAnnotation(${globalIdx}, ${p})" onmouseleave="unhighlightAnnotation(${globalIdx}, ${p})">
                <td>${p}</td>
                <td class="label-cell">${label}</td>
                <td class="type-cell">${entityType}</td>
                <td class="text-cell">${text}</td>
                <td class="bbox-cell">${bboxStr}</td>
            </tr>`;
        }).join('');
        const tableContainer = tbody.closest('.table-container');
        if (tableContainer) tableContainer.scrollTop = 0;
    }

    const paginationEl = document.getElementById(`pagination-${templateName}`);
    if (paginationEl) {
        if (totalPages <= 1 || annotationsTableCollapsed) {
            paginationEl.style.display = 'none';
        } else {
            paginationEl.style.display = '';
            paginationEl.innerHTML = `
                <button onclick="renderTablePage('${templateName}', 0)" ${page === 0 ? 'disabled' : ''}>&laquo;</button>
                <button onclick="renderTablePage('${templateName}', ${page - 1})" ${page === 0 ? 'disabled' : ''}>&lsaquo;</button>
                <span class="page-info">${data.length ? start + 1 : 0}–${Math.min(start + ROWS_PER_PAGE, data.length)} of ${data.length}</span>
                <button onclick="renderTablePage('${templateName}', ${page + 1})" ${page >= totalPages - 1 ? 'disabled' : ''}>&rsaquo;</button>
                <button onclick="renderTablePage('${templateName}', ${totalPages - 1})" ${page >= totalPages - 1 ? 'disabled' : ''}>&raquo;</button>
            `;
        }
    }
}

function closeImageModal() {
    document.getElementById('imageModal').style.display = '';
}

// ============== Runs Modal ==============

let runsShowErrorsOnly = false;
let currentRunLog = '';

async function openRunsModal() {
    if (document.body.getAttribute('data-mode') === 'static') return;
    const modal = document.getElementById('runsModal');
    modal.style.display = 'flex';

    const listEl = document.getElementById('runs-list');
    listEl.innerHTML = '<div style="padding:20px;color:#9498ab">Loading runs...</div>';

    try {
        const response = await fetch('/api/inspector/runs');
        const data = await response.json();
        const runs = data.runs || [];

        if (runs.length === 0) {
            listEl.innerHTML = '<div style="padding:20px;color:#9498ab">No runs found</div>';
            return;
        }

        listEl.innerHTML = runs.map(run => `
            <div class="run-item" data-run-id="${run.id}" onclick="selectRun('${run.id}')">
                <div class="run-item-header">
                    <span class="run-item-id">${run.timestamp}</span>
                    <span class="run-status-badge ${run.status || 'unknown'}">${run.status || '?'}</span>
                </div>
                <div class="run-item-stats">
                    <span>${run.template_count} templates</span>
                    ${run.error_count ? `<span style="color:#fca5a5">${run.error_count} errors</span>` : ''}
                    ${run.elapsed ? `<span>${run.elapsed}s</span>` : ''}
                </div>
            </div>
        `).join('');
    } catch (e) {
        listEl.innerHTML = `<div style="padding:20px;color:#fca5a5">Failed to load runs: ${e.message}</div>`;
    }
}

function closeRunsModal(event) {
    if (event && event.target !== event.currentTarget) return;
    document.getElementById('runsModal').style.display = '';
}

async function selectRun(runId) {
    // Update active state in list
    document.querySelectorAll('.run-item').forEach(el => el.classList.remove('active'));
    const item = document.querySelector(`.run-item[data-run-id="${runId}"]`);
    if (item) item.classList.add('active');

    const headerEl = document.getElementById('run-log-header');
    const contentEl = document.getElementById('run-log-content');

    headerEl.innerHTML = `<span>${runId}</span>
        <button class="log-filter-toggle ${runsShowErrorsOnly ? 'active' : ''}" onclick="toggleErrorFilter()">
            Errors Only
        </button>
        <button class="log-filter-toggle" onclick="switchToRun('${runId}')">
            Load Run
        </button>`;
    contentEl.innerHTML = 'Loading log...';

    try {
        const response = await fetch(`/api/inspector/run/${runId}/log`);
        const data = await response.json();
        currentRunLog = data.log || '(no log file)';
        renderLog(contentEl);
    } catch (e) {
        contentEl.textContent = `Failed to load log: ${e.message}`;
    }
}

function renderLog(contentEl) {
    if (!contentEl) contentEl = document.getElementById('run-log-content');
    const lines = currentRunLog.split('\n');

    const errorPatterns = /error|traceback|exception|timed out|failed/i;
    const warnPatterns = /warning|warn/i;

    const html = lines.map(line => {
        const isError = errorPatterns.test(line);
        const isWarn = !isError && warnPatterns.test(line);

        if (runsShowErrorsOnly && !isError) return null;

        if (isError) return `<span class="log-line-error">${escapeHtml(line)}</span>`;
        if (isWarn) return `<span class="log-line-warn">${escapeHtml(line)}</span>`;
        return `<span class="log-line-info">${escapeHtml(line)}</span>`;
    }).filter(Boolean).join('\n');

    contentEl.innerHTML = html || '<span class="log-line-info">(no matching lines)</span>';
}

function toggleErrorFilter() {
    runsShowErrorsOnly = !runsShowErrorsOnly;
    const btn = document.querySelector('.log-filter-toggle');
    if (btn) btn.classList.toggle('active', runsShowErrorsOnly);
    renderLog();
}

function switchToRun(runId) {
    closeRunsModal();
    const selector = document.getElementById('run-selector');
    if (selector) {
        selector.value = runId;
        switchRun();
    }
}

// ============== Page Navigation ==============

function scrollToPage(templateName, pageNum) {
    const section = document.getElementById(templateName);
    if (!section) return;

    const pageContainer = section.querySelector(`.page-container[data-page="${pageNum}"]`);
    if (pageContainer) {
        pageContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // Keep active page state in sync
    activePageByTemplate[templateName] = pageNum;

    // Update active thumb
    section.querySelectorAll('.page-thumb').forEach(thumb => {
        thumb.classList.toggle('active', parseInt(thumb.dataset.page) === pageNum);
    });
}

// ============== Tooltip ==============

let customTooltip = null;

function createTooltip() {
    if (!customTooltip) {
        customTooltip = document.createElement('div');
        customTooltip.className = 'custom-tooltip';
        document.body.appendChild(customTooltip);
    }
    return customTooltip;
}

function showTooltip(event, label, entityType, textContent, bbox) {
    const tooltip = createTooltip();
    
    tooltip.innerHTML = `
        <div class="tooltip-label">${label}</div>
        <div class="tooltip-type">${entityType}</div>
        <div class="tooltip-text">${textContent}</div>
        <div class="tooltip-coords">Box: [${bbox.join(', ')}]</div>
    `;
    
    const x = event.clientX + 15;
    const y = event.clientY + 15;
    
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
    tooltip.classList.add('visible');
}

function hideTooltip() {
    if (customTooltip) {
        customTooltip.classList.remove('visible');
    }
}

function updateTooltipPosition(event) {
    if (customTooltip && customTooltip.classList.contains('visible')) {
        const x = event.clientX + 15;
        const y = event.clientY + 15;
        customTooltip.style.left = x + 'px';
        customTooltip.style.top = y + 'px';
    }
}

function highlightAnnotation(idx, page) {
    // Find and highlight the corresponding SVG rect
    const pageContainer = document.querySelector(`.page-container[data-page="${page}"]`);
    if (!pageContainer) return;
    
    const rect = pageContainer.querySelector(`rect[data-idx="${idx}"]`);
    if (rect) {
        rect.classList.add('highlighted');
        rect.style.fill = 'rgba(255, 0, 0, 0.3)';
        rect.style.stroke = 'red';
        rect.style.strokeWidth = '2';
        
        // Scroll the rect into view within the viewer-content container
        const viewer = pageContainer.closest('.viewer-content');
        if (viewer) {
            const viewerRect = viewer.getBoundingClientRect();
            const elemRect = rect.getBoundingClientRect();
            const offset = elemRect.top - viewerRect.top - (viewerRect.height / 2) + (elemRect.height / 2);
            viewer.scrollBy({ top: offset, behavior: 'smooth' });
        }
    }
}

function unhighlightAnnotation(idx, page) {
    const pageContainer = document.querySelector(`.page-container[data-page="${page}"]`);
    if (!pageContainer) return;
    
    const rect = pageContainer.querySelector(`rect[data-idx="${idx}"]`);
    if (rect) {
        rect.classList.remove('highlighted');
        rect.style.fill = '';
        rect.style.stroke = '';
        rect.style.strokeWidth = '';
    }
}

function scrollToAnnotationRow(idx, page) {
    if (!currentlyLoadedTemplate) return;
    const state = tablePageState[currentlyLoadedTemplate];
    if (!state) return;

    // Check if table is visible (panel must be expanded)
    const tableContainer = document.getElementById(`table-${currentlyLoadedTemplate}`);
    if (!tableContainer) return;
    const panelContent = tableContainer.querySelector('.bottom-panels-content');
    if (!panelContent || panelContent.style.display === 'none') return;

    // Find which page this annotation is on
    const annIdx = state.annotations.findIndex(a =>
        (a._idx !== undefined ? a._idx : 0) === idx && (a.page || 1) === page
    );
    if (annIdx === -1) return;

    const targetPage = Math.floor(annIdx / ROWS_PER_PAGE);
    if (targetPage !== state.currentPage) {
        renderTablePage(currentlyLoadedTemplate, targetPage);
    }

    // Now find and highlight the row
    requestAnimationFrame(() => {
        const row = document.querySelector(`.annotation-row[data-idx="${idx}"][data-page="${page}"]`);
        if (!row) return;
        row.scrollIntoView({ behavior: 'smooth', block: 'center' });
        row.style.background = 'rgba(37, 99, 235, 0.3)';
        setTimeout(() => { row.style.background = ''; }, 1500);
    });
}

// ============== Active Page Tracking ==============

function setupPageScrollTracking(templateName) {
    const section = document.getElementById(templateName);
    if (!section) return;
    const viewer = section.querySelector('.viewer-content');
    if (!viewer) return;

    // Default to page 1
    activePageByTemplate[templateName] = 1;

    const pages = viewer.querySelectorAll('.page-container[data-page]');
    if (pages.length <= 1) return;

    const observer = new IntersectionObserver((entries) => {
        let bestPage = activePageByTemplate[templateName] || 1;
        let bestRatio = 0;
        entries.forEach(entry => {
            if (entry.isIntersecting && entry.intersectionRatio > bestRatio) {
                bestRatio = entry.intersectionRatio;
                bestPage = parseInt(entry.target.dataset.page);
            }
        });
        if (bestPage !== activePageByTemplate[templateName]) {
            activePageByTemplate[templateName] = bestPage;
            // Sync thumbnail highlight
            section.querySelectorAll('.page-thumb').forEach(thumb => {
                thumb.classList.toggle('active', parseInt(thumb.dataset.page) === bestPage);
            });
        }
    }, { root: viewer, threshold: [0, 0.25, 0.5, 0.75, 1.0] });

    pages.forEach(page => observer.observe(page));
}

// ============== Page-Level Notes ==============

function openPageNoteDialog(templateName) {
    if (document.body.getAttribute('data-mode') === 'static') return;
    const page = activePageByTemplate[templateName] || 1;

    // Check existing page-level notes
    const existing = notesCache.filter(n =>
        n.template_name === templateName && n.page === page && n.annotation_idx == null
    );

    pendingNote = { templateName, page, annIdx: null, label: null, entityType: null, textContent: null, bbox: null };

    const metaEl = document.getElementById('note-meta');
    metaEl.innerHTML = `
        <div class="note-meta-row"><span class="note-meta-label">Template:</span> <span>${templateName}</span></div>
        <div class="note-meta-row"><span class="note-meta-label">Page:</span> <span>${page}</span></div>
        <div class="note-meta-row note-page-level-badge">Page-level note</div>
        ${existing.length > 0 ? `
            <div class="note-existing">
                <strong>Existing notes:</strong>
                ${existing.map(n => `
                    <div class="note-existing-item">
                        <span class="note-existing-text">${escapeHtml(n.note)}</span>
                        <button class="note-delete-btn" onclick="deleteNote('${n.id}')" title="Delete note">&times;</button>
                    </div>
                `).join('')}
            </div>
        ` : ''}
    `;

    document.getElementById('note-text').value = '';
    document.getElementById('noteDialog').style.display = 'flex';
    setTimeout(() => document.getElementById('note-text').focus(), 50);
}

// ============== Annotation Notes ==============

let pendingNote = null;  // stores context for the note being created

function openNoteMenu(event, templateName, page, annIdx, label, entityType, textContent, bbox) {
    event.preventDefault();
    event.stopPropagation();
    if (document.body.getAttribute('data-mode') === 'static') return;
    hideTooltip();

    // Check if there's already a note for this annotation
    const existing = notesCache.filter(n =>
        n.template_name === templateName && n.annotation_idx === annIdx && n.page === page
    );

    pendingNote = { templateName, page, annIdx, label, entityType, textContent, bbox };

    const metaEl = document.getElementById('note-meta');
    metaEl.innerHTML = `
        <div class="note-meta-row"><span class="note-meta-label">Template:</span> <span>${templateName}</span></div>
        <div class="note-meta-row"><span class="note-meta-label">Label:</span> <span>${label}</span></div>
        <div class="note-meta-row"><span class="note-meta-label">Type:</span> <span>${entityType}</span></div>
        <div class="note-meta-row"><span class="note-meta-label">Text:</span> <span>${textContent || '(empty)'}</span></div>
        <div class="note-meta-row"><span class="note-meta-label">Page:</span> <span>${page}</span></div>
        <div class="note-meta-row"><span class="note-meta-label">Box:</span> <span>[${bbox.join(', ')}]</span></div>
        ${existing.length > 0 ? `
            <div class="note-existing">
                <strong>Existing notes:</strong>
                ${existing.map(n => `
                    <div class="note-existing-item">
                        <span class="note-existing-text">${escapeHtml(n.note)}</span>
                        <button class="note-delete-btn" onclick="deleteNote('${n.id}')" title="Delete note">&times;</button>
                    </div>
                `).join('')}
            </div>
        ` : ''}
    `;

    document.getElementById('note-text').value = '';
    document.getElementById('noteDialog').style.display = 'flex';
    setTimeout(() => document.getElementById('note-text').focus(), 50);
}

function closeNoteDialog() {
    document.getElementById('noteDialog').style.display = 'none';
    pendingNote = null;
}

async function saveNote() {
    if (!pendingNote || !currentRunId) return;
    const noteText = document.getElementById('note-text').value.trim();
    if (!noteText) return;

    const btn = document.getElementById('note-save-btn');
    btn.disabled = true;
    btn.textContent = 'Saving...';

    try {
        const resp = await fetch(`/api/inspector/run/${currentRunId}/notes`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                template_name: pendingNote.templateName,
                page: pendingNote.page,
                annotation_idx: pendingNote.annIdx,
                label: pendingNote.label,
                entity_type: pendingNote.entityType,
                text_content: pendingNote.textContent,
                bbox: pendingNote.bbox,
                note: noteText,
            })
        });

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const entry = await resp.json();
        notesCache.push(entry);

        // Mark the rect as having a note
        markNoteOnRect(pendingNote.templateName, pendingNote.page, pendingNote.annIdx, true);

        // Update notes panel if visible
        updateNotesPanel(pendingNote.templateName);

        closeNoteDialog();
    } catch (e) {
        console.error('Failed to save note:', e);
        alert(`Failed to save: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Save Note';
    }
}

async function deleteNote(noteId) {
    if (!currentRunId) return;
    try {
        await fetch(`/api/inspector/run/${currentRunId}/notes/${noteId}`, { method: 'DELETE' });
        const deleted = notesCache.find(n => n.id === noteId);
        notesCache = notesCache.filter(n => n.id !== noteId);

        // Unmark rect if no more notes for that annotation
        if (deleted) {
            const remaining = notesCache.filter(n =>
                n.template_name === deleted.template_name &&
                n.annotation_idx === deleted.annotation_idx &&
                n.page === deleted.page
            );
            if (remaining.length === 0) {
                markNoteOnRect(deleted.template_name, deleted.page, deleted.annotation_idx, false);
            }
            updateNotesPanel(deleted.template_name);
        }

        // Refresh the note dialog meta if still open
        if (pendingNote && deleted &&
            pendingNote.templateName === deleted.template_name &&
            pendingNote.annIdx === deleted.annotation_idx) {
            openNoteMenu(
                new MouseEvent('contextmenu'),
                pendingNote.templateName, pendingNote.page, pendingNote.annIdx,
                pendingNote.label, pendingNote.entityType, pendingNote.textContent, pendingNote.bbox
            );
        }
    } catch (e) {
        console.error('Failed to delete note:', e);
    }
}

function markNoteOnRect(templateName, page, annIdx, hasNote) {
    const section = document.getElementById(templateName);
    if (!section) return;
    const rect = section.querySelector(
        `.page-container[data-page="${page}"] rect[data-idx="${annIdx}"]`
    );
    if (rect) {
        rect.classList.toggle('has-note', hasNote);
    }
}

function getTemplateNotes(templateName) {
    return notesCache.filter(n => n.template_name === templateName);
}

function updateNotesPanel(templateName) {
    const panel = document.getElementById(`notes-panel-${templateName}`);
    if (!panel) return;

    const notes = getTemplateNotes(templateName);
    const body = panel.querySelector('.notes-panel-body');

    if (notes.length === 0) {
        body.innerHTML = '<div class="notes-empty">No notes yet. Use + Note or right-click a bbox.</div>';
    } else {
        body.innerHTML = notes.map(n => {
            const isPageLevel = n.annotation_idx == null;
            const summary = n.note.length > 60 ? n.note.substring(0, 60) + '...' : n.note;
            const tag = isPageLevel ? 'Page' : (n.label || '?');
            return `
            <div class="note-card-compact${isPageLevel ? ' note-compact-page-level' : ''}" onclick="openNoteDetail('${n.id}')" title="${escapeHtml(n.note)}">
                <span class="note-compact-tag${isPageLevel ? ' page-note-tag' : ''}">${escapeHtml(tag)}</span>
                <span class="note-compact-page">p${n.page || '?'}</span>
                <span class="note-compact-text">${escapeHtml(summary)}</span>
            </div>`;
        }).join('');
    }

    // Update count badges (inner panel + header)
    const innerBadge = panel.querySelector('.notes-count-inner');
    if (innerBadge) innerBadge.textContent = notes.length;
    const container = panel.closest('.bottom-panels');
    if (container) {
        const headerBadge = container.querySelector('.bottom-panels-header .notes-count');
        if (headerBadge) headerBadge.textContent = notes.length;
    }
}


function openNoteDetail(noteId) {
    const n = notesCache.find(x => x.id === noteId);
    if (!n) return;

    const isPageLevel = n.annotation_idx == null;

    pendingNote = {
        templateName: n.template_name,
        page: n.page,
        annIdx: n.annotation_idx,
        label: n.label,
        entityType: n.entity_type,
        textContent: n.text_content,
        bbox: n.bbox
    };

    const metaEl = document.getElementById('note-meta');
    metaEl.innerHTML = `
        <div class="note-meta-row"><span class="note-meta-label">Template:</span> <span>${n.template_name}</span></div>
        <div class="note-meta-row"><span class="note-meta-label">Page:</span> <span>${n.page || '?'}</span></div>
        ${isPageLevel ? '<div class="note-meta-row note-page-level-badge">Page-level note</div>' : `
            <div class="note-meta-row"><span class="note-meta-label">Label:</span> <span>${escapeHtml(n.label || '')}</span></div>
            <div class="note-meta-row"><span class="note-meta-label">Type:</span> <span>${escapeHtml(n.entity_type || '')}</span></div>
            ${n.text_content ? `<div class="note-meta-row"><span class="note-meta-label">Text:</span> <span>${escapeHtml(n.text_content)}</span></div>` : ''}
            ${n.bbox ? `<div class="note-meta-row"><span class="note-meta-label">Box:</span> <span>[${n.bbox.join(', ')}]</span></div>` : ''}
        `}
        <div class="note-existing">
            <div class="note-existing-item">
                <span class="note-existing-text">${escapeHtml(n.note)}</span>
                <button class="note-delete-btn" onclick="deleteNote('${n.id}')" title="Delete note">&times;</button>
            </div>
        </div>
    `;

    document.getElementById('note-text').value = '';
    document.getElementById('noteDialog').style.display = 'flex';
    setTimeout(() => document.getElementById('note-text').focus(), 50);
}

// ============== Init ==============

window.addEventListener('DOMContentLoaded', async function() {
    const isDevServerMode = document.body.dataset.mode === 'dynamic';

    // Restore sidebar collapsed state from localStorage
    if (localStorage.getItem('inspectorSidebarCollapsed') === 'true') {
        toggleSidebar();
    }

    if (isDevServerMode) {
        showGenerationUI();

        // Handle hash routing for dev server
        if (window.location.hash) {
            const { templateName, graph } = parseHash(window.location.hash);
            if (templateName) {
                await renderTemplate(templateName);
                if (graph) activateGraphView(templateName);
            }
        }

        // Listen for hash changes
        window.addEventListener('hashchange', async function() {
            const { templateName, graph } = parseHash(window.location.hash);
            if (templateName) {
                await renderTemplate(templateName);
                if (graph) activateGraphView(templateName);
            }
        });
    } else {
        // Static site mode — data injected at build time via window.STATIC_DATA
        const devHeader = document.getElementById('dev-header') || document.querySelector('.dev-header');
        if (devHeader) devHeader.style.display = 'none';
        // Disable right-click note menu in static mode
        document.addEventListener('contextmenu', function(e) {
            if (e.target.closest('svg')) e.preventDefault();
        });
        templatesDataCache = (window.STATIC_DATA && window.STATIC_DATA.templates) || [];
        document.getElementById('empty-state').style.display = 'none';
        document.getElementById('template-container').style.display = '';
        buildSidebarFromTemplates(templatesDataCache);

        const { templateName: initialTemplate, graph: initialGraph } = parseHash(window.location.hash);
        const initial = (initialTemplate && templatesDataCache.find(t => t.template_name === initialTemplate))
            ? initialTemplate
            : (templatesDataCache[0] && templatesDataCache[0].template_name);
        if (initial) {
            await renderTemplate(initial);
            if (initialGraph && initial === initialTemplate) activateGraphView(initial);
        }

        window.addEventListener('hashchange', async function() {
            const { templateName, graph } = parseHash(window.location.hash);
            if (templateName) {
                await renderTemplate(templateName);
                if (graph) activateGraphView(templateName);
            }
        });
    }

    document.querySelectorAll('.sidebar-nav a[data-template]').forEach(link => {
        link.addEventListener('click', async function(e) {
            e.preventDefault();
            const templateName = this.getAttribute('data-template');

            // Update hash for both static and dev modes
            window.location.hash = templateName;

            await renderTemplate(templateName);

            document.querySelectorAll('.sidebar-nav a').forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });
});
