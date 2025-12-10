# Knowledge Map Visualisation

Interactive 3D visualization of the document network for admin users.

## Goals

1. **Find gaps in coverage** - Sparse areas indicate missing topics
2. **Spot duplicate/redundant content** - Dense clusters of very similar documents
3. **Explore the knowledge base** - Interactive navigation of document relationships

## Target Users

- Local admins (testing, content curation)
- Global admins (production oversight)
- NOT end users (admin-only feature)

## Scale Requirements

- Current: ~20k documents
- Future: 100k+ documents
- Pre-computed visualization (not live-updated)

---

## Technical Approaches

### Approach A: Embedding Scatter Plot

Reduce 1536-dim embedding vectors to 3D for visualization.

**Algorithms:**
- **PCA** - Fast (~5 sec for 20k), no extra deps, shows variance directions
- **UMAP** - Slower (~1-3 min for 20k), needs `umap-learn`, better cluster preservation

**Pros:**
- Shows semantic similarity (similar content clusters together)
- Works without explicit link data
- Simple implementation

**Cons:**
- No explicit connections shown
- Clusters may be less defined than graph approach

### Approach B: Graph Visualization (Wikipedia-style)

Create a graph where documents are nodes and edges represent relationships.

**Inspiration:** Wikipedia visualization video
- Graph made with python-igraph
- Distributed Recursive Layout algorithm for positioning
- Leiden algorithm for community detection

**Edge creation options:**
1. Explicit links (if we have hyperlink data between docs)
2. Embedding similarity threshold (e.g., cosine > 0.8 = edge)
3. K-nearest neighbors (each doc connected to its K most similar)

**Pros:**
- Clear community structure
- Shows explicit relationships
- Familiar network visualization

**Cons:**
- More complex to implement
- Need to define edge criteria
- Graph layout algorithms can be slow for 100k+ nodes

### Approach C: Hybrid

Combine both approaches:
1. Use embedding similarity to create edges (threshold-based)
2. Use graph layout algorithm for positioning
3. Color by source/community
4. Show both structure AND semantic relationships

---

## Implementation Plan

### Phase 1: MVP (Embedding Scatter)

**Backend:**
- API endpoint: `/useradmin/api/generate-visualisation`
- Pull all vectors + metadata from ChromaDB
- Run PCA to reduce to 3D coordinates
- Save to `BACKUP_PATH/_visualisation.json`
- Background job with progress tracking

**Frontend:**
- New admin page: `/useradmin/visualise`
- Plotly 3D scatter plot
- Color by source
- Hover shows: title, source, doc_type, link
- Filter by source (checkboxes)
- "Regenerate" button to recompute

**Output format:**
```json
{
  "generated_at": "2025-12-09T12:00:00Z",
  "algorithm": "pca",
  "point_count": 20000,
  "sources": ["source1", "source2", ...],
  "points": [
    {
      "x": 1.23,
      "y": -0.45,
      "z": 2.10,
      "id": "doc_abc123",
      "title": "How to Filter Water",
      "source": "ready_gov",
      "doc_type": "guide",
      "url": "https://...",
      "local_url": "/backup/ready_gov/..."
    }
  ]
}
```

### Phase 2: Algorithm Options

- Add UMAP as alternative (requires `umap-learn`)
- Dropdown: `Algorithm: [PCA (fast)] [UMAP (better clusters)]`
- Cache multiple versions if needed

### Phase 3: Density/Duplicate Detection

- Compute nearest neighbor distances during generation
- Add `density_score` to each point
- Highlight high-density points (potential duplicates)
- Filter: "Show potential duplicates"

### Phase 4: Graph Visualization (Optional)

- Add edge creation based on similarity threshold
- Switch to graph layout (igraph)
- Community detection (Leiden algorithm)
- Show edges on hover/click

---

## UI Design

### Page Layout

```
+------------------------------------------+
| Header: Visualise    [Regenerate] [Algo v]|
+------------------------------------------+
| Filters:                                  |
| [ ] source1 (1234)  [ ] source2 (5678)   |
| [Select All] [Select None]               |
+------------------------------------------+
|                                          |
|         [3D Plotly Scatter]              |
|                                          |
|    (rotate, zoom, pan with mouse)        |
|                                          |
+------------------------------------------+
| Status: Generated 2 hours ago | 20,000 pts|
+------------------------------------------+
```

### Controls

- **Mouse drag**: Rotate view
- **Scroll**: Zoom in/out
- **Hover**: Show document info tooltip
- **Click**: Open document in new tab (optional)
- **Source checkboxes**: Filter visible points
- **Algorithm dropdown**: PCA / UMAP
- **Regenerate button**: Recompute visualization

### Color Scheme

Each source gets a distinct color from a palette:
- Use categorical color scale (Plotly's qualitative palettes)
- Legend shows source names
- Clicking legend item toggles visibility

---

## Performance Considerations

### Browser Limits

| Points | Performance |
|--------|-------------|
| <10k | Excellent |
| 10-50k | Good |
| 50-100k | Acceptable (WebGL) |
| >100k | May need sampling |

### Compute Time (estimated)

| Algorithm | 20k docs | 100k docs |
|-----------|----------|-----------|
| PCA | ~5 sec | ~30 sec |
| UMAP | ~1-3 min | ~15-30 min |

### Memory

- 100k x 1536 x 4 bytes = ~600MB during compute
- Output JSON: ~3-5MB for 100k points
- Browser memory: ~50-100MB for rendering

---

## Dependencies

### Required (Phase 1)
- numpy (already installed)
- scikit-learn (for PCA, likely already installed)

### Optional (Phase 2+)
- umap-learn (~50MB) - for UMAP algorithm
- python-igraph - for graph layout
- leidenalg - for community detection

---

## File Locations

- Visualization data: `BACKUP_PATH/_visualisation.json`
- Routes: `admin/routes/visualise.py` (new)
- Template: `admin/templates/visualise.html` (new)
- Job type: `generate_visualisation`

---

## Inspiration

Wikipedia network visualization video:
- Data from Wikipedia dumps
- Graph made with python-igraph
- Distributed Recursive Layout algorithm
- Leiden algorithm for community detection
- Valid articles exclude redirects, disambiguation, soft redirects
- Valid links from article body only (not "See Also" or footnotes)

---

## Questions to Resolve

1. Should clicking a point open the article, or show more details in a sidebar?
2. Do we want edge visualization in Phase 1, or save for later?
3. Sampling strategy for 100k+ points - random? stratified by source?
4. Should regeneration be a background job (like indexing) or synchronous?

---

*Created: December 2025*
