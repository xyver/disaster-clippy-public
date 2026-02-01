# Building Code Explorer

Planning document for a Disaster Clippy deployment focused on building codes, regulations, and insurance policies to help homeowners prepare their properties for disasters.

**Part of the Sheltrium ecosystem** - integrates County Map risk data with Clippy knowledge search.

---

## Integration with Sheltrium

Building Code Explorer is a feature within the Sheltrium webapp, not a standalone project.

### Three-Project Ecosystem (from Strategic Doc)

| Project | Purpose | Data |
|---------|---------|------|
| **County Map** | Geographic risk visualization | Official datasets (FEMA NRI, NOAA, USGS, etc.) |
| **Disaster Clippy** | Knowledge/how-to search | Curated websites, wikis, PDFs, building codes |
| **Sheltrium** | Product/service marketplace | Products, vendors, experts, user accounts |

### How Building Code Explorer Connects Them

```
User enters address in Sheltrium
         |
         v
+------------------+
| Location Service |  --> loc_id: USA-FL-086
+------------------+
         |
         +---> County Map API (risk score)
         |         |
         |         v
         |     Risk Score Response:
         |     {
         |       "hurricane_risk": 0.82,
         |       "flood_risk": 0.71,
         |       "wind_zone": "170mph",
         |       "fema_flood_zone": "AE",
         |       "seismic_category": "A",
         |       "wildfire_risk": 0.12
         |     }
         |
         +---> Building Codes (Clippy variant)
                   |
                   v
               Enable sources:
               - federal/*
               - states/USA-FL/*
               - counties/USA-FL-086/*
               - user insurance (if uploaded)
         |
         v
+------------------+
| Chat Interface   |  User asks questions, LLM has:
| (Sheltrium UI)   |  - Risk score context
+------------------+  - Building code sources
                      - Insurance docs
```

### LLM Context Injection

When user starts a chat, the LLM receives:

```
SYSTEM CONTEXT:

User Location: Miami-Dade County, FL (USA-FL-086)
Special Zones: HVHZ (High-Velocity Hurricane Zone), FEMA Flood Zone AE

Risk Profile (from County Map):
- Hurricane Risk: 82/100 (Very High)
- Flood Risk: 71/100 (High)
- Design Wind Speed: 170 mph
- Seismic Design Category: A (Low)
- Wildfire Risk: 12/100 (Low)

Enabled Building Code Sources:
- Federal: FEMA P-55, NFIP Manual, HUD Standards
- State: Florida Building Code 8th Ed, FL Fire Marshal
- County: Miami-Dade Amendments, HVHZ Requirements
- Insurance: State Farm Policy (uploaded by user)

User can ask about building requirements, insurance discounts,
property upgrades, and disaster preparedness for their specific location.
```

### Revenue Connection

From Sheltrium strategic doc - Building Code Explorer feeds into:

1. **Product recommendations** - "Your roof needs 170 mph rating" -> link to approved products
2. **Expert consultations** - "Talk to a contractor about HVHZ compliance" -> book expert
3. **Insurance partnerships** - "Complete these upgrades" -> premium reduction verification
4. **Prep score improvement** - Verified upgrades improve user's preparedness score

---

## Concept

Users enter their location and insurance provider(s), then ask questions about:
- Building code requirements for their area
- Insurance policy requirements and discounts
- Property upgrades that improve safety AND reduce premiums
- Compliance requirements at different jurisdictional levels

**Key Value:** Everything properly sourced back to the original documents - users can verify and cite the actual regulations.

---

## Document Hierarchy (USA Focus)

Documents layer from broad to specific. When answering questions, we need to consider all applicable layers:

```
Federal (applies everywhere)
    |
    v
State (50 states, each with different rules)
    |
    v
County / Municipality (thousands of local jurisdictions)
    |
    v
Insurance Company (user-specific policies)
```

### Layer Details

| Layer | Example Sources | Scope |
|-------|-----------------|-------|
| **Federal** | FEMA guidelines, HUD standards, ADA requirements, federal flood insurance (NFIP) | ~10-20 core documents |
| **State** | State building codes, state fire marshal rules, state insurance regulations | ~50 states x 5-20 docs each |
| **County/Local** | Local amendments, zoning, permit requirements, local fire codes | Highly variable - start with major metros? |
| **Insurance** | Policy documents, underwriting guidelines, discount programs | User uploads or selects from known providers |

### Conflict Resolution (Code Cascade Logic)

Building codes cascade differently than data rollups - instead of aggregating UP, rules flow DOWN and can be overridden:

```
Federal (baseline)
    |
    v  State can ADOPT, AMEND, or ADD
State (modifies federal)
    |
    v  County can ADOPT, AMEND, or ADD to state
County (modifies state)
    |
    v  Insurance can REQUIRE MORE than code
Insurance (additional requirements)
```

**Cascade Rules:**

| Scenario | Resolution |
|----------|------------|
| Local stricter than state | Local wins |
| Local weaker than state | State minimum still applies |
| Insurance stricter than code | Insurance requirement applies (for premium) |
| Insurance weaker than code | Code still required (legal minimum) |

**How We Present This:**

For each topic, show the **effective requirement** with source trail:

```
EFFECTIVE REQUIREMENT: 170 mph wind rating

Why:
- Federal (FEMA): Recommends 130 mph for coastal FL
- State (FL Building Code): Requires 150 mph for HVHZ
- County (Miami-Dade): Requires 170 mph + NOA certification  <-- GOVERNING
- Insurance (State Farm): Requires code compliance (no additional)
```

**Metadata for Cascade:**

Sources need to indicate their relationship to parent:

```json
{
  "source_id": "miami-dade-amendments",
  "jurisdiction": {
    "loc_id": "USA-FL-086",
    "admin_level": 2
  },
  "code_relationship": {
    "base_code": "fl-building-code",
    "relationship": "amends",
    "sections_modified": ["1609", "1620", "2404"],
    "effective_date": "2024-01-01"
  }
}
```

---

## User Flow

```
1. User enters location
   - Address, or
   - Zip code, or
   - City/State selection

2. System determines applicable jurisdictions
   - Federal (always)
   - State (from location)
   - County (from location)
   - City/Municipality (from location)

3. User selects insurance provider(s)
   - Dropdown of known providers, or
   - "Upload my policy" option, or
   - "Skip - just show codes"

4. System enables relevant source filters
   - All applicable document sources now active
   - Search scoped to user's context

5. User asks questions
   - "What wind rating do I need for my roof?"
   - "Does adding hurricane shutters reduce my premium?"
   - "What are the setback requirements for a fence?"
```

---

## Location Service Integration

Bryan has existing location-finding scripts in another project.

### Required Output Format

Location service should return data compatible with loc_id system:

```json
{
  "input": "123 Main St, Miami, FL 33101",
  "loc_id": "USA-FL-086",
  "hierarchy": {
    "country": "USA",
    "state": "FL",
    "state_name": "Florida",
    "county_fips": "086",
    "county_name": "Miami-Dade",
    "municipality": "Miami",
    "zip": "33101"
  },
  "coordinates": {
    "lat": 25.7617,
    "lon": -80.1918
  },
  "special_zones": [
    {
      "type": "FEMA_FLOOD",
      "zone": "AE",
      "description": "High-risk flood area with base flood elevation"
    },
    {
      "type": "HVHZ",
      "zone": "HIGH_VELOCITY_HURRICANE",
      "description": "Miami-Dade/Broward High-Velocity Hurricane Zone"
    },
    {
      "type": "WILDFIRE",
      "zone": null,
      "description": null
    }
  ]
}
```

### Special Zones That Affect Building Codes

| Zone Type | Source | Affects |
|-----------|--------|---------|
| FEMA Flood Zone (A, AE, V, VE, X) | FEMA NFHL | Foundation, elevation, flood vents |
| HVHZ (High-Velocity Hurricane Zone) | FL Building Code | Impact windows, roof, shutters |
| Wildfire Risk (WUI) | USFS | Defensible space, exterior materials |
| Seismic Design Category (A-F) | IBC/ASCE 7 | Structural, foundations |
| Wind Speed Zone | ASCE 7 | Roof, windows, garage doors |
| Coastal Construction Control Line | State DEP | Setbacks, foundation type |

### Integration Points

1. **county-map-data already has:**
   - FEMA NRI (risk scores, seismic, wind)
   - Wildfire risk data
   - County FIPS mapping
   - Geometry for spatial queries

2. **Need to add or source:**
   - FEMA NFHL flood zone lookup (by lat/lon or address)
   - HVHZ boundary (FL-specific)
   - Coastal construction lines (state-specific)
   - Wind speed map (ASCE 7 reference)

**Questions for Bryan:**
- What format does your location service currently return?
- Does it already have FEMA flood zone lookup?
- Can we add special zone detection to it?

---

## Source Organization (Adapted from county-map-data Pipeline)

Reusing the hierarchical data pipeline architecture from Bryan's county-map project.

### Folder Structure

```
building-codes/
  index.json                    # Router - which jurisdictions have docs
  catalog.json                  # All source metadata for search context

  federal/                      # Admin level 0 - applies everywhere
    fema-guidelines/
      _manifest.json
      _metadata.json
      _index.json
      _vectors.json
      pdfs/
    hud-standards/
    nfip-flood-insurance/
    ada-accessibility/

  states/                       # Admin level 1 - state-specific
    USA-FL/
      index.json                # FL sources summary
      fl-building-code/
        _manifest.json
        _metadata.json
        ...
      fl-fire-marshal/
    USA-CA/
      ca-building-code/
      ca-title-24-energy/
    USA-TX/
      ...

  counties/                     # Admin level 2 - county/municipal
    USA-FL-086/                 # Miami-Dade (FIPS 086)
      miami-dade-amendments/
      miami-dade-hvhz/          # High-Velocity Hurricane Zone
    USA-CA-037/                 # Los Angeles County
      la-county-fire-code/

  insurance/                    # User-specific (private)
    user-{session_id}/
      state-farm-policy/
      citizens-policy/
```

### loc_id Integration

Reuse the county-map loc_id format for user locations:

| Level | Format | Example |
|-------|--------|---------|
| Country | `USA` | `USA` |
| State | `USA-{ST}` | `USA-FL` |
| County | `USA-{ST}-{FIPS}` | `USA-FL-086` |
| ZIP | `USA-{ST}-Z{ZIP5}` | `USA-FL-Z33101` |

**User location lookup flow:**
```
User enters: "33101" or "Miami, FL"
           |
           v
Location service returns:
{
  "loc_id": "USA-FL-086",
  "state": "FL",
  "county_fips": "086",
  "county_name": "Miami-Dade",
  "zip": "33101",
  "special_zones": ["FEMA-AE", "HVHZ"]
}
           |
           v
Enable sources:
  - federal/* (always)
  - states/USA-FL/*
  - counties/USA-FL-086/*
  - insurance/user-xyz/* (if uploaded)
```

### index.json Router (Building Codes Version)

```json
{
  "_description": "Building code source routing by jurisdiction",
  "_schema_version": "1.0.0",

  "federal": {
    "path": "federal",
    "applies_to": "all",
    "sources": ["fema-guidelines", "hud-standards", "nfip-flood-insurance", "ada-accessibility"]
  },

  "_default_state": {
    "has_folder": false,
    "sources": []
  },

  "USA-FL": {
    "name": "Florida",
    "has_folder": true,
    "path": "states/USA-FL",
    "sources": ["fl-building-code", "fl-fire-marshal"],
    "special_zones": {
      "HVHZ": {
        "name": "High-Velocity Hurricane Zone",
        "counties": ["USA-FL-086", "USA-FL-011"],
        "additional_sources": ["fl-hvhz-requirements"]
      }
    }
  },

  "USA-FL-086": {
    "name": "Miami-Dade County",
    "has_folder": true,
    "path": "counties/USA-FL-086",
    "parent": "USA-FL",
    "sources": ["miami-dade-amendments", "miami-dade-hvhz"]
  },

  "USA-CA": {
    "name": "California",
    "has_folder": true,
    "path": "states/USA-CA",
    "sources": ["ca-building-code", "ca-title-24-energy"]
  }
}
```

### Source Metadata Schema

Each source has a `_manifest.json` with jurisdiction info:

```json
{
  "source_id": "fl-building-code",
  "name": "Florida Building Code 8th Edition (2023)",
  "description": "Statewide building code adopted by Florida",
  "jurisdiction": {
    "type": "state",
    "loc_id": "USA-FL",
    "name": "Florida",
    "admin_level": 1
  },
  "document_type": "building_code",
  "effective_date": "2024-01-01",
  "supersedes": "fl-building-code-7th",
  "base_code": "ICC-IBC-2021",
  "amendments": true,
  "source_url": "https://floridabuilding.org/",
  "license": "Public record",
  "tags": ["construction", "structural", "mechanical", "plumbing", "electrical"]
}
```

### Scaling Strategy

| Level | Who Indexes | Count | Priority |
|-------|-------------|-------|----------|
| Federal | Us | ~10-20 docs | Phase 1 |
| State (priority) | Us | FL, CA, TX, LA, NC first | Phase 1 |
| State (remaining) | Us over time | 45 more states | Phase 2-3 |
| County (major metros) | Us | Top 20 counties | Phase 2 |
| County (others) | Community/partners | Thousands | Future |
| Insurance | User uploads | Per-user | Phase 2 |

---

## Document Acquisition Strategy

### Federal (We Index)
- FEMA publications (public domain)
- HUD guidelines
- NFIP flood insurance manual
- ADA accessibility requirements

### State (We Index Major States First)
Priority states (high disaster risk + population):
1. Florida (hurricanes)
2. California (earthquakes, fires)
3. Texas (hurricanes, tornadoes)
4. Louisiana (hurricanes, floods)
5. North Carolina (hurricanes)

**Question:** Are state building codes copyrighted? Need to check licensing.

### County/Local (Phased Approach)
- Start with major metros in priority states
- Allow user contribution (with verification)
- Partner with local governments?

### Insurance (User-Provided)
- User uploads their policy PDF
- We extract and index relevant sections
- Keep private to that user (not shared)

---

## PDF Indexer Implementation

The section-aware PDF indexer is implemented in `scraper/pdf.py`. It handles building codes and structured documents.

### Implemented Features

| Feature | Status | Notes |
|---------|--------|-------|
| **Section numbering** | Done | Detects 1, 1.1, 1.2.3, Chapter X, Appendix A patterns |
| **Page references** | Done | Generates #page=N URLs for citations |
| **Section hierarchy** | Done | Tracks parent sections, builds breadcrumb paths |
| **TOC skipping** | Done | `--start-page` option to skip title/TOC pages |
| **Natural chunking** | Done | Preserves section boundaries, only splits very large sections |
| **Cross-references** | Future | "See Section 5.4" linking not yet implemented |
| **Tables** | Future | Table extraction not yet implemented |
| **OCR** | Future | Scanned docs need OCR pre-processing |

### Usage

```bash
# Preview PDF structure
python -c "
from scraper.pdf import PDFScraper
scraper = PDFScraper(source_name='building-codes')
info = scraper.get_pdf_info('document.pdf', start_page=5)
print(f'Sections: {info[\"detected_sections\"]}')
for s in info['section_preview'][:10]:
    print(f'  Page {s[\"page\"]}: [{s[\"number\"]}] {s[\"title\"][:50]}')
"

# Process via CLI
python ingest.py pdf create fortified-2025 --license "IBHS"
python ingest.py pdf add --collection fortified-2025 --file "document.pdf"
python ingest.py pdf process fortified-2025 --sectioned --start-page 5
```

### Section Detection Patterns

The indexer detects these section header formats:

```
# Pattern 1: Number on same line as title
1.2.3 Roof Deck Requirements

# Pattern 2: Number on own line (common in FORTIFIED docs)
4
Designation Requirements for FORTIFIED Roof

# Pattern 3: Named sections
Chapter 3: Wind Resistance
Appendix A. Technical Resources

# Pattern 4: Subsections with split lines
4.4.1
Sealing the Roof Deck for Shingle and Metal Roof Covers
```

### Output Format

Each section becomes a searchable chunk with:

- **Title**: "2025 FORTIFIED Home Standard - 4.4.1 Sealing the Roof Deck..."
- **URL**: `file://path/to/doc.pdf#page=37`
- **Content**: Section text with breadcrumb context
- **Source**: Collection ID for filtering

Example citation in LLM response:
> "Roof decks must be sealed per Section 4.4.1 (page 37) of the FORTIFIED Home Standard."

---

## Search & Response Behavior

### Source Resolution Pipeline

When user sets their location, we resolve applicable sources:

```
User Location (loc_id: USA-FL-086)
         |
         v
+------------------+
| index.json       |  Read routing table
| lookup           |
+------------------+
         |
         v
Build source list:
  1. federal/* (always)
  2. Walk up hierarchy:
     - USA-FL-086 -> sources: [miami-dade-amendments, miami-dade-hvhz]
     - USA-FL -> sources: [fl-building-code, fl-fire-marshal]
     - Check special_zones: HVHZ -> [fl-hvhz-requirements]
  3. insurance/user-xyz/* (if exists)
         |
         v
Final enabled_sources = [
  "fema-guidelines", "hud-standards", "nfip-flood-insurance",
  "fl-building-code", "fl-fire-marshal", "fl-hvhz-requirements",
  "miami-dade-amendments", "miami-dade-hvhz",
  "user-xyz-state-farm"
]
```

### Multi-Layer Search

When user asks a question, search should:
1. Search all enabled sources (federal + state + county + insurance)
2. Tag results by jurisdiction layer (admin_level in metadata)
3. Present results grouped or annotated by source

### Response Format

```
Q: "What wind rating do I need for my roof in Miami?"

A: Based on your location (Miami-Dade County, FL), here are the requirements:

**Federal (FEMA):**
- Recommends minimum 130 mph wind rating for coastal Florida
- Source: FEMA P-55, page 47

**Florida State Building Code:**
- Requires 170 mph for High-Velocity Hurricane Zone (HVHZ)
- Source: FL Building Code 7th Ed, Section 1609.2

**Miami-Dade County:**
- Requires Miami-Dade NOA (Notice of Acceptance) certification
- Additional testing requirements beyond state code
- Source: Miami-Dade County Code, Chapter 8

**Your Insurance (State Farm):**
- 10% discount for impact-rated roofing
- Requires professional installation certificate
- Source: Your policy, page 23
```

---

## Data Model Changes

### New: User Session Context

Stored per-session, enables source filtering:

```json
{
    "session_id": "xyz-123",
    "created_at": "2025-01-27T10:00:00Z",

    "location": {
        "input": "123 Main St, Miami, FL 33101",
        "loc_id": "USA-FL-086",
        "hierarchy": {
            "country": "USA",
            "state": "FL",
            "county_fips": "086",
            "county_name": "Miami-Dade",
            "zip": "33101"
        },
        "special_zones": ["FEMA-AE", "HVHZ"]
    },

    "insurance_providers": [
        {
            "name": "State Farm",
            "source_id": "insurance-xyz-123-state-farm",
            "uploaded": true
        }
    ],

    "resolved_sources": {
        "federal": ["fema-guidelines", "hud-standards", "nfip-flood-insurance"],
        "state": ["fl-building-code", "fl-fire-marshal"],
        "county": ["miami-dade-amendments", "miami-dade-hvhz"],
        "special_zone": ["fl-hvhz-requirements"],
        "insurance": ["insurance-xyz-123-state-farm"]
    },

    "enabled_source_ids": [
        "fema-guidelines", "hud-standards", "nfip-flood-insurance",
        "fl-building-code", "fl-fire-marshal", "fl-hvhz-requirements",
        "miami-dade-amendments", "miami-dade-hvhz",
        "insurance-xyz-123-state-farm"
    ]
}
```

### Source _manifest.json Schema

Each source folder contains jurisdiction metadata:

```json
{
    "source_id": "miami-dade-amendments",
    "name": "Miami-Dade County Building Code Amendments",
    "description": "Local amendments to Florida Building Code for Miami-Dade County",

    "jurisdiction": {
        "type": "county",
        "loc_id": "USA-FL-086",
        "name": "Miami-Dade County",
        "admin_level": 2,
        "parent": "USA-FL"
    },

    "code_info": {
        "document_type": "building_code_amendment",
        "base_code": "fl-building-code",
        "relationship": "amends",
        "sections_modified": ["1609", "1620", "2404"],
        "effective_date": "2024-01-01",
        "supersedes": "miami-dade-amendments-2020"
    },

    "special_zones": ["HVHZ"],

    "source_url": "https://www.miamidade.gov/permits/library.asp",
    "license": "Public record",
    "tags": ["hurricane", "wind", "impact-resistant", "structural"]
}
```

### index.json Entry Schema

```json
{
    "USA-FL-086": {
        "name": "Miami-Dade County",
        "has_folder": true,
        "path": "counties/USA-FL-086",
        "parent": "USA-FL",
        "admin_level": 2,
        "sources": ["miami-dade-amendments", "miami-dade-hvhz"],
        "special_zones": ["HVHZ", "FEMA-AE", "FEMA-VE"],
        "zone_sources": {
            "HVHZ": ["fl-hvhz-requirements"]
        }
    }
}
```

---

## catalog.json for LLM Context

Like county-map, aggregate source info for the AI to understand what's available:

```json
{
    "catalog_version": "1.0",
    "last_updated": "2025-01-27",
    "total_sources": 45,

    "coverage_summary": {
        "federal": 4,
        "states": 5,
        "counties": 12,
        "insurance_templates": 3
    },

    "sources": [
        {
            "source_id": "fema-guidelines",
            "name": "FEMA Building Guidelines",
            "jurisdiction": {"type": "federal", "admin_level": 0},
            "document_type": "guidelines",
            "topics": ["flood", "wind", "earthquake", "construction"],
            "llm_summary": "Federal emergency preparedness guidelines. Covers flood zone construction, wind resistance, seismic design. Reference for all states."
        },
        {
            "source_id": "fl-building-code",
            "name": "Florida Building Code 8th Edition",
            "jurisdiction": {"type": "state", "loc_id": "USA-FL", "admin_level": 1},
            "document_type": "building_code",
            "base_code": "ICC-IBC-2021",
            "topics": ["structural", "mechanical", "plumbing", "electrical", "hurricane"],
            "llm_summary": "Florida statewide building code. Stricter than IBC for wind/hurricane. HVHZ counties have additional requirements."
        }
    ],

    "jurisdiction_tree": {
        "USA": {
            "federal": ["fema-guidelines", "hud-standards", "nfip-flood-insurance"],
            "states": {
                "FL": {
                    "sources": ["fl-building-code", "fl-fire-marshal"],
                    "counties": {
                        "086": ["miami-dade-amendments"]
                    }
                },
                "CA": {
                    "sources": ["ca-building-code", "ca-title-24-energy"]
                }
            }
        }
    }
}
```

The AI uses this to:
1. Know what jurisdictions have data
2. Understand document relationships (FL code is based on IBC)
3. Route questions to appropriate sources

---

## Open Questions

1. **Copyright/Licensing:** Can we legally index and redistribute building codes? ICC codes are copyrighted. Some states have adopted them but licensing unclear.

2. **Update Frequency:** Building codes update periodically. How do we handle versioning?

3. **User-Uploaded Insurance:** Privacy implications? Store locally only? Encrypt?

4. **Conflicting Requirements:** How to present when local code differs from state?

5. **Verification:** How do users know our indexed content matches current official codes?

6. **Scope Creep:** Do we include permit processes, contractor requirements, HOA rules?

---

## MVP Scope

For initial version, limit to:

- [x] PDF indexer with section-aware chunking (DONE)
- [x] Page reference URLs (#page=N) for citations (DONE)
- [x] CLI integration (ingest.py pdf process --sectioned) (DONE)
- [ ] 3-5 states (FL, CA, TX) - start with FORTIFIED standard
- [ ] Federal FEMA/HUD docs
- [ ] Basic location lookup (zip to state/county)
- [ ] No insurance upload yet (just codes)

---

## Implementation Path

### What to Build Where

| Component | Location | Status |
|-----------|----------|--------|
| PDF Indexer upgrade | disaster-clippy/scraper/pdf.py | DONE |
| CLI integration | disaster-clippy/ingest.py | DONE |
| Building code sources | New folder (building-codes/) | Not started |
| Location service | Existing scripts | Needs output format |
| Risk score API | county-map-data | Needs endpoint |
| Source resolver | New module | Not started |
| Chat integration | Sheltrium webapp | Not started |

### Phase 1: Core Infrastructure (DONE)

1. **PDF Indexer Upgrade** (disaster-clippy) - COMPLETE
   - Page-aware extraction via `_extract_pages()`
   - Section detection via `_detect_section_headers()`
   - #page=N URL generation in `process_file_sectioned()`
   - CLI: `python ingest.py pdf process <collection> --sectioned --start-page N`

2. **Risk Score API** (county-map-data)
   - Endpoint: `/api/risk/{loc_id}`
   - Returns: hurricane, flood, wind, seismic, wildfire scores
   - Include special zones (FEMA flood zone, HVHZ, etc.)

3. **Location Service Output** (existing scripts)
   - Standardize output to loc_id format
   - Add special zone detection

### Phase 2: Building Code Sources

4. **Test Document: 2025 FORTIFIED Home Standard**
   - 100 pages, well-structured sections
   - Good test case for section detection
   - Process with: `--sectioned --start-page 5`

5. **Federal Documents**
   - FEMA P-55 (Coastal Construction Manual)
   - NFIP Flood Insurance Manual
   - HUD standards

6. **Florida (First State)**
   - FL Building Code 8th Edition
   - Miami-Dade county amendments
   - HVHZ requirements

### Phase 3: Sheltrium Integration

7. **Source Resolver Module**
   - Input: loc_id + special zones
   - Output: list of enabled source_ids
   - Uses index.json routing

8. **LLM Context Builder**
   - Combines risk score + enabled sources
   - Generates system prompt for chat

9. **Chat UI in Sheltrium**
   - Location entry (address or zip)
   - Insurance upload (optional)
   - Chat interface with sourced responses

---

## Open Questions

1. **Copyright/Licensing:** Can we legally index and redistribute building codes? ICC codes are copyrighted. Some states have adopted them but licensing unclear.

2. **Update Frequency:** Building codes update periodically. How do we handle versioning?

3. **User-Uploaded Insurance:** Privacy implications? Store locally only? Encrypt?

4. **Conflicting Requirements:** How to present when local code differs from state?

5. **Verification:** How do users know our indexed content matches current official codes?

6. **Scope Creep:** Do we include permit processes, contractor requirements, HOA rules?

7. **Risk Score API Design:** Separate endpoint or bundled with location lookup?

---

## MVP Scope

For initial version, limit to:

- [ ] PDF indexer with page-aware chunking
- [ ] Risk score API endpoint (county-map-data)
- [ ] Federal FEMA/HUD docs indexed
- [ ] Florida state codes indexed
- [ ] Miami-Dade county amendments indexed
- [ ] Basic location -> sources resolver
- [ ] LLM context injection with risk score
- [ ] Chat integration in Sheltrium (basic)

---

*Created: January 2025*
*Part of Sheltrium ecosystem*
