# Wayback Scanner Documentation

## Table of Contents
1. [Rationale and Overview](#rationale-and-overview)
2. [System Architecture](#system-architecture)
3. [Core Classes and Functions](#core-classes-and-functions)
4. [Data Collection Process](#data-collection-process)
5. [Diff Analysis Methodology](#diff-analysis-methodology)
6. [Keyword Analysis System](#keyword-analysis-system)
7. [Output Files and Columns](#output-files-and-columns)
8. [Magnitude Scoring System](#magnitude-scoring-system)
9. [How to Use This Analysis](#how-to-use-this-analysis)
10. [Technical Implementation Details](#technical-implementation-details)
11. [Troubleshooting and Optimization](#troubleshooting-and-optimization)

---

## Rationale and Overview

### Why This Tool Exists

The Wayback Scanner was designed to track the evolution of Colossal Biosciences' website messaging over time. As a company at the forefront of de-extinction technology, Colossal's public communications provide valuable insights into:

- **Strategic positioning changes** - How the company presents its mission and goals
- **Technical progress updates** - Evolution of claims about capabilities and timelines
- **Market messaging shifts** - Changes in how they address funding, partnerships, and commercial viability
- **Public relations adaptations** - Responses to criticism, regulatory changes, or scientific developments

### Core Problem Solved

Traditional web monitoring only captures current states. This tool leverages the Internet Archive's Wayback Machine to:

1. **Reconstruct historical website states** across multiple years
2. **Quantify changes** using semantic similarity algorithms
3. **Track keyword trends** across specific topics (ethics, timelines, technology, etc.)
4. **Generate actionable reports** for researchers, journalists, and analysts

### Key Innovation

Unlike simple screenshot comparisons, this system:
- Extracts semantic content from HTML while preserving structure
- Uses cosine distance on character n-grams for change detection
- Provides multi-dimensional magnitude scoring for different types of changes
- Generates both machine-readable CSV data and human-readable markdown reports

---

## System Architecture

### High-Level Flow

```
Internet Archive API → HTML Download → Text Extraction → Diff Analysis → Reports
```

### Core Components

1. **WaybackCollector**: Handles data acquisition and preprocessing
2. **DiffAnalyzer**: Performs change detection and quantification
3. **Magnitude Scoring**: Multi-metric change assessment
4. **Report Generation**: CSV exports and markdown summaries

### Data Pipeline

```
Raw HTML → Cleaned Text → Similarity Vectors → Change Metrics → Trend Analysis
```

---

## Core Classes and Functions

### WaybackCollector Class

**Purpose**: Manages the entire data collection process from Wayback Machine API to cleaned text files.

#### Key Methods:

**`__init__(self, base_domain, paths, timezone)`**
- Initializes collector with target domain and paths to monitor
- Sets up timezone handling for temporal analysis

**`sample_snapshots(self, all_snapshots: List[Dict], sampling: str) -> List[Dict]`**
- **Purpose**: Reduces snapshot volume while preserving temporal resolution
- **Sampling Options**:
  - `"quarterly"`: 4 snapshots per year, evenly distributed
  - `"monthly"`: 12 snapshots per year, one per month
- **Algorithm**: Groups snapshots by time period, selects one representative snapshot per period
- **Returns**: Filtered list maintaining chronological order

**`collect_snapshots(self) -> pd.DataFrame`**
- **Purpose**: Main orchestration method for data collection
- **Process**:
  1. Queries Wayback CDX API for all available snapshots
  2. Applies sampling strategy to reduce data volume
  3. Downloads HTML content for each selected snapshot
  4. Extracts meaningful text and metadata
  5. Saves processed data to local files
- **Error Handling**: Skips failed downloads, logs errors, continues processing
- **Output**: DataFrame with metadata for all successfully processed snapshots

**`_download_and_process(self, snapshot: Dict) -> Dict`**
- **Purpose**: Downloads single snapshot and extracts content
- **Process**:
  1. Fetches HTML from Wayback Machine archive URL
  2. Extracts taglines/slogans using CSS selectors
  3. Converts HTML to clean text using `html_to_text()`
  4. Saves text to file with standardized naming
  5. Returns metadata dictionary
- **Fallback**: Returns error information if download fails

### DiffAnalyzer Class

**Purpose**: Analyzes collected snapshots to identify and quantify changes over time.

#### Core Analysis Methods:

**`_keyword_counts(self, text: str) -> Dict[str, int]`**
- **Purpose**: Counts keyword occurrences across predefined categories
- **Categories Tracked**:
  - `de_extinction`: Core technology terms
  - `climate_benefit`: Environmental impact claims
  - `animal_welfare`: Ethical considerations
  - `indigenous`: Cultural sensitivity topics
  - `technology`: Technical capabilities (CRISPR, genomics)
  - `funding`: Financial/investment mentions
  - `timeline`: Temporal claims and deadlines
- **Method**: Uses regex word boundary matching for accuracy
- **Returns**: Dictionary mapping category names to occurrence counts

**`_calculate_change_magnitude(self, A_clean: str, B_clean: str, diff_lines: List[str], kw_delta: Dict[str, int]) -> Dict[str, float]`**
- **Purpose**: Multi-dimensional scoring of change magnitude
- **Metrics Calculated**:
  1. **Similarity Score** (0-1): Inverse cosine distance, higher = more similar
  2. **Character Change Ratio** (0-1): Proportion of text that changed
  3. **Diff Density** (0-1): Changes per total lines of content
  4. **Keyword Intensity** (0-1): Magnitude of keyword frequency changes
  5. **Structural Score** (0-1): Additions/deletions relative to content size
  6. **Combined Magnitude** (0-1): Weighted average of all metrics
- **Weighting**: Similarity (30%), Diff Density (25%), Char Change (20%), Keywords (15%), Structure (10%)
- **Returns**: Dictionary with all individual scores plus combined magnitude

**`_categorize_change_magnitude(self, magnitude: float) -> str`**
- **Purpose**: Converts numerical magnitude to human-readable categories
- **Categories**:
  - `"Major"`: ≥0.8 (fundamental content overhauls)
  - `"Substantial"`: ≥0.6 (significant messaging changes)
  - `"Moderate"`: ≥0.4 (noticeable updates)
  - `"Minor"`: ≥0.2 (small tweaks and additions)
  - `"Minimal"`: <0.2 (trivial changes, typos)

**`_pairwise(self, group: pd.DataFrame) -> List[Dict[str, Any]]`**
- **Purpose**: Compares consecutive snapshots for a single URL path
- **Process**:
  1. Sorts snapshots chronologically
  2. For each adjacent pair:
     - Loads text content from saved files
     - Calculates cosine distance using character 3-grams
     - Generates unified diff with meaningful context
     - Computes keyword frequency changes
     - Calculates magnitude metrics
     - Extracts tagline/slogan changes
  3. Filters diff output to focus on substantive changes
- **Returns**: List of comparison dictionaries with all metrics

**`run(self) -> Tuple[pd.DataFrame, pd.DataFrame]`**
- **Purpose**: Main analysis orchestration method
- **Process**:
  1. Groups snapshots by URL path
  2. Runs pairwise comparison for each path
  3. Filters results based on significance threshold
  4. Generates keyword trend analysis
  5. Creates multiple output formats (CSV, markdown)
  6. Produces time-series data for plotting
- **Returns**: (all_diffs_dataframe, significant_changes_dataframe)

### Utility Functions

**`html_to_text(url: str, html: str) -> str`**
- **Purpose**: Converts raw HTML to comprehensive, semantically-structured text with enhanced content preservation
- **Enhanced Features**:
  - **Semantic Content Labeling**: Different content types are labeled (headings, paragraphs, lists, tables, etc.)
  - **Metadata Extraction**: Comprehensive extraction of titles, meta descriptions, Open Graph data, keywords
  - **Structured Content Preservation**: Maintains hierarchy and relationships between content elements
  - **Navigation & Messaging**: Preserves header, navigation, and footer content that may contain strategic messaging
  - **Accessibility Content**: Extracts image alt text and link information
  - **Error-Resilient Processing**: Safe extraction methods prevent failures from malformed HTML

- **Detailed Process**:
  1. **HTML Parsing**: Uses BeautifulSoup with lxml parser for robust handling
  2. **Selective Content Removal**: Removes scripts, styles, forms while preserving messaging-relevant navigation
  3. **Metadata Extraction**: 
     - Page titles and Open Graph titles
     - Meta descriptions and Open Graph descriptions  
     - Keywords meta tags
  4. **Structured Content Extraction**:
     - Navigation elements (labeled as `[NAV_n]`)
     - Header content (labeled as `[HEADER_n]`)
     - Hierarchical headings (labeled as `[H1]`, `[H2]`, etc.)
     - Main content areas (labeled as `[MAIN_CONTENT]`)
     - Paragraphs (labeled as `[P_n]`)
     - Lists with structure (labeled as `[UL_n_START]`, `[UL_n_ITEM_n]`, `[UL_n_END]`)
     - Tables with headers and rows (labeled as `[TABLE_n_HEADER_n]`, `[TABLE_n_ROW_n]`)
     - Blockquotes (labeled as `[QUOTE_n]`)
     - Image alt text (labeled as `[IMG_ALT_n]`)
     - Important links with URLs (labeled as `[LINK_n]` or `[LINK_TEXT_n]`)
     - Footer content (labeled as `[FOOTER_n]`)
  5. **Content Assembly**: Combines metadata and structured content with clear separators
  6. **Normalization**: Applies whitespace normalization while preserving semantic structure

- **Enhanced Output Format**:
  ```
  [SOURCE] https://colossal.com/page
  [TITLE] Revolutionary De-Extinction Technology
  [META_DESC] Bringing back extinct species using advanced genetic engineering
  [OG_DESC] Colossal Biosciences: De-extinction company reviving woolly mammoths
  [KEYWORDS] de-extinction, CRISPR, genetic engineering, mammoth
  [CONTENT_START]
  
  [NAV_0] Home About Science Species News Contact
  [HEADER_0] Colossal Biosciences - De-extinction Leader
  [H1] Reviving Extinct Species for a Better Planet
  [MAIN_CONTENT] Our mission is to develop breakthrough genetic technologies...
  [P_0] We use advanced CRISPR gene editing to restore extinct species...
  [UL_0_START]
  [UL_0_ITEM_0] Woolly Mammoth revival project
  [UL_0_ITEM_1] Thylacine restoration initiative  
  [UL_0_END]
  [TABLE_0_START]
  [TABLE_0_HEADER_0] Species | Status | Timeline
  [TABLE_0_ROW_0] Woolly Mammoth | In Progress | 2028
  [TABLE_0_END]
  [QUOTE_0] "We're not just bringing back species, we're restoring ecosystems"
  [IMG_ALT_0] Woolly mammoth in arctic tundra landscape
  [LINK_0] Learn more about our science -> /science/approach
  [FOOTER_0] Copyright 2025 Colossal Biosciences. All rights reserved.
  ```

- **Analysis Benefits**:
  - **Granular Change Detection**: Semantic labels enable precise identification of what type of content changed
  - **Messaging Evolution Tracking**: Can track changes in navigation, headers, and footer messaging separately
  - **Content Type Analysis**: Different weighting can be applied to different content types (headings vs. body text)
  - **Accessibility Monitoring**: Alt text and link changes show evolving accessibility and SEO strategies
  - **Strategic Communication Analysis**: Link text and navigation changes reveal strategic positioning shifts

**`cosine_distance(a: str, b: str) -> float`**
- **Purpose**: Measures semantic similarity between two text documents
- **Method**:
  1. Generates 3-character shingles from each text
  2. Creates frequency vectors for each shingle set
  3. Calculates cosine distance between vectors
  4. Falls back to Jaccard similarity for edge cases
- **Range**: 0.0 (identical) to 1.0 (completely different)
- **Why 3-grams**: Balances granularity with noise resistance

**`extract_taglines(soup: BeautifulSoup) -> Dict[str, str]`**
- **Purpose**: Extracts marketing slogans and key messaging
- **Selectors Used**:
  - HTML headings (h1-h6)
  - Hero section text (.hero h1, .hero h2)
  - Banner headlines (.banner h1)
  - Designated tagline classes (.tagline, .slogan)
  - Meta descriptions and Open Graph descriptions
- **Returns**: Dictionary mapping selector names to extracted text

---

## Data Collection Process

### Snapshot Sampling Strategy

The system uses intelligent sampling to balance temporal resolution with processing efficiency:

**Quarterly Sampling** (default for historical analysis):
- Selects 4 representative snapshots per year
- Captures major seasonal updates and campaigns
- Reduces processing time while maintaining trend visibility

**Monthly Sampling** (for detailed analysis):
- Selects 12 snapshots per year
- Captures more granular changes and iterative updates
- Higher processing cost but better change detection

### URL Path Selection

Currently monitored paths (configurable via `CANONICAL_PATHS`):
- `/` - Homepage (primary messaging)
- `/about` - Company description and mission
- `/science` - Technical approach and capabilities
- `/species` - Target species and project status
- `/news` - Press releases and updates

### Error Handling

The collection process is designed for resilience:
- **Network Failures**: Skips failed downloads, continues with remaining snapshots
- **Parsing Errors**: Logs HTML parsing issues, saves partial content
- **Missing Snapshots**: Handles gaps in Wayback Machine coverage gracefully
- **Rate Limiting**: Implements delays between requests to respect archive.org policies

---

## Diff Analysis Methodology

### Change Detection Algorithm

**Step 1: Text Preprocessing**
- Remove source URL differences (focus on content changes)
- Normalize whitespace and formatting
- Preserve semantic structure while reducing noise

**Step 2: Similarity Calculation**
- Generate character 3-grams from both text versions
- Create frequency vectors for statistical comparison
- Calculate cosine distance between vectors
- Apply significance threshold (default: 0.05)

**Step 3: Contextual Diff Generation**
- Create unified diff with 3 lines of context
- Filter out trivial changes (very short additions/deletions)
- Focus on substantive content modifications
- Preserve meaningful structural changes

**Step 4: Semantic Analysis**
- Count keyword occurrences in both versions
- Calculate frequency deltas across topic categories
- Identify shifts in messaging emphasis
- Track terminology evolution over time

### Significance Threshold

**Current Setting**: 0.05 cosine distance
- **Rationale**: Captures meaningful changes while filtering noise
- **Calibration**: Based on analysis of known significant updates
- **Adjustment**: Can be lowered to 0.01 for hyper-sensitive detection

### False Positive Mitigation

**Source URL Filtering**: Ignores changes in archive URLs that don't reflect content changes
**Length Filtering**: Skips very short text changes that are likely formatting artifacts
**Context Preservation**: Maintains surrounding text to distinguish meaningful changes from noise
**Category Weighting**: Emphasizes changes in high-impact keyword categories

---

## Keyword Analysis System

### Overview

The keyword analysis system is a core component that tracks semantic changes in messaging across 15 specialized categories. This system enables researchers to quantify how companies shift their communication strategies over time by monitoring the frequency of domain-specific terminology.

### Keyword Categories and Rationale

#### **1. De-extinction Core Concepts** (`de_extinction`)
**Purpose**: Track central terminology around species revival
**Key Terms**: de-extinction, extinct, revive, resurrection, bring back, restore, species recovery, lost species
**Analysis Value**: Measures how prominently the core mission is presented

#### **2. Functional De-extinction** (`functional_de_extinction`)
**Purpose**: Monitor discussion of proxy species vs. true resurrection
**Key Terms**: functional de-extinction, proxy, surrogate, ecological replacement, ecosystem function, keystone species
**Analysis Value**: Tracks shift between purist vs. pragmatic approaches

#### **3. Climate & Environmental Benefits** (`climate_benefit`)
**Purpose**: Quantify environmental justification rhetoric
**Key Terms**: climate change, carbon sequestration, permafrost, ecosystem services, rewilding, biodiversity
**Analysis Value**: Shows emphasis on environmental vs. scientific motivations

#### **4. Conservation Alignment** (`conservation_alignment`)
**Purpose**: Track positioning within established conservation frameworks
**Key Terms**: IUCN, conservation status, endangered, threatened, WWF, red list, habitat protection
**Analysis Value**: Measures integration with mainstream conservation messaging

#### **5. Ethics & Animal Welfare** (`ethics_welfare`)
**Purpose**: Monitor attention to ethical considerations
**Key Terms**: welfare, ethics, bioethics, suffering, moral implications, responsibility, animal rights
**Analysis Value**: Tracks responsiveness to ethical criticism

#### **6. Indigenous & Cultural Considerations** (`indigenous_cultural`)
**Purpose**: Track cultural sensitivity and community engagement
**Key Terms**: indigenous, traditional knowledge, cultural heritage, sacred, community consent, spiritual significance
**Analysis Value**: Measures cultural awareness evolution

#### **7. Risk & Caution** (`risk_caution`)
**Purpose**: Monitor acknowledgment of potential dangers
**Key Terms**: risk, safety, precaution, unintended consequences, biosafety, containment, monitoring
**Analysis Value**: Shows balance between optimism and responsibility

#### **8. Hype & Breakthrough Claims** (`hype_breakthrough`)
**Purpose**: Track promotional and sensational language
**Key Terms**: moonshot, revolutionary, groundbreaking, game-changing, unprecedented, breakthrough
**Analysis Value**: Quantifies promotional intensity over time

#### **9. Technology & Methods** (`technology_methods`)
**Purpose**: Monitor technical approach communication
**Key Terms**: CRISPR, gene editing, ancient DNA, genomics, synthetic biology, cloning, stem cells
**Analysis Value**: Tracks technical sophistication and approach evolution

#### **10. Business & Funding** (`business_funding`)
**Purpose**: Track commercial and investment messaging
**Key Terms**: funding, investment, venture capital, valuation, revenue, commercialization, partnership
**Analysis Value**: Shows business maturity and market positioning

#### **11. Timeline Claims** (`timeline_claims`)
**Purpose**: Monitor temporal promises and expectations
**Key Terms**: years, timeline, soon, 2025-2035, phase, milestone, target, expect, plan
**Analysis Value**: Tracks ambition vs. realism in project timelines

#### **12. Regulatory & Legal** (`regulatory_legal`)
**Purpose**: Track engagement with regulatory frameworks
**Key Terms**: regulation, FDA, USDA, approval, compliance, oversight, legal framework
**Analysis Value**: Shows regulatory awareness and preparation

#### **13. Target Species** (`target_species`)
**Purpose**: Monitor species-specific messaging emphasis
**Key Terms**: mammoth, thylacine, dodo, passenger pigeon, dire wolf, northern white rhino
**Analysis Value**: Tracks project prioritization and marketing focus

#### **14. Scientific Validation** (`scientific_validation`)
**Purpose**: Track emphasis on research credibility
**Key Terms**: peer review, publication, study, evidence, Nature, Science, reproducible, methodology
**Analysis Value**: Measures scientific legitimacy positioning

#### **15. Opposition & Criticism** (`opposition_criticism`)
**Purpose**: Monitor acknowledgment of skepticism
**Key Terms**: criticism, opposition, controversy, skepticism, unrealistic, playing god, waste
**Analysis Value**: Shows awareness and response to criticism

### Keyword Selection Methodology

#### **Comprehensiveness Criteria**:
- **Domain Coverage**: Terms span entire de-extinction discourse ecosystem
- **Stakeholder Perspectives**: Includes scientific, ethical, business, and cultural viewpoints
- **Temporal Relevance**: Covers both current terminology and emerging concepts
- **Sentiment Spectrum**: Includes positive, negative, and neutral framings

#### **Accuracy Optimization**:
- **Synonym Inclusion**: Multiple variations of key concepts (e.g., "de-extinction", "de extinction", "deextinction")
- **Technical Precision**: Scientific terminology alongside colloquial terms
- **Cultural Sensitivity**: Proper diacritical marks (māori, Māori) and regional variations
- **Temporal Specificity**: Exact years (2025-2035) for timeline analysis

#### **Reliability Features**:
- **Word Boundary Matching**: Uses regex `\b` anchors to avoid false positives
- **Case Insensitive**: Captures variations in capitalization
- **Multi-word Phrases**: Handles complex terms like "functional de-extinction"
- **Context Independence**: Terms selected to be meaningful regardless of surrounding text

### Implementation Details

#### **Counting Algorithm**:
```python
def _keyword_counts(self, text: str) -> Dict[str, int]:
    text_l = text.lower()
    counts = {}
    for bucket, words in self.keywords.items():
        c = 0
        for w in words:
            c += len(re.findall(r"\b" + re.escape(w.lower()) + r"\b", text_l))
        counts[bucket] = c
    return counts
```

#### **Change Detection**:
- **Delta Calculation**: Compares keyword frequencies between consecutive snapshots
- **Trend Analysis**: Tracks frequency changes over time across categories
- **Magnitude Weighting**: Keyword changes contribute 15% to overall change magnitude score

### Analytical Applications

#### **Messaging Evolution Tracking**:
- Monitor shift from technical focus to environmental benefits
- Track response to regulatory pressure through increased compliance terminology
- Observe adaptation to criticism via ethics and risk acknowledgment

#### **Strategic Positioning Analysis**:
- Compare emphasis across different website sections
- Identify seasonal or event-driven messaging campaigns
- Detect correlation between external events and terminology shifts

#### **Stakeholder Communication Patterns**:
- Business terminology for investor-focused content
- Scientific validation language for academic credibility
- Cultural sensitivity terms for community engagement

#### **Competitive Intelligence**:
- Benchmark terminology evolution against industry standards
- Identify unique positioning strategies through keyword emphasis
- Track adoption of emerging field terminology

### Quality Assurance & Validation

#### **False Positive Mitigation**:
- Manual review of high-frequency terms for relevance
- Context validation for ambiguous terms
- Regular updates based on field terminology evolution

#### **Coverage Validation**:
- Cross-reference with academic literature terminology
- Include stakeholder-specific language from industry reports
- Monitor emerging terminology through scientific publications

#### **Bias Reduction**:
- Include both positive and negative framing terms
- Balance scientific, business, and ethical perspectives
- Avoid over-weighting any single stakeholder viewpoint

---

## Output Files and Columns

### Primary Outputs

The system generates several complementary data files:

#### 1. `diffs_all.csv` - Complete Change Log
**Purpose**: Every detected change between consecutive snapshots
**Key Columns**:
- `path`: URL path being monitored
- `slug`: Filename-safe version of path
- `from_ts` / `to_ts`: Timestamp range for comparison
- `from_url` / `to_url`: Wayback Machine archive URLs
- `cosine_distance`: Similarity metric (0=identical, 1=completely different)
- `from_chars` / `to_chars`: Character counts for size analysis
- `char_change`: Net change in content length
- `delta_[category]`: Keyword frequency changes by topic
- `diff_snippet`: Sample of actual text changes
- `magnitude_score`: Combined change magnitude (0-1)
- `change_category`: Human-readable magnitude classification
- `similarity_score`: Text similarity (inverse of cosine_distance)
- `char_change_ratio`: Normalized character change magnitude
- `diff_density`: Changes per line of content
- `keyword_intensity`: Magnitude of keyword shifts
- `structural_score`: Additions/deletions relative to content size

#### 2. `changes_significant.csv` - Filtered Important Changes
**Purpose**: Only changes exceeding significance threshold
**Content**: Same columns as diffs_all.csv but filtered for analysis focus
**Use Case**: Primary dataset for trend analysis and reporting

#### 3. `keyword_trends.csv` - Temporal Keyword Analysis
**Purpose**: Track messaging evolution across topic categories
**Key Columns**:
- `slug`: URL path identifier
- `path`: Full URL path
- `timestamp`: Snapshot timestamp
- `year`: Extracted year for temporal grouping
- `dt_local`: Human-readable local time
- `[category]`: Keyword count for each tracked category
  - `de_extinction`: De-extinction technology mentions
  - `climate_benefit`: Environmental impact claims
  - `animal_welfare`: Ethical considerations
  - `indigenous`: Cultural sensitivity topics
  - `technology`: Technical capability claims
  - `funding`: Investment and financial mentions
  - `timeline`: Temporal claims and deadlines

#### 4. `magnitude_timeseries.csv` - Time Series Analysis Data
**Purpose**: Optimized for plotting change magnitude over time
**Key Columns**:
- `date`: Parsed datetime for time series analysis
- `slug`: URL path identifier
- `magnitude_score`: Combined change magnitude
- `change_category`: Categorical magnitude description
- `similarity_score`: Individual similarity metric
- `char_change_ratio`: Individual character change metric
- `diff_density`: Individual diff density metric
- `keyword_intensity`: Individual keyword change metric
- `structural_score`: Individual structural change metric
- `magnitude_rolling_3`: 3-snapshot rolling average for trend smoothing

#### 5. `magnitude_monthly.csv` - Aggregated Monthly Statistics
**Purpose**: Monthly summaries for long-term trend analysis
**Key Columns**:
- `year_month`: Time period identifier
- `slug`: URL path identifier
- `magnitude_score_mean`: Average monthly change magnitude
- `magnitude_score_max`: Peak monthly change magnitude
- `magnitude_score_count`: Number of changes detected per month
- `cosine_distance_mean`: Average monthly similarity score
- `char_change_mean`: Average monthly content size changes

### Report Files

#### 1. `wayback_changes_summary.md` - Human-Readable Report
**Structure**:
- Executive summary with magnitude statistics
- Category breakdown of change types
- Top 5 highest magnitude changes
- Detailed change log with diff snippets
- Enhanced formatting with magnitude metrics

#### 2. `wayback_index.csv` - Snapshot Inventory
**Purpose**: Metadata for all collected snapshots
**Key Columns**:
- `path`: URL path
- `timestamp`: Wayback Machine timestamp
- `archive_url`: Direct link to archived version
- `dt_utc` / `dt_local`: Parsed timestamps
- `text_path`: Local file path for extracted text
- `taglines`: Extracted marketing messages (JSON format)

---

## Magnitude Scoring System

### Rationale

Simple similarity scores don't capture the full complexity of website changes. The magnitude scoring system provides multi-dimensional assessment:

### Individual Metrics

**1. Similarity Score (30% weight)**
- **Range**: 0.0 to 1.0
- **Calculation**: 1.0 minus cosine distance
- **Interpretation**: Higher values indicate more similar content
- **Use Case**: Overall content stability assessment

**2. Character Change Ratio (20% weight)**
- **Range**: 0.0 to 1.0
- **Calculation**: |new_length - old_length| / average_length
- **Interpretation**: Proportion of content that changed in size
- **Use Case**: Detecting major content additions or removals

**3. Diff Density (25% weight)**
- **Range**: 0.0 to 1.0
- **Calculation**: changed_lines / total_lines
- **Interpretation**: Intensity of line-by-line changes
- **Use Case**: Distinguishing between focused edits and wholesale rewrites

**4. Keyword Intensity (15% weight)**
- **Range**: 0.0 to 1.0
- **Calculation**: sum(|keyword_deltas|) / baseline_keyword_count
- **Interpretation**: Magnitude of topic emphasis shifts
- **Use Case**: Tracking strategic messaging changes

**5. Structural Score (10% weight)**
- **Range**: 0.0 to 1.0
- **Calculation**: (additions + deletions) / max_content_lines
- **Interpretation**: Structural reorganization magnitude
- **Use Case**: Detecting layout and organization changes

### Combined Magnitude Score

**Formula**:
```
magnitude = (similarity * 0.3) + (char_ratio * 0.2) + (diff_density * 0.25) + (keyword_intensity * 0.15) + (structural * 0.1)
```

**Calibration**: Weights based on empirical analysis of known significant changes

### Change Categories

**Major (≥0.8)**:
- Complete page redesigns
- Fundamental strategy pivots
- New product announcements

**Substantial (≥0.6)**:
- Significant messaging updates
- Major feature additions
- Important policy changes

**Moderate (≥0.4)**:
- Content updates and expansions
- Minor messaging adjustments
- Regular maintenance updates

**Minor (≥0.2)**:
- Small content additions
- Typo corrections
- Minor formatting changes

**Minimal (<0.2)**:
- Trivial edits
- Technical updates
- Negligible changes

---

## How to Use This Analysis

### For Researchers

**1. Longitudinal Studies**:
- Use `keyword_trends.csv` to track messaging evolution
- Analyze `magnitude_timeseries.csv` for change patterns
- Cross-reference with external events (funding rounds, scientific publications)

**2. Comparative Analysis**:
- Compare magnitude scores across different URL paths
- Identify which pages change most frequently
- Track consistency of messaging across site sections

**3. Event Correlation**:
- Filter `changes_significant.csv` by date ranges
- Correlate high-magnitude changes with external events
- Analyze response patterns to public relations challenges

### For Journalists

**1. Story Development**:
- Use markdown reports for human-readable summaries
- Focus on "Major" and "Substantial" category changes
- Extract specific text changes from diff snippets

**2. Fact-Checking**:
- Track evolution of specific claims using keyword analysis
- Identify when companies modify controversial statements
- Document timeline inconsistencies

**3. Trend Identification**:
- Use monthly aggregations to identify long-term patterns
- Track keyword category trends over time
- Identify periods of intensive messaging changes

### For Business Analysts

**1. Competitive Intelligence**:
- Monitor messaging strategy evolution
- Track response patterns to market events
- Analyze communication frequency and intensity

**2. Strategic Planning**:
- Learn from competitor messaging optimization
- Understand industry communication patterns
- Benchmark change management approaches

**3. Market Research**:
- Track evolution of value propositions
- Monitor technology positioning changes
- Analyze customer communication strategies

### Data Analysis Workflows

**Basic Trend Analysis**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load time series data
df = pd.read_csv('reports/magnitude_timeseries.csv')
df['date'] = pd.to_datetime(df['date'])

# Plot magnitude over time
plt.figure(figsize=(12, 6))
for slug in df['slug'].unique():
    data = df[df['slug'] == slug]
    plt.plot(data['date'], data['magnitude_rolling_3'], label=slug)
plt.legend()
plt.title('Website Change Magnitude Over Time')
plt.show()
```

**Keyword Evolution Analysis**:
```python
# Load keyword trends
trends = pd.read_csv('reports/keyword_trends.csv')

# Analyze funding mentions over time
funding_trends = trends.groupby('year')['funding'].sum()
plt.plot(funding_trends.index, funding_trends.values)
plt.title('Funding-Related Mentions by Year')
plt.show()
```

**Event Impact Analysis**:
```python
# Load significant changes
changes = pd.read_csv('reports/changes_significant.csv')
changes['date'] = pd.to_datetime(changes['from_ts'], format='%Y%m%d%H%M%S')

# Filter around specific events
event_date = '2023-06-01'
window = changes[
    (changes['date'] >= '2023-05-01') &
    (changes['date'] <= '2023-07-01')
]

# Analyze change patterns around events
print(window.groupby('change_category').size())
```

---

## Technical Implementation Details

### Performance Optimizations

**1. Intelligent Sampling**:
- Reduces API calls by 75% while maintaining temporal resolution
- Configurable sampling strategies for different analysis needs
- Preserves chronological ordering for accurate diff analysis

**2. Incremental Processing**:
- Checks for existing data before making new API calls
- Skips already-processed snapshots to enable resume functionality
- Maintains data consistency across multiple runs

**3. Memory Management**:
- Processes snapshots individually to avoid memory overload
- Streams large datasets rather than loading entirely into memory
- Efficient text storage using normalized whitespace

**4. Error Recovery**:
- Graceful handling of network timeouts and API errors
- Comprehensive logging for debugging and monitoring
- Partial result preservation for interrupted runs

### Scalability Considerations

**Storage Requirements**:
- ~1MB per snapshot for text content
- ~100KB per snapshot for metadata
- Total storage scales linearly with time range and sampling frequency

**Processing Time**:
- ~2-3 seconds per snapshot for download and processing
- ~1 second per comparison for diff analysis
- Total runtime: 5-10 minutes for typical analysis (70 snapshots)

**API Rate Limits**:
- Respects Wayback Machine usage guidelines
- Implements request delays to avoid overwhelming archive.org
- Handles rate limiting gracefully with exponential backoff

### Dependencies and Requirements

**Core Libraries**:
- `pandas`: Data manipulation and analysis
- `requests`: HTTP client for API communication
- `beautifulsoup4`: HTML parsing and content extraction
- `waybackpy`: Wayback Machine API integration
- `python-dateutil`: Timezone-aware datetime handling

**System Requirements**:
- Python 3.7+ for type hints and dataclass support
- 500MB+ available disk space for data storage
- Internet connection for Wayback Machine API access
- UTF-8 locale support for international text handling

---

## Troubleshooting and Optimization

### Common Issues

**1. "No snapshots found" Error**:
- **Cause**: URL path not archived by Wayback Machine
- **Solution**: Check `CANONICAL_PATHS` configuration, verify URLs exist
- **Alternative**: Use broader path patterns or historical URLs

**2. "Rate limit exceeded" Warning**:
- **Cause**: Too many rapid API requests
- **Solution**: Increase delay between requests, use smaller batch sizes
- **Prevention**: Run during off-peak hours, use existing data when possible

**3. "Text extraction failed" Error**:
- **Cause**: Malformed HTML or encoding issues
- **Solution**: Check individual snapshot URLs manually
- **Fallback**: System automatically uses basic text extraction

**4. "Memory usage too high" Warning**:
- **Cause**: Processing too many snapshots simultaneously
- **Solution**: Reduce sampling frequency or process in smaller batches
- **Optimization**: Clear intermediate data more frequently

### Performance Tuning

**For Faster Processing**:
- Use quarterly sampling instead of monthly
- Reduce number of monitored paths
- Increase significance threshold to filter more aggressively
- Process shorter time ranges

**For Higher Accuracy**:
- Use monthly or weekly sampling
- Lower significance threshold to 0.01
- Increase diff context lines for better change detection
- Add more keyword categories for specific domains

**For Better Change Detection**:
- Lower `SIGNIFICANT_CHANGE` threshold
- Increase keyword category coverage
- Use shorter sampling intervals
- Implement custom similarity metrics for specific content types

### Advanced Configuration

**Custom Keyword Categories**:
```python
CUSTOM_KEYWORDS = {
    "regulatory": ["FDA", "approval", "regulation", "compliance"],
    "partnerships": ["partner", "collaboration", "agreement", "alliance"],
    "milestones": ["milestone", "achievement", "breakthrough", "success"]
}
```

**Sampling Strategy Customization**:
```python
def custom_sampling(snapshots, target_count):
    # Implement domain-specific sampling logic
    # e.g., bias toward known event dates
    pass
```

**Content-Specific Text Extraction**:
```python
def domain_specific_extraction(html, url):
    # Custom extraction for specific website structures
    # e.g., focus on specific CSS classes or sections
    pass
```

### Monitoring and Maintenance

**Regular Health Checks**:
- Monitor storage usage and clean old data
- Verify API connectivity and rate limit compliance
- Check for new paths or URL structure changes
- Validate keyword category relevance over time

**Data Quality Assurance**:
- Spot-check diff results against manual inspection
- Verify magnitude scores align with intuitive change assessment
- Monitor for systematic biases in change detection
- Update keyword lists based on evolving terminology

**Performance Monitoring**:
- Track processing times and optimize bottlenecks
- Monitor API response times and error rates
- Analyze memory usage patterns for large datasets
- Profile diff calculation performance for optimization opportunities

---

## Conclusion

This Wayback Scanner provides a comprehensive framework for analyzing website evolution over time. By combining automated data collection, sophisticated change detection, and multi-dimensional magnitude scoring, it enables researchers, journalists, and analysts to understand how organizations adapt their public messaging in response to internal developments and external pressures.

The system's strength lies in its balance of automation and configurability - providing robust default behavior while allowing customization for specific analysis needs. The multi-format output ensures compatibility with both automated analysis workflows and human interpretation.

For questions, issues, or enhancement requests, consult the source code comments and logging output for additional technical details.
