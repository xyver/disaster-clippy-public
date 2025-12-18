"""
Document Type Classifier - Prototype v2

Classifies documents into types based on content analysis.
Returns top 3 types with confidence scores.

Document Types:
- Hands-On: guide, manual, field_guide
- Research: academic, article, reference, training
- Response: report, case_study, protocol, checklist
- Planning: plan, policy, assessment
- Other: news, faq, multimedia, misc

"misc" is for low-content pages that don't have enough signals to classify.
"""

import re
import json
from dataclasses import dataclass
from typing import Optional, List, Tuple


# Document type definitions for LLM prompt
DOCUMENT_TYPES = {
    "guide": "Step-by-step instructions, tutorials, DIY builds, how-to content",
    "manual": "Equipment/product documentation, technical specs, user manuals",
    "field_guide": "Identification guides for plants, animals, materials, symptoms",
    "academic": "Peer-reviewed studies, scientific research papers",
    "article": "Informational content, explainers, encyclopedia entries, wiki articles",
    "reference": "Data sheets, specifications, lookup tables, definitions",
    "training": "Educational curricula, courses, learning modules",
    "report": "After-action reports, incident analysis, situation reports",
    "case_study": "Detailed analysis of specific events or implementations",
    "protocol": "Medical protocols, standard operating procedures",
    "checklist": "Quick-reference cards, action checklists",
    "plan": "Emergency plans, preparedness plans, action plans",
    "policy": "Government guidelines, regulations, standards",
    "assessment": "Risk assessments, vulnerability analyses",
    "news": "News articles, press releases, time-sensitive updates",
    "faq": "Frequently asked questions, Q&A format",
    "multimedia": "Video transcripts, podcast notes, infographic descriptions",
    "misc": "Low-content pages, stubs, index pages, or unclassifiable content",
}


@dataclass
class TypeScore:
    type: str
    confidence: float
    signals: list[str]  # For debugging - what triggered this score


class DocumentClassifier:
    """
    Rule-based document classifier with modular signal detectors.

    Usage:
        classifier = DocumentClassifier()
        results = classifier.classify(content, title, url)
        # Returns: [TypeScore(type='guide', confidence=0.72, signals=[...]), ...]
    """

    # Minimum content length to attempt classification (chars)
    MIN_CONTENT_LENGTH = 200

    # Minimum total score to avoid "misc" classification
    MIN_CONFIDENCE_THRESHOLD = 0.20

    def __init__(self):
        # Signal weights - tune these based on testing
        self.weights = {
            # Guide signals
            'numbered_steps': 0.20,
            'imperative_verbs': 0.12,
            'materials_list': 0.18,
            'how_to_title': 0.18,
            'tutorial_language': 0.12,
            'warning_caution': 0.08,
            'narrative_guide': 0.15,  # NEW: "I built", "here's how"
            'project_indicators': 0.12,  # NEW: photos, results, costs

            # Academic signals
            'abstract_section': 0.25,
            'citations': 0.20,
            'methodology': 0.18,
            'academic_title': 0.12,
            'formal_language': 0.10,
            'references_section': 0.15,

            # Article signals
            'article_structure': 0.15,
            'informational_language': 0.12,
            'wiki_patterns': 0.10,

            # Reference signals
            'data_tables': 0.20,
            'specifications': 0.18,
            'definition_patterns': 0.15,
            'list_heavy': 0.12,

            # Report signals
            'report_sections': 0.20,
            'incident_language': 0.18,
            'lessons_learned': 0.15,
            'timeline_dates': 0.12,

            # Protocol/Checklist signals
            'protocol_language': 0.20,
            'checklist_format': 0.22,
            'procedure_steps': 0.15,

            # Planning signals
            'plan_sections': 0.20,
            'policy_language': 0.18,
            'assessment_patterns': 0.15,

            # News/FAQ signals
            'news_patterns': 0.18,
            'faq_format': 0.25,

            # Low content penalties
            'very_short': -0.25,
            'stub_indicators': -0.20,
            'disambiguation': -0.35,
            'index_page': -0.30,
        }

    def classify(self, content: str, title: str = "", url: str = "") -> list[TypeScore]:
        """
        Classify a document and return top 3 types with confidence.
        """
        # Handle empty/minimal content
        if not content or len(content.strip()) < self.MIN_CONTENT_LENGTH:
            return [TypeScore(type='misc', confidence=0.90, signals=['content_too_short'])]

        # Normalize content for analysis
        content_lower = content.lower()
        title_lower = title.lower() if title else ""
        url_lower = url.lower() if url else ""

        # Run all detectors
        all_scores = {}
        all_signals = {}

        # Guide detection (guide, manual, field_guide)
        for doc_type, score, signals in self._detect_guide_signals(content, content_lower, title_lower, url_lower):
            if score > 0:
                all_scores[doc_type] = all_scores.get(doc_type, 0) + score
                all_signals.setdefault(doc_type, []).extend(signals)

        # Academic detection (academic, article, reference, training)
        for doc_type, score, signals in self._detect_academic_signals(content, content_lower, title_lower):
            if score > 0:
                all_scores[doc_type] = all_scores.get(doc_type, 0) + score
                all_signals.setdefault(doc_type, []).extend(signals)

        # Report detection (report, case_study, protocol, checklist)
        for doc_type, score, signals in self._detect_report_signals(content, content_lower, title_lower):
            if score > 0:
                all_scores[doc_type] = all_scores.get(doc_type, 0) + score
                all_signals.setdefault(doc_type, []).extend(signals)

        # Planning detection (plan, policy, assessment)
        for doc_type, score, signals in self._detect_planning_signals(content, content_lower, title_lower):
            if score > 0:
                all_scores[doc_type] = all_scores.get(doc_type, 0) + score
                all_signals.setdefault(doc_type, []).extend(signals)

        # Other detection (news, faq, multimedia)
        for doc_type, score, signals in self._detect_other_signals(content, content_lower, title_lower, url_lower):
            if score > 0:
                all_scores[doc_type] = all_scores.get(doc_type, 0) + score
                all_signals.setdefault(doc_type, []).extend(signals)

        # Check for misc/low-quality indicators
        misc_penalty, misc_signals = self._detect_misc_signals(content, content_lower, title_lower)

        # Apply misc penalty to all scores
        if misc_penalty < 0:
            for doc_type in all_scores:
                all_scores[doc_type] += misc_penalty
                all_signals[doc_type].extend(misc_signals)

        # If no strong signals, classify as misc
        if not all_scores or max(all_scores.values()) < self.MIN_CONFIDENCE_THRESHOLD:
            return [TypeScore(
                type='misc',
                confidence=0.80,
                signals=misc_signals if misc_signals else ['no_strong_signals']
            )]

        # Normalize scores to 0-1 range (cap at 1.0)
        normalized = {}
        for doc_type, score in all_scores.items():
            normalized[doc_type] = max(0.0, min(1.0, score))

        # Sort by score and take top 3
        sorted_types = sorted(normalized.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_type, confidence in sorted_types[:3]:
            if confidence > 0.05:
                results.append(TypeScore(
                    type=doc_type,
                    confidence=round(confidence, 2),
                    signals=all_signals.get(doc_type, [])
                ))

        if not results:
            return [TypeScore(type='misc', confidence=0.70, signals=['below_threshold'])]

        return results

    def _detect_guide_signals(self, content: str, content_lower: str, title_lower: str, url_lower: str):
        """
        Detect signals for: guide, manual, field_guide

        Returns list of (type, score, signals) tuples.
        """
        results = []
        guide_score = 0.0
        guide_signals = []
        manual_score = 0.0
        manual_signals = []
        field_guide_score = 0.0
        field_guide_signals = []

        # === GUIDE SIGNALS ===

        # --- Numbered steps ---
        step_patterns = [
            r'^\s*\d+\.\s+\w',
            r'^\s*step\s+\d+',
            r'^\s*\d+\)\s+\w',
        ]
        step_count = 0
        for pattern in step_patterns:
            matches = re.findall(pattern, content_lower, re.MULTILINE)
            step_count += len(matches)

        if step_count >= 3:
            guide_score += self.weights['numbered_steps']
            guide_signals.append(f'numbered_steps:{step_count}')
        elif step_count >= 1:
            guide_score += self.weights['numbered_steps'] * 0.4
            guide_signals.append(f'some_steps:{step_count}')

        # --- Imperative verbs ---
        imperative_verbs = [
            'cut', 'place', 'mix', 'add', 'remove', 'install', 'connect',
            'attach', 'secure', 'measure', 'pour', 'heat', 'cool', 'wait',
            'check', 'verify', 'ensure', 'make sure', 'be careful',
            'take', 'put', 'set', 'turn', 'open', 'close', 'press',
            'build', 'create', 'assemble', 'construct', 'prepare',
            'gather', 'collect', 'obtain', 'drill', 'screw', 'nail',
            'clean', 'wash', 'rinse', 'dry', 'wipe', 'apply', 'spread',
        ]

        imperative_count = 0
        for verb in imperative_verbs:
            pattern = r'(?:^|[.\n])\s*' + re.escape(verb) + r'\s'
            matches = re.findall(pattern, content_lower)
            imperative_count += len(matches)

        if imperative_count >= 5:
            guide_score += self.weights['imperative_verbs']
            guide_signals.append(f'imperative_verbs:{imperative_count}')
        elif imperative_count >= 2:
            guide_score += self.weights['imperative_verbs'] * 0.4
            guide_signals.append(f'some_imperatives:{imperative_count}')

        # --- Materials/ingredients list ---
        materials_patterns = [
            r'materials?\s*(?:needed|required|list)?:',
            r'ingredients?:',
            r'you(?:\'ll|\s+will)\s+need',
            r'tools?\s*(?:needed|required)?:',
            r'supplies:', r'equipment:', r'parts list', r'bill of materials',
        ]

        for pattern in materials_patterns:
            if re.search(pattern, content_lower):
                guide_score += self.weights['materials_list']
                guide_signals.append('materials_list')
                break

        # --- "How to" in title ---
        how_to_patterns = [
            r'how\s+to\s+', r'guide\s+to', r'tutorial', r'step.by.step',
            r'instructions', r'\bdiy\b', r'make\s+your\s+own', r'build\s+(?:a|your)',
            r'building\s+(?:a|the)', r'making\s+(?:a|the)', r'constructing',
        ]

        for pattern in how_to_patterns:
            if re.search(pattern, title_lower):
                guide_score += self.weights['how_to_title']
                guide_signals.append('how_to_title')
                break

        # --- Tutorial language ---
        tutorial_patterns = [
            r'in this (?:tutorial|guide|article)',
            r'(?:we|you) will (?:learn|show|demonstrate|build|make)',
            r'follow (?:these|the) steps',
            r'let\'s (?:get started|begin|start)',
            r'here\'s how', r'that\'s it!', r'you(?:\'re|\s+are) done',
        ]

        tutorial_count = sum(1 for p in tutorial_patterns if re.search(p, content_lower))
        if tutorial_count >= 2:
            guide_score += self.weights['tutorial_language']
            guide_signals.append(f'tutorial_language:{tutorial_count}')

        # --- Warning/caution ---
        warning_patterns = [
            r'(?:^|\n)\s*(?:warning|caution|danger|note|tip|important):?\s',
            r'be careful', r'safety (?:first|note|warning)',
            r'do not\s+\w+\s+(?:if|when|unless)', r'protective (?:gear|equipment|glasses|gloves)',
        ]

        for pattern in warning_patterns:
            if re.search(pattern, content_lower):
                guide_score += self.weights['warning_caution']
                guide_signals.append('warning_caution')
                break

        # --- NEW: Narrative guide patterns ("I built", "here's what I used") ---
        narrative_patterns = [
            r'\bi (?:built|made|constructed|created|installed|designed)',
            r'here(?:\'s| is) (?:what|how)',
            r'my (?:project|build|design|setup|installation)',
            r'(?:this|the) project',
            r'i decided to',
            r'i wanted to',
            r'the (?:result|results|outcome)',
            r'it (?:works|worked) (?:great|well|perfectly)',
            r'lessons? (?:i )?learned',
            r'what i would do differently',
            r'total (?:cost|time|materials)',
        ]

        narrative_count = sum(1 for p in narrative_patterns if re.search(p, content_lower))
        if narrative_count >= 3:
            guide_score += self.weights['narrative_guide']
            guide_signals.append(f'narrative_guide:{narrative_count}')
        elif narrative_count >= 1:
            guide_score += self.weights['narrative_guide'] * 0.4
            guide_signals.append(f'some_narrative:{narrative_count}')

        # --- NEW: Project indicators (photos, measurements, costs) ---
        project_patterns = [
            r'(?:photo|picture|image)s?\s*(?:below|above|show)',
            r'(?:here|see)\s+(?:the|a)\s+(?:photo|picture|image)',
            r'\$\d+', r'\d+\s*(?:dollars|usd)',  # Cost mentions
            r'\d+\s*(?:inches|feet|meters|cm|mm)',  # Measurements
            r'(?:performance|efficiency|output)\s*(?:is|was|of)',
            r'test(?:ing|ed)?\s+(?:the|this|my)',
        ]

        project_count = sum(1 for p in project_patterns if re.search(p, content_lower))
        if project_count >= 3:
            guide_score += self.weights['project_indicators']
            guide_signals.append(f'project_indicators:{project_count}')
        elif project_count >= 1:
            guide_score += self.weights['project_indicators'] * 0.4
            guide_signals.append(f'some_project:{project_count}')

        # === MANUAL SIGNALS ===
        manual_patterns = [
            r'(?:user|owner|operator)(?:\'s)?\s*(?:manual|guide)',
            r'installation\s+(?:manual|guide|instructions)',
            r'operating\s+instructions',
            r'technical\s+(?:manual|specifications)',
            r'maintenance\s+(?:manual|guide|instructions)',
            r'troubleshooting', r'warranty', r'model\s+(?:number|no\.?)',
        ]

        manual_count = sum(1 for p in manual_patterns if re.search(p, content_lower) or re.search(p, title_lower))
        if manual_count >= 2:
            manual_score += 0.40
            manual_signals.append(f'manual_patterns:{manual_count}')
        elif manual_count >= 1:
            manual_score += 0.20
            manual_signals.append(f'some_manual:{manual_count}')

        # === FIELD GUIDE SIGNALS ===
        field_guide_patterns = [
            r'field\s+guide', r'identification\s+(?:guide|key)',
            r'how\s+to\s+identify', r'identifying\s+',
            r'species', r'characteristics', r'distinguishing\s+features',
            r'edible', r'poisonous', r'medicinal\s+(?:uses|properties)',
            r'habitat', r'found\s+in', r'native\s+to',
        ]

        field_count = sum(1 for p in field_guide_patterns if re.search(p, content_lower) or re.search(p, title_lower))
        if field_count >= 3:
            field_guide_score += 0.45
            field_guide_signals.append(f'field_guide_patterns:{field_count}')
        elif field_count >= 1:
            field_guide_score += 0.20
            field_guide_signals.append(f'some_field:{field_count}')

        # Return all detected types
        if guide_score > 0:
            results.append(('guide', guide_score, guide_signals))
        if manual_score > 0:
            results.append(('manual', manual_score, manual_signals))
        if field_guide_score > 0:
            results.append(('field_guide', field_guide_score, field_guide_signals))

        return results

    def _detect_academic_signals(self, content: str, content_lower: str, title_lower: str):
        """
        Detect signals for: academic, article, reference, training
        """
        results = []
        academic_score = 0.0
        academic_signals = []
        article_score = 0.0
        article_signals = []
        reference_score = 0.0
        reference_signals = []
        training_score = 0.0
        training_signals = []

        # === ACADEMIC SIGNALS ===

        # Abstract section
        if re.search(r'(?:^|\n)\s*abstract\s*[\n:]', content_lower):
            academic_score += self.weights['abstract_section']
            academic_signals.append('abstract_section')

        # Citations
        numeric_citations = re.findall(r'\[\d+(?:[,\-]\d+)*\]', content)
        author_citations = re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)', content)
        total_citations = len(numeric_citations) + len(author_citations)

        if total_citations >= 5:
            academic_score += self.weights['citations']
            academic_signals.append(f'citations:{total_citations}')
        elif total_citations >= 2:
            academic_score += self.weights['citations'] * 0.4
            academic_signals.append(f'some_citations:{total_citations}')

        # Methodology section
        methodology_patterns = [
            r'(?:^|\n)\s*(?:methodology|methods?)\s*[\n:]',
            r'(?:^|\n)\s*(?:materials?\s+and\s+methods?)',
            r'(?:^|\n)\s*(?:experimental\s+(?:design|setup|procedure))',
            r'(?:^|\n)\s*(?:data\s+(?:collection|analysis))',
        ]
        if any(re.search(p, content_lower) for p in methodology_patterns):
            academic_score += self.weights['methodology']
            academic_signals.append('methodology_section')

        # Academic title
        academic_title_patterns = [
            r'^a\s+study\s+of', r'^analysis\s+of', r'^(?:an\s+)?investigation',
            r'^(?:an\s+)?evaluation', r'^effects?\s+of', r'^impact\s+of',
            r':\s*a\s+(?:systematic\s+)?review', r'et\s+al\.',
        ]
        if any(re.search(p, title_lower) for p in academic_title_patterns):
            academic_score += self.weights['academic_title']
            academic_signals.append('academic_title')

        # Formal language
        formal_patterns = [
            r'this (?:study|paper|research|article)',
            r'(?:we|the authors?) (?:found|conclude|argue|propose)',
            r'(?:findings?|results?) (?:suggest|indicate|show)',
            r'(?:statistically?\s+)?significant', r'p\s*[<>=]\s*0?\.\d+',
            r'hypothesis', r'correlation', r'sample size',
        ]
        formal_count = sum(1 for p in formal_patterns if re.search(p, content_lower))
        if formal_count >= 3:
            academic_score += self.weights['formal_language']
            academic_signals.append(f'formal_language:{formal_count}')

        # References section
        if re.search(r'(?:^|\n)\s*references?\s*[\n:]', content_lower):
            academic_score += self.weights['references_section']
            academic_signals.append('references_section')

        # === ARTICLE SIGNALS ===
        article_patterns = [
            r'(?:^|\n)\s*(?:introduction|overview|background)\s*[\n:]',
            r'(?:^|\n)\s*(?:conclusion|summary)\s*[\n:]',
            r'(?:in|for)\s+(?:this|the)\s+article',
            r'(?:we|this article)\s+(?:will|shall)\s+(?:discuss|explore|examine)',
        ]
        article_count = sum(1 for p in article_patterns if re.search(p, content_lower))
        if article_count >= 2:
            article_score += self.weights['article_structure']
            article_signals.append(f'article_structure:{article_count}')

        # Wiki-style patterns
        wiki_patterns = [
            r'^[A-Z][a-z]+\s+(?:is|are|was|were)\s+(?:a|an|the)',  # "Solar energy is a..."
            r'(?:also\s+)?(?:known|called|referred)\s+(?:as|to)',
            r'(?:types|kinds|forms)\s+of\s+',
            r'(?:history|origin|development)\s+of\s+',
        ]
        wiki_count = sum(1 for p in wiki_patterns if re.search(p, content_lower))
        if wiki_count >= 2:
            article_score += self.weights['wiki_patterns']
            article_signals.append(f'wiki_patterns:{wiki_count}')

        # === REFERENCE SIGNALS ===
        # Data tables
        table_indicators = content.count('|') + content.count('\t\t')
        if table_indicators >= 20:
            reference_score += self.weights['data_tables']
            reference_signals.append(f'data_tables:{table_indicators}')

        # Specifications
        spec_patterns = [
            r'specifications?:', r'dimensions?:', r'weight:',
            r'capacity:', r'voltage:', r'power:', r'rating:',
            r'\d+\s*(?:v|w|a|hz|rpm|psi|bar|kw|mw)',
        ]
        spec_count = sum(1 for p in spec_patterns if re.search(p, content_lower))
        if spec_count >= 3:
            reference_score += self.weights['specifications']
            reference_signals.append(f'specifications:{spec_count}')

        # Definition patterns
        def_patterns = [
            r'^[A-Z][a-z]+:\s+', r'definition:', r'meaning:',
            r'(?:is|are)\s+defined\s+as', r'refers\s+to',
        ]
        def_count = sum(1 for p in def_patterns if re.search(p, content_lower))
        if def_count >= 2:
            reference_score += self.weights['definition_patterns']
            reference_signals.append(f'definition_patterns:{def_count}')

        # === TRAINING SIGNALS ===
        training_patterns = [
            r'(?:learning|training)\s+(?:objectives?|outcomes?|goals?)',
            r'(?:by the end|after completing)',
            r'(?:module|lesson|unit|chapter)\s+\d+',
            r'(?:quiz|test|assessment|exercise)s?:',
            r'key\s+(?:concepts?|terms?|points?)',
            r'(?:review|practice)\s+questions?',
        ]
        training_count = sum(1 for p in training_patterns if re.search(p, content_lower))
        if training_count >= 2:
            training_score += 0.35
            training_signals.append(f'training_patterns:{training_count}')

        # Return results
        if academic_score > 0:
            results.append(('academic', academic_score, academic_signals))
        if article_score > 0:
            results.append(('article', article_score, article_signals))
        if reference_score > 0:
            results.append(('reference', reference_score, reference_signals))
        if training_score > 0:
            results.append(('training', training_score, training_signals))

        return results

    def _detect_report_signals(self, content: str, content_lower: str, title_lower: str):
        """
        Detect signals for: report, case_study, protocol, checklist
        """
        results = []
        report_score = 0.0
        report_signals = []
        case_study_score = 0.0
        case_study_signals = []
        protocol_score = 0.0
        protocol_signals = []
        checklist_score = 0.0
        checklist_signals = []

        # === REPORT SIGNALS ===
        report_section_patterns = [
            r'(?:^|\n)\s*(?:executive\s+summary)',
            r'(?:^|\n)\s*(?:findings|recommendations)',
            r'(?:^|\n)\s*(?:situation|incident)\s+(?:report|summary)',
            r'after.action\s+(?:report|review)',
            r'(?:initial|final|status)\s+report',
        ]
        report_count = sum(1 for p in report_section_patterns if re.search(p, content_lower) or re.search(p, title_lower))
        if report_count >= 1:
            report_score += self.weights['report_sections']
            report_signals.append(f'report_sections:{report_count}')

        # Incident language
        incident_patterns = [
            r'(?:the|this)\s+incident', r'(?:occurred|happened)\s+(?:on|at|in)',
            r'(?:response|responded)\s+to', r'(?:casualties|injuries|damage)',
            r'(?:deployed|dispatched)', r'(?:timeline|chronology)',
        ]
        incident_count = sum(1 for p in incident_patterns if re.search(p, content_lower))
        if incident_count >= 2:
            report_score += self.weights['incident_language']
            report_signals.append(f'incident_language:{incident_count}')

        # Lessons learned
        lessons_patterns = [
            r'lessons?\s+learned', r'what\s+(?:worked|didn\'t work)',
            r'(?:areas?\s+for\s+)?improvement', r'(?:best|good)\s+practices',
            r'recommendations?\s+for\s+(?:future|next)',
        ]
        if any(re.search(p, content_lower) for p in lessons_patterns):
            report_score += self.weights['lessons_learned']
            report_signals.append('lessons_learned')

        # Timeline/dates
        date_pattern = r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}'
        dates_found = len(re.findall(date_pattern, content_lower))
        if dates_found >= 3:
            report_score += self.weights['timeline_dates']
            report_signals.append(f'timeline_dates:{dates_found}')

        # === CASE STUDY SIGNALS ===
        case_study_patterns = [
            r'case\s+study', r'(?:this|the)\s+case',
            r'(?:background|context):', r'(?:challenge|problem):',
            r'(?:solution|approach|intervention):',
            r'(?:results?|outcomes?|impact):', r'(?:key\s+)?takeaways?',
        ]
        case_count = sum(1 for p in case_study_patterns if re.search(p, content_lower) or re.search(p, title_lower))
        if case_count >= 3:
            case_study_score += 0.40
            case_study_signals.append(f'case_study_patterns:{case_count}')
        elif case_count >= 1:
            case_study_score += 0.20
            case_study_signals.append(f'some_case_study:{case_count}')

        # === PROTOCOL SIGNALS ===
        protocol_patterns = [
            r'(?:standard\s+)?(?:operating\s+)?procedure',
            r'protocol', r'(?:treatment|clinical)\s+(?:guidelines?|protocol)',
            r'(?:do|perform|administer)\s+(?:the\s+following|as\s+follows)',
            r'(?:contraindications?|indications?|precautions?)',
            r'(?:dosage|dose|administration)',
        ]
        protocol_count = sum(1 for p in protocol_patterns if re.search(p, content_lower) or re.search(p, title_lower))
        if protocol_count >= 2:
            protocol_score += self.weights['protocol_language']
            protocol_signals.append(f'protocol_patterns:{protocol_count}')

        # === CHECKLIST SIGNALS ===
        # Look for checkbox patterns or bullet lists with action items
        checkbox_patterns = [
            r'(?:\[\s*\]|\[ \]|\[x\]|\[X\])',  # [ ] or [x]
            r'(?:^|\n)\s*[-*]\s+(?:check|verify|confirm|ensure)',
            r'checklist', r'(?:pre|post).(?:flight|trip|event|operation)',
        ]
        checkbox_count = sum(len(re.findall(p, content)) for p in checkbox_patterns)
        if checkbox_count >= 5:
            checklist_score += self.weights['checklist_format']
            checklist_signals.append(f'checklist_format:{checkbox_count}')
        elif checkbox_count >= 2:
            checklist_score += self.weights['checklist_format'] * 0.4
            checklist_signals.append(f'some_checklist:{checkbox_count}')

        # Return results
        if report_score > 0:
            results.append(('report', report_score, report_signals))
        if case_study_score > 0:
            results.append(('case_study', case_study_score, case_study_signals))
        if protocol_score > 0:
            results.append(('protocol', protocol_score, protocol_signals))
        if checklist_score > 0:
            results.append(('checklist', checklist_score, checklist_signals))

        return results

    def _detect_planning_signals(self, content: str, content_lower: str, title_lower: str):
        """
        Detect signals for: plan, policy, assessment
        """
        results = []
        plan_score = 0.0
        plan_signals = []
        policy_score = 0.0
        policy_signals = []
        assessment_score = 0.0
        assessment_signals = []

        # === PLAN SIGNALS ===
        plan_patterns = [
            r'(?:emergency|disaster|contingency|response)\s+plan',
            r'(?:action|implementation|strategic)\s+plan',
            r'(?:objectives?|goals?)\s*:', r'(?:timeline|schedule|milestones?)',
            r'(?:responsible\s+)?(?:party|parties|person)',
            r'(?:resources?\s+)?(?:needed|required|allocated)',
            r'(?:phase|stage)\s+\d+',
        ]
        plan_count = sum(1 for p in plan_patterns if re.search(p, content_lower) or re.search(p, title_lower))
        if plan_count >= 2:
            plan_score += self.weights['plan_sections']
            plan_signals.append(f'plan_patterns:{plan_count}')

        # === POLICY SIGNALS ===
        policy_patterns = [
            r'(?:policy|regulation|ordinance|statute|law)',
            r'(?:shall|must|is required to)',
            r'(?:effective\s+date|enacted|adopted)',
            r'(?:section|article|clause)\s+\d+',
            r'(?:compliance|enforcement|violation)',
            r'(?:authority|jurisdiction)',
        ]
        policy_count = sum(1 for p in policy_patterns if re.search(p, content_lower) or re.search(p, title_lower))
        if policy_count >= 3:
            policy_score += self.weights['policy_language']
            policy_signals.append(f'policy_patterns:{policy_count}')
        elif policy_count >= 1:
            policy_score += self.weights['policy_language'] * 0.4
            policy_signals.append(f'some_policy:{policy_count}')

        # === ASSESSMENT SIGNALS ===
        assessment_patterns = [
            r'(?:risk|vulnerability|threat|hazard)\s+assessment',
            r'(?:needs?\s+)?assessment', r'(?:probability|likelihood)',
            r'(?:impact|consequence|severity)\s+(?:level|rating|score)',
            r'(?:high|medium|low)\s+(?:risk|priority)',
            r'(?:mitigation|control)\s+(?:measures?|strategies?)',
            r'(?:swot|pestle|gap)\s+analysis',
        ]
        assessment_count = sum(1 for p in assessment_patterns if re.search(p, content_lower) or re.search(p, title_lower))
        if assessment_count >= 2:
            assessment_score += self.weights['assessment_patterns']
            assessment_signals.append(f'assessment_patterns:{assessment_count}')

        # Return results
        if plan_score > 0:
            results.append(('plan', plan_score, plan_signals))
        if policy_score > 0:
            results.append(('policy', policy_score, policy_signals))
        if assessment_score > 0:
            results.append(('assessment', assessment_score, assessment_signals))

        return results

    def _detect_other_signals(self, content: str, content_lower: str, title_lower: str, url_lower: str):
        """
        Detect signals for: news, faq, multimedia
        """
        results = []
        news_score = 0.0
        news_signals = []
        faq_score = 0.0
        faq_signals = []
        multimedia_score = 0.0
        multimedia_signals = []

        # === NEWS SIGNALS ===
        news_patterns = [
            r'(?:breaking|latest|update):', r'(?:press\s+)?release',
            r'(?:reported|announced|confirmed)\s+(?:today|yesterday|that)',
            r'according\s+to\s+(?:officials?|sources?|reports?)',
            r'(?:spokesperson|official)\s+said',
            r'(?:developing|unfolding)\s+(?:story|situation)',
        ]
        news_count = sum(1 for p in news_patterns if re.search(p, content_lower))
        if news_count >= 2:
            news_score += self.weights['news_patterns']
            news_signals.append(f'news_patterns:{news_count}')

        # Recent dates in news-style
        recent_date = r'\b(?:today|yesterday|this\s+(?:morning|afternoon|week))\b'
        if re.search(recent_date, content_lower):
            news_score += 0.10
            news_signals.append('recent_date_reference')

        # === FAQ SIGNALS ===
        faq_patterns = [
            r'(?:frequently\s+asked\s+questions?|faq)',
            r'(?:^|\n)\s*q:\s*', r'(?:^|\n)\s*a:\s*',
            r'(?:^|\n)\s*(?:question|answer)\s*\d*:',
            r'\?\s*\n+[A-Z]',  # Question followed by answer
        ]
        faq_count = sum(len(re.findall(p, content_lower)) for p in faq_patterns)
        if faq_count >= 3:
            faq_score += self.weights['faq_format']
            faq_signals.append(f'faq_format:{faq_count}')

        # FAQ in title/URL
        if re.search(r'faq|frequently.asked', title_lower) or re.search(r'faq', url_lower):
            faq_score += 0.20
            faq_signals.append('faq_in_title_url')

        # === MULTIMEDIA SIGNALS ===
        multimedia_patterns = [
            r'(?:video|watch|view)\s+(?:the|this|our)',
            r'(?:click|press)\s+play', r'(?:duration|length):\s*\d+',
            r'(?:transcript|captions?|subtitles?)',
            r'\[video\]', r'\[audio\]', r'(?:podcast|webinar|recording)',
        ]
        multi_count = sum(1 for p in multimedia_patterns if re.search(p, content_lower))
        if multi_count >= 2:
            multimedia_score += 0.35
            multimedia_signals.append(f'multimedia_patterns:{multi_count}')

        # Return results
        if news_score > 0:
            results.append(('news', news_score, news_signals))
        if faq_score > 0:
            results.append(('faq', faq_score, faq_signals))
        if multimedia_score > 0:
            results.append(('multimedia', multimedia_score, multimedia_signals))

        return results

    def _detect_misc_signals(self, content: str, content_lower: str, title_lower: str) -> tuple[float, list[str]]:
        """
        Detect signals indicating low-quality/misc content.
        Returns negative scores (penalties).
        """
        penalty = 0.0
        signals = []

        # Very short content
        content_length = len(content.strip())
        if content_length < 500:
            penalty += self.weights['very_short']
            signals.append(f'very_short:{content_length}')
        elif content_length < 1000:
            penalty += self.weights['very_short'] * 0.4
            signals.append(f'short:{content_length}')

        # Stub indicators
        stub_patterns = [
            r'this (?:article|page|section) is a stub',
            r'needs? (?:more |additional )?(?:content|information|expansion)',
            r'under construction', r'coming soon', r'placeholder',
            r'\{\{stub',
        ]
        if any(re.search(p, content_lower) for p in stub_patterns):
            penalty += self.weights['stub_indicators']
            signals.append('stub_indicator')

        # Disambiguation pages
        disambig_patterns = [
            r'disambiguation', r'may (?:also\s+)?refer to:',
            r'can (?:also\s+)?refer to:', r'^list of\s+',
        ]
        if any(re.search(p, content_lower) or re.search(p, title_lower) for p in disambig_patterns):
            penalty += self.weights['disambiguation']
            signals.append('disambiguation')

        # Index/navigation pages
        index_patterns = [
            r'^(?:index|contents?|sitemap|site map)$',
            r'^\s*(?:home|main)\s*(?:page)?\s*$',
            r'click here to', r'browse\s+(?:by|all)',
        ]
        if any(re.search(p, title_lower) for p in index_patterns):
            penalty += self.weights['index_page']
            signals.append('index_page')

        return penalty, signals

    def classify_batch(self, documents: list[dict]) -> list[dict]:
        """Classify multiple documents."""
        results = []
        for doc in documents:
            types = self.classify(
                content=doc.get('content', ''),
                title=doc.get('title', ''),
                url=doc.get('url', '')
            )
            results.append({
                'doc_id': doc.get('doc_id', doc.get('url', '')),
                'title': doc.get('title', ''),
                'types': [{'type': t.type, 'confidence': t.confidence, 'signals': t.signals} for t in types]
            })
        return results


class LLMClassifier:
    """
    LLM-based document classifier using llama-cpp-python.

    Usage:
        classifier = LLMClassifier()
        results = classifier.classify(content, title)
        # Returns: [TypeScore(type='article', confidence=0.85, signals=['llm']), ...]
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize LLM classifier.

        Args:
            model_path: Optional path to GGUF model file. If None, auto-detects from config.
        """
        self._runtime = None
        self._model_path = model_path
        self._load_error = None

    def _get_runtime(self):
        """Lazy-load the LlamaRuntime."""
        if self._runtime is None:
            try:
                from offline_tools.llama_runtime import get_llama_runtime
                self._runtime = get_llama_runtime(self._model_path)
            except ImportError as e:
                self._load_error = f"llama_runtime not available: {e}"
            except Exception as e:
                self._load_error = f"Failed to load runtime: {e}"
        return self._runtime

    def is_available(self) -> bool:
        """Check if the LLM runtime is available."""
        runtime = self._get_runtime()
        return runtime is not None and runtime.is_available()

    def classify(self, content: str, title: str = "", url: str = "") -> list[TypeScore]:
        """
        Classify a document using LLM.

        Args:
            content: Full text content (will be truncated)
            title: Document title
            url: Document URL

        Returns:
            List of TypeScore objects, sorted by confidence descending
        """
        runtime = self._get_runtime()
        if not runtime or not runtime.is_available():
            return [TypeScore(type='misc', confidence=0.5, signals=[f'llm_unavailable:{self._load_error or "no model"}'])]

        # Truncate content for LLM context
        content_preview = content[:2000] if content else ""

        # Build type descriptions for prompt
        type_list = "\n".join([f"- {k}: {v}" for k, v in DOCUMENT_TYPES.items()])

        # Build messages for chat-style prompt
        system_prompt = """You are a document classifier. Analyze document content and classify it into types.
Respond with ONLY valid JSON in this exact format (no other text):
{"types": [{"type": "type_name", "confidence": 0.X}, {"type": "type_name2", "confidence": 0.X}], "reasoning": "brief explanation"}"""

        user_prompt = f"""Classify this document into types.

Available document types:
{type_list}

Document title: {title}
Document URL: {url}

Content preview:
{content_preview}

Return the top 3 most likely types with confidence scores between 0.0 and 1.0.
If the content is too short or unclassifiable, use "misc" as the primary type."""

        try:
            # Use chat method for better instruction following
            response = runtime.chat(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                max_tokens=256,
                temperature=0.0  # Deterministic for classification
            )

            # Parse JSON from response
            return self._parse_llm_response(response)

        except Exception as e:
            return [TypeScore(type='misc', confidence=0.5, signals=[f'llm_error:{str(e)[:30]}'])]

    def _parse_llm_response(self, response: str) -> list[TypeScore]:
        """Parse LLM JSON response into TypeScore list."""
        try:
            # Try to extract JSON from response
            # LLM might include extra text, so find the JSON part
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                return [TypeScore(type='misc', confidence=0.5, signals=['llm_no_json'])]

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            types = data.get("types", [])
            reasoning = data.get("reasoning", "")

            results = []
            for t in types[:3]:  # Take top 3
                type_name = t.get("type", "misc")
                confidence = float(t.get("confidence", 0.5))

                # Validate type name
                if type_name not in DOCUMENT_TYPES:
                    type_name = "misc"

                # Clamp confidence to 0-1
                confidence = max(0.0, min(1.0, confidence))

                results.append(TypeScore(
                    type=type_name,
                    confidence=round(confidence, 2),
                    signals=[f'llm:{reasoning[:50]}'] if reasoning else ['llm']
                ))

            if not results:
                return [TypeScore(type='misc', confidence=0.5, signals=['llm_empty'])]

            return results

        except json.JSONDecodeError:
            return [TypeScore(type='misc', confidence=0.5, signals=['llm_json_error'])]
        except Exception as e:
            return [TypeScore(type='misc', confidence=0.5, signals=[f'llm_parse_error:{str(e)[:20]}'])]

    def classify_batch(self, documents: list[dict],
                       progress_callback=None) -> list[dict]:
        """
        Classify multiple documents with LLM.

        Args:
            documents: List of dicts with 'content', 'title', 'url' keys
            progress_callback: Optional callback(current, total)

        Returns:
            List of dicts with 'doc_id', 'title', 'types' keys
        """
        results = []
        total = len(documents)

        for i, doc in enumerate(documents):
            types = self.classify(
                content=doc.get('content', ''),
                title=doc.get('title', ''),
                url=doc.get('url', '')
            )
            results.append({
                'doc_id': doc.get('doc_id', doc.get('url', '')),
                'title': doc.get('title', ''),
                'types': [{'type': t.type, 'confidence': t.confidence, 'signals': t.signals} for t in types]
            })

            if progress_callback:
                progress_callback(i + 1, total)

        return results


class HybridClassifier:
    """
    Combines rule-based and LLM classification.

    Uses rules first for speed, falls back to LLM for uncertain cases.
    """

    def __init__(self, model_path: Optional[str] = None, llm_threshold: float = 0.5):
        """
        Args:
            model_path: Optional path to GGUF model file. If None, auto-detects from config.
            llm_threshold: If rule-based confidence is below this, use LLM
        """
        self.rule_classifier = DocumentClassifier()
        self.llm_classifier = LLMClassifier(model_path)
        self.llm_threshold = llm_threshold

    def classify(self, content: str, title: str = "", url: str = "") -> Tuple[list[TypeScore], str]:
        """
        Classify using hybrid approach.

        Returns:
            Tuple of (results, method) where method is 'rules' or 'llm'
        """
        # Try rule-based first
        rule_results = self.rule_classifier.classify(content, title, url)

        # Check confidence
        if rule_results and rule_results[0].confidence >= self.llm_threshold:
            return rule_results, 'rules'

        # Fall back to LLM if available
        if self.llm_classifier.is_available():
            llm_results = self.llm_classifier.classify(content, title, url)
            return llm_results, 'llm'

        # If LLM not available, return rule results anyway
        return rule_results, 'rules_fallback'


# --- Testing utilities ---

def test_classifier():
    """Quick test with sample documents."""
    classifier = DocumentClassifier()

    # Sample guide content
    guide_content = """
    How to Build a Solar Water Heater

    Materials needed:
    - Black pipe or tubing
    - Plywood backing

    Step 1: Prepare the backing
    Cut the plywood to size.

    Step 2: Install the tubing
    Attach the black tubing in a serpentine pattern.

    Warning: Be careful when handling glass.

    That's it! Your solar water heater is ready.
    """

    # Sample narrative guide (less structured)
    narrative_guide = """
    My Solar Collector Project

    I decided to build a solar air heater for my garage. Here's what I used
    and how it turned out.

    The total cost was about $150 for materials. I started with a 4x8 sheet
    of plywood and painted it black. The results were better than expected -
    on a sunny day it heats my 400 sq ft garage by about 15 degrees.

    Here's a photo of the finished collector. It works great even in winter.
    The lessons I learned: use thicker glazing and add more insulation.
    """

    # Sample academic content
    academic_content = """
    A Study of Solar Water Heating Efficiency

    Abstract
    This study examines the thermal efficiency of passive solar water heating
    systems. We analyzed data from 45 installations over 24 months.

    Methods
    Data collection involved temperature sensors. Statistical analysis was
    performed using ANOVA with p < 0.05 considered significant.

    Results
    The findings suggest that solar water heaters achieve 62% efficiency.
    Previous research by Smith et al. (2019) reported similar results.

    References
    [1] Smith, J. et al. (2019). Solar heating in subtropical climates.
    """

    # Sample FAQ
    faq_content = """
    Frequently Asked Questions

    Q: How much does a solar water heater cost?
    A: Typical systems range from $2,000 to $5,000 installed.

    Q: How long do they last?
    A: Most systems last 20-30 years with proper maintenance.

    Q: Do they work in winter?
    A: Yes, but efficiency is reduced on cloudy days.
    """

    print("=== Guide Test (Structured) ===")
    for r in classifier.classify(guide_content, "How to Build a Solar Water Heater"):
        print(f"  {r.type}: {r.confidence} - {r.signals}")

    print("\n=== Guide Test (Narrative) ===")
    for r in classifier.classify(narrative_guide, "My Solar Collector Project"):
        print(f"  {r.type}: {r.confidence} - {r.signals}")

    print("\n=== Academic Test ===")
    for r in classifier.classify(academic_content, "A Study of Solar Water Heating Efficiency"):
        print(f"  {r.type}: {r.confidence} - {r.signals}")

    print("\n=== FAQ Test ===")
    for r in classifier.classify(faq_content, "Solar Water Heater FAQ"):
        print(f"  {r.type}: {r.confidence} - {r.signals}")


if __name__ == "__main__":
    test_classifier()
