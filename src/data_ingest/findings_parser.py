import re
import pandas as pd
import json
import os

class UltrasoundNegationParser:
    """Custom parser for ultrasound findings with radiology-specific negation"""
    
    def __init__(self):
        # Pre-negation patterns (appear BEFORE the concept)
        self.pre_negation_patterns = [
            r'\bno\b',
            r'\bwithout\b',
            r'\babsent\b',
            r'\bdenies\b',
            r'\bnot\s+(?:seen|present|identified|appreciated|visible)\b',
            r'\bfree\s+of\b',
            r'\blacks?\b',
            r'\bnegative\s+for\b',
            r'\bno\s+evidence\s+of\b',
            r'\bno\s+longer\s+(?:seen|visible|present|identified)\b',
        ]
        
        # Post-negation patterns (appear AFTER the concept)
        self.post_negation_patterns = [
            r'\b(?:is|are|was|were)\s+no\s+longer\s+(?:seen|visible|present|identified|appreciated)\b',
            r'\b(?:is|are|was|were)\s+not\s+(?:seen|visible|present|identified|appreciated)\b',
            r'\b(?:has|have)\s+resolved\b',
            r'\b(?:is|are)\s+absent\b',
            r'\b(?:was|were)\s+ruled\s+out\b',
        ]
        
        # Terminators that end negation scope
        self.terminators = [
            r'\bbut\b',
            r'\bhowever\b',
            r'\balthough\b',
            r'\bshowing\b',
            r'\bdemonstrat(?:es|ing)\b',
            r'\bpresent\b',  # unless it's "not present"
        ]
        
        # Feature patterns
        self.feature_patterns = {
            'margin': [
                (r'\bcircumscribed\b', 'circumscribed'),
                (r'\b(?:macro|macrolobulated)\b', 'macrolobulated'),
                (r'\b(?:micro|microlobulated)\b', 'microlobulated'),
                (r'\bindistinct\b', 'indistinct'),
                (r'\bangular\b', 'angular'),
                (r'\bspiculated\b', 'spiculated'),
                (r'\birregular\s+margins?\b', 'irregular'),
            ],
            'shape': [
                (r'\boval\b', 'oval'),
                (r'\bround\b', 'round'),
                (r'\birregular\s+(?:shape|mass|lesion)\b', 'irregular'),
            ],
            'orientation': [
                (r'\bparallel\b', 'parallel'),
                (r'\bnot\s+parallel\b', 'not parallel'),
                (r'\bantiparallel\b', 'not parallel'),
            ],
            'echo': [
                (r'\banechoic\b', 'anechoic'),
                (r'\bhypoechoic\b', 'hypoechoic'),
                (r'\bisoechoic\b', 'isoechoic'),
                (r'\bhyperechoic\b', 'hyperechoic'),
                (r'\bcomplex\s+(?:echo|echogenicity)\b', 'complex'),
                (r'\bheterogeneous\b', 'heterogeneous'),
            ],
            'posterior': [
                (r'\b(?:posterior\s+)?enhancement\b', 'enhancement'),
                (r'\b(?:posterior|acoustic)\s+shadowing\b', 'shadowing'),
                (r'\bno\s+posterior\s+(?:acoustic\s+)?features?\b', 'no posterior features'),
            ],
            'boundary': [
                (r'\babrupt\s+interface\b', 'abrupt interface'),
                (r'\bechogenic\s+halo\b', 'echogenic halo'),
                (r'\barchitectural\s+distortion\b', 'architectural distortion'),
            ],
        }
    
    def normalize_text(self, text):
        """
        Normalize the text by extracting only the ultrasound section if it exists
        and if "MAMMO" appears before "ULTRASOUND:".
        Returns the normalized text and normalization info for the audit.
        """
        if pd.isna(text):
            return None, {
                'original_length': 0,
                'normalized_length': 0,
                'mammo_found': False,
                'ultrasound_section_found': False,
                'normalization_applied': False
            }
        
        # Store original text length for the audit
        original_length = len(text)
        
        # Convert to uppercase temporarily for the section search
        text_upper = text.upper()
        
        # Look for ULTRASOUND: section
        ultrasound_match = re.search(r'ULTRASOUND\s*:', text_upper)
        
        normalization_info = {
            'original_length': original_length,
            'mammo_found': False,
            'ultrasound_section_found': False,
            'normalization_applied': False
        }
        
        # Check if we should apply section extraction
        should_extract_section = False
        
        if ultrasound_match:
            normalization_info['ultrasound_section_found'] = True
            
            # Check if "MAMMO" appears before "ULTRASOUND:"
            section_before_ultrasound = text_upper[:ultrasound_match.start()]
            mammo_match = re.search(r'\bMAMMO', section_before_ultrasound)
            
            if mammo_match:
                normalization_info['mammo_found'] = True
                normalization_info['mammo_position'] = mammo_match.start()
                should_extract_section = True
        
        # Only extract the ULTRASOUND section if we found MAMMO before it
        if should_extract_section:
            # Extract text after ULTRASOUND:
            section_start = ultrasound_match.end()
            
            # Find the next section heading (if any)
            next_section_match = re.search(r'\b[A-Z]{3,}(?:\s+[A-Z]+)*\s*:', text_upper[section_start:])
            
            if next_section_match:
                # Extract only the ULTRASOUND section
                section_end = section_start + next_section_match.start()
                normalized_text = text[section_start:section_end].strip()
            else:
                # No next section, take all text after ULTRASOUND:
                normalized_text = text[section_start:].strip()
            
            normalization_info['normalization_applied'] = True
            normalization_info['section_start_pos'] = section_start
            normalization_info['normalized_length'] = len(normalized_text)
        else:
            # No ULTRASOUND: section found or no MAMMO before it, use the entire text
            normalized_text = text
            normalization_info['normalized_length'] = original_length
        
        return normalized_text, normalization_info
    
    def is_in_axillary_context(self, text, match_start, match_end):
        """Check if match is related to axillary nodes"""
        window = 100
        context_start = max(0, match_start - window)
        context_end = min(len(text), match_end + window)
        context = text[context_start:context_end].lower()
        
        axillary_indicators = [
            'axillary', 'axilla', 'lymph node', 'lymph nodes',
        ]
        
        return any(indicator in context for indicator in axillary_indicators)
    
    def is_negated(self, text, match_start, match_end):
        """Check if a feature at given position is negated"""
        text_lower = text.lower()
        
        # Look back up to 150 characters
        lookback_text = text_lower[max(0, match_start - 150):match_start]
        
        # Look ahead up to 150 characters (for post-negation)
        lookahead_text = text_lower[match_end:min(len(text_lower), match_end + 150)]
        
        # Check for terminators in lookback (which end negation scope)
        for terminator in self.terminators:
            term_match = re.search(terminator, lookback_text)
            if term_match:
                # Only consider text after the terminator
                lookback_text = lookback_text[term_match.end():]
        
        # Check for PRE-negation patterns (before the match)
        for negation in self.pre_negation_patterns:
            if re.search(negation, lookback_text):
                return True
        
        # Check for POST-negation patterns (after the match)
        for negation in self.post_negation_patterns:
            if re.search(negation, lookahead_text):
                return True
        
        return False
    
    def parse_findings(self, findings_text):
        """Parse findings text and return both feature values and JSON audit"""
        if pd.isna(findings_text):
            return {
                'features': {f: None for f in self.feature_patterns.keys()},
                'audit': {
                    'original_text': None,
                    'normalization': None,
                    'decisions': []
                }
            }
        
        # Normalize text - extract ultrasound section if exists and MAMMO is before it
        normalized_text, normalization_info = self.normalize_text(findings_text)
        
        if normalized_text is None:
            return {
                'features': {f: None for f in self.feature_patterns.keys()},
                'audit': {
                    'original_text': findings_text,
                    'normalization': normalization_info,
                    'decisions': []
                }
            }
        
        text_lower = normalized_text.lower()
        decisions = []
        feature_values = {f: [] for f in self.feature_patterns.keys()}
        
        for feature_type, patterns in self.feature_patterns.items():
            for pattern, value_name in patterns:
                for match in re.finditer(pattern, text_lower):
                    # Skip if in axillary node context
                    if self.is_in_axillary_context(text_lower, match.start(), match.end()):
                        continue
                    
                    is_neg = self.is_negated(text_lower, match.start(), match.end())
                    
                    # Get some context around the match
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text_lower), match.end() + 50)
                    context_text = text_lower[context_start:context_end]
                    
                    decision = {
                        'feature_type': feature_type,
                        'value': value_name,
                        'negated': is_neg,
                        'position': match.start(),
                        'match_text': match.group(0),
                        'context': context_text
                    }
                    
                    decisions.append(decision)
                    
                    # Only add to feature values if NOT negated
                    if not is_neg:
                        feature_values[feature_type].append(value_name)
        
        # Sort decisions by position for readability
        decisions.sort(key=lambda x: x['position'])
        
        # Convert feature lists to comma-separated strings (only if values exist)
        features = {}
        for feature_type, values in feature_values.items():
            if values:
                features[feature_type] = ', '.join(values)
            else:
                features[feature_type] = None
        
        return {
            'features': features,
            'audit': {
                'original_text': findings_text,
                'normalized_text': normalized_text,
                'normalization': normalization_info,
                'decisions': decisions
            }
        }

def add_ultrasound_classifications(radiology_df, output_path):
    """Add classifications using custom parser with audit saved as separate JSON file"""
    parser = UltrasoundNegationParser()
    
    # Filter for ultrasound modality and RIGHT or LEFT laterality
    mask = (radiology_df['MODALITY'] == 'US') & (radiology_df['Study_Laterality'].isin(['RIGHT', 'LEFT']))
    filtered_df = radiology_df[mask].copy()
    
    print(f"Processing {len(filtered_df)} records with custom parser...")
    
    # Ensure all feature columns exist
    for feature_type in parser.feature_patterns.keys():
        if feature_type not in radiology_df.columns:
            radiology_df[feature_type] = None
    
    # Create audit dictionary with record ID as key
    audit_data = {}
    
    # Parse each row
    for idx, row in filtered_df.iterrows():
        findings_text = row.get('FINDINGS', '')
        result = parser.parse_findings(findings_text)
        
        # Update feature columns with non-negated values only
        for feature_type, value in result['features'].items():
            radiology_df.loc[idx, feature_type] = value
        
        # Add to audit data - use a unique identifier from the row
        # Assuming there's a column like 'Study_ID' or similar
        record_id = row.get('Study_ID', str(idx))
        audit_data[record_id] = result['audit']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    audit_output_path = f'{output_path}/ultrasound_parsing_audit.json'
    # Save audit data to JSON file
    with open(audit_output_path, 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, indent=2)
    
    print(f"Parsing audit saved to {os.path.abspath(audit_output_path)}")
    
    return radiology_df