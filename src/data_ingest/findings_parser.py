import re
import pandas as pd

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
        """Parse findings text and extract all features with negation status"""
        if pd.isna(findings_text):
            return {f: None for f in self.feature_patterns.keys()}
        
        text_lower = findings_text.lower()
        results = {}
        
        for feature_type, patterns in self.feature_patterns.items():
            found_values = []
            
            for pattern, value_name in patterns:
                for match in re.finditer(pattern, text_lower):
                    # Skip if in axillary node context
                    if self.is_in_axillary_context(text_lower, match.start(), match.end()):
                        continue
                    
                    # NOW PASSING match.end() as well for lookahead
                    is_neg = self.is_negated(text_lower, match.start(), match.end())
                    found_values.append({
                        'value': value_name,
                        'negated': is_neg,
                        'position': match.start()
                    })
            
            # Keep only non-negated values, or negated if no non-negated found
            if found_values:
                non_negated = [f for f in found_values if not f['negated']]
                if non_negated:
                    results[feature_type] = ', '.join([f['value'] for f in non_negated])
                    results[f'{feature_type}_negated'] = False
                else:
                    results[feature_type] = ', '.join([f['value'] for f in found_values])
                    results[f'{feature_type}_negated'] = True
            else:
                results[feature_type] = None
                results[f'{feature_type}_negated'] = None
        
        return results

def add_ultrasound_classifications(radiology_df):
    """Add classifications using custom parser"""
    parser = UltrasoundNegationParser()
    
    # Filter for RIGHT or LEFT laterality
    mask = radiology_df['Study_Laterality'].isin(['RIGHT', 'LEFT'])
    filtered_df = radiology_df[mask].copy()
    
    print(f"Processing {len(filtered_df)} records with custom parser...")
    
    # Parse each row
    parsed_results = []
    for idx, row in filtered_df.iterrows():
        findings_text = row.get('FINDINGS', '')
        result = parser.parse_findings(findings_text)
        result['original_index'] = idx
        parsed_results.append(result)
    
    # Merge back
    parsed_df = pd.DataFrame(parsed_results)
    for col in parsed_df.columns:
        if col != 'original_index':
            radiology_df.loc[parsed_df['original_index'], col] = parsed_df[col].values
    
    return radiology_df