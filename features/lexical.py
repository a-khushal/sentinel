import math
from typing import Dict, List
from collections import Counter
import re

class LexicalFeatures:
    VOWELS = set('aeiouAEIOU')
    CONSONANTS = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
    
    def __init__(self):
        self.ngram_cache = {}
    
    def extract(self, domain: str) -> Dict[str, float]:
        domain = domain.lower().rstrip('.')
        parts = domain.split('.')
        
        if len(parts) > 1:
            sld = parts[-2]
        else:
            sld = parts[0]
        
        return {
            'length': len(domain),
            'sld_length': len(sld),
            'num_labels': len(parts),
            'max_label_length': max(len(p) for p in parts) if parts else 0,
            'entropy': self._shannon_entropy(sld),
            'char_entropy': self._shannon_entropy(domain),
            'digit_ratio': self._digit_ratio(sld),
            'vowel_ratio': self._vowel_ratio(sld),
            'consonant_ratio': self._consonant_ratio(sld),
            'special_ratio': self._special_ratio(domain),
            'has_digits': 1.0 if any(c.isdigit() for c in sld) else 0.0,
            'digit_count': sum(1 for c in sld if c.isdigit()),
            'unique_chars': len(set(sld)),
            'repeated_chars': self._repeated_chars_ratio(sld),
            'bigram_entropy': self._ngram_entropy(sld, 2),
            'trigram_entropy': self._ngram_entropy(sld, 3),
            'consonant_sequence': self._max_consonant_sequence(sld),
            'hex_ratio': self._hex_ratio(sld),
            'numeric_sequence': self._max_numeric_sequence(sld),
            'underscore_count': domain.count('_'),
            'hyphen_count': domain.count('-'),
        }
    
    def _shannon_entropy(self, s: str) -> float:
        if not s:
            return 0.0
        
        freq = Counter(s)
        length = len(s)
        entropy = 0.0
        
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)
        
        return entropy
    
    def _ngram_entropy(self, s: str, n: int) -> float:
        if len(s) < n:
            return 0.0
        
        ngrams = [s[i:i+n] for i in range(len(s) - n + 1)]
        freq = Counter(ngrams)
        length = len(ngrams)
        entropy = 0.0
        
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)
        
        return entropy
    
    def _digit_ratio(self, s: str) -> float:
        if not s:
            return 0.0
        return sum(1 for c in s if c.isdigit()) / len(s)
    
    def _vowel_ratio(self, s: str) -> float:
        if not s:
            return 0.0
        return sum(1 for c in s if c in self.VOWELS) / len(s)
    
    def _consonant_ratio(self, s: str) -> float:
        if not s:
            return 0.0
        return sum(1 for c in s if c in self.CONSONANTS) / len(s)
    
    def _special_ratio(self, s: str) -> float:
        if not s:
            return 0.0
        return sum(1 for c in s if not c.isalnum() and c != '.') / len(s)
    
    def _repeated_chars_ratio(self, s: str) -> float:
        if len(s) < 2:
            return 0.0
        repeated = sum(1 for i in range(1, len(s)) if s[i] == s[i-1])
        return repeated / (len(s) - 1)
    
    def _max_consonant_sequence(self, s: str) -> int:
        max_seq = 0
        current = 0
        for c in s.lower():
            if c in self.CONSONANTS:
                current += 1
                max_seq = max(max_seq, current)
            else:
                current = 0
        return max_seq
    
    def _hex_ratio(self, s: str) -> float:
        if not s:
            return 0.0
        hex_chars = set('0123456789abcdefABCDEF')
        return sum(1 for c in s if c in hex_chars) / len(s)
    
    def _max_numeric_sequence(self, s: str) -> int:
        max_seq = 0
        current = 0
        for c in s:
            if c.isdigit():
                current += 1
                max_seq = max(max_seq, current)
            else:
                current = 0
        return max_seq
    
    def extract_batch(self, domains: List[str]) -> List[Dict[str, float]]:
        return [self.extract(d) for d in domains]

