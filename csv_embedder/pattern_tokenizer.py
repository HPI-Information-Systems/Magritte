""" This script defines the PatternTokenizer, which splits a string into tokens based on:
     - Special Characters
     - Patterns of characters
"""
import pdb

import regex
from typing import List


def upper_replacement(match, shrink=False):
    size = len(match.group())
    if size>1 and shrink:
        return "L*"
    elif size:
        return "L" * size
    else:
        return ""


def lower_replacement(match, shrink = False):
    size = len(match.group())
    if size>1 and shrink:
        return "l*"
    elif size:
        return "l" * size
    else:
        return ""


def digit_replacement(match, shrink = False):
    size = len(match.group())
    if size>1 and shrink:
        return "d*"
    elif size:
        return "d" * size
    else:
        return ""

def text_replacement(match, shrink=False):
    size = len(match.group())
    if size>1 and shrink:
        return "T*"
    elif size:
        return "T" * size
    else:
        return ""

def symbol_replacement(match, shrink=False):
    size = len(match.group())
    if size>1 and shrink:
        return "S*"
    elif size:
        return "s" * size
    else:
        return ""


def alphanumeric_replacement(match, shrink = False):
    size = len(match.group())
    if size>1 and shrink:
        return "A*"
    elif size:
        return "A" * size
    else:
        return ""

class PatternTokenizer():
    def __init__(self, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]", **kwargs):

        self.unk_token=unk_token
        self.sep_token=sep_token
        self.pad_token=pad_token
        self.cls_token=cls_token
        self.mask_token=mask_token


    def __call__(self, text: str, return_text = False, verbose=False)->list[str]:
        """
        This method only handles a single sentence (or sequence) of text.
        """
        # text = regex.sub(r'\s', '', text)
        # We use the [^\W\d_] pattern as a trick to match unicode letters
        # The special characters (e.g. whitespace) are tokenized individually even if a sequence appears
        # The mask token is tokenized individually if it appears
        matches = [m for m in regex.finditer(fr"{regex.escape(self.mask_token)}|" #the mask tokens OR
                                                    r"(\p{L}|\p{M}|\p{N})+|"
                                                    # r"(\p{N})+|"
                                                    # r"[^\W_]+|" #any character which is not a NOT alphanumeric character nor underscore
                                                    # r"\d+|"
                                                    r"(\S)|"
                                                    r"(\s)|"
                                                    r"(\r\n|\r|\n)", text)]

        if verbose: print(text)
        tokens = []
        for m in matches:
            token_str = m.group()
            if token_str != self.mask_token:
                token_str = regex.sub(r'(\p{Lu}\p{M}|\p{Lt}\p{M}|\p{Lu}|\p{Lt})+', upper_replacement, token_str)  # upper OR titlecase
                token_str = regex.sub(r'(\p{Ll}\p{M}|\p{Lm}\p{M}|\p{Lo}\p{M}|\p{Ll}|\p{Lm}|\p{Lo}|\p{M})+', lower_replacement, token_str) #order is important
                token_str = regex.sub(r'(\p{N})+', digit_replacement, token_str)
                token_str = regex.sub(r'((l+L+)|(L+l+))+(l*)(L*)', text_replacement, token_str) #one or more times you have L+l+ or l+L+ and optionally can end with lowercase or uppercase
                token_str = regex.sub(r'((d+T+)|(d+L+)|(d+l+)|(l+d+)|(L+d+)|(T+d+))+(d*)(L*)(l*)(T*)', alphanumeric_replacement, token_str)
                token_str = regex.sub(r'[\U00002200-\U0001FFFF]', symbol_replacement, token_str) # symbols like control symbols, emojis, pictograms
                try:
                    assert len(set(token_str))==1
                except AssertionError:
                    print("Line:", m.group())
                    print("Tokenized:",token_str)
                    raise NotImplementedError
                if len(token_str)>1:
                    token_str = token_str[0]+"*"
            if verbose: print(f"{token_str}", end="")
            if return_text:
                tokens+= [token_str]
            else:
                tokens +=  [token_str]

        if verbose: print("")
        return tokens