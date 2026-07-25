import re

PERSIAN_DIGITS='۰۱۲۳۴۵۶۷۸۹'
ARABIC_DIGITS='٠١٢٣٤٥٦٧٨٩'
DIGIT_TRANS=str.maketrans(PERSIAN_DIGITS+ARABIC_DIGITS,'0123456789'*2)
LETTER_MAP={'ب':'B','ج':'J','د':'D','س':'S','ص':'S','ط':'T','ق':'Q','ل':'L','م':'M','ن':'N','و':'V','ه':'H','ی':'Y','ت':'T','ع':'E','پ':'P','الف':'A'}

def normalize_plate(text):
    text=(text or '').translate(DIGIT_TRANS).upper()
    for k,v in LETTER_MAP.items(): text=text.replace(k,v)
    return re.sub(r'[^0-9A-Z]','',text)

def plausible_plate(text):
    t=normalize_plate(text)
    digits=sum(ch.isdigit() for ch in t)
    return 6 <= len(t) <= 10 and digits >= 5

def format_iran_plate(text):
    t=normalize_plate(text)
    # Keep stable machine-readable text; add separators for common 2+1+3+2 layout.
    if len(t)==8:
        return f'{t[:2]}-{t[2]}-{t[3:6]}-{t[6:]}'
    return t
