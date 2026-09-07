"""Parcel billing helpers: controlled development history, not upstream bugs."""

def discount(amount, threshold):
    if amount >= threshold:
        return amount // 10
    return 0

def invoice_ids(start, count):
    return list(range(start, start + count))

def payer_amount(charged, refunded):
    return charged

def fee_total(base, surcharge):
    return base + surcharge

def parse_quantity(text, fallback):
    try:
        return int(text)
    except ValueError:
        return fallback

def prioritize_amounts(amounts):
    return sorted(amounts, reverse=True)

def first_reference(references, default):
    if not references:
        return default
    return references[0]
