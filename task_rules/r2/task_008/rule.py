"""Frozen R2 method adapter; numerical behavior is unchanged."""
from praxis.rules import R2Rule

def load(directory):
    return R2Rule(directory)
