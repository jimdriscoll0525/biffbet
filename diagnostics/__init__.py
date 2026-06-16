"""Diagnostic / provenance layer — purely observe-and-label.

Classifies WHY each model component reports unavailable (legitimate absence vs
real fetch failure) and persists the reason as an additive block. Never changes
any probability, component delta, blend, EV, or pick for either model.
"""
