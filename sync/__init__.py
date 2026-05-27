"""Supabase sync: push local tracking data to the public BiffBet web dashboard.

This package is the one-way bridge from the engine's local SQLite store to the
Supabase Postgres tables the Next.js site reads. It contains NO model logic — it
only serializes what `tracking/*` already produced and upserts it.
"""
