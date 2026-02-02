---
created: 2026-02-02T00:00
title: Update CLAUDE.md with new modules
area: docs
files:
  - CLAUDE.md
  - src/clustering.py
  - src/traffic_api.py
  - src/config.py
  - quantum-traffic-ui/src/components/Dashboard/AlgorithmComparison.tsx
---

## Problem

CLAUDE.md is missing documentation for recently added modules and configuration:

1. **clustering.py** - Hierarchical K-means clustering for 200+ delivery points with auto-computed optimal K via silhouette score
2. **traffic_api.py** - Real-time traffic API integration (TomTom/HERE) with caching and fallback to simulation
3. **AlgorithmComparison.tsx** - Frontend component for comparing QAOA, Greedy, Simulated Annealing, and Brute Force algorithms

New config variables not documented:
- `CLUSTER_THRESHOLD`, `MAX_CLUSTER_SIZE` (clustering settings)
- `TRAFFIC_API_ENABLED`, `TRAFFIC_API_PROVIDER`, `TOMTOM_API_KEY`, `HERE_API_KEY` (traffic API settings)

Frontend state also needs update for `compareRoutes()`, `comparisonResult`, `isComparing` in routeStore.

## Solution

Update CLAUDE.md sections:
1. Add clustering.py and traffic_api.py to "Core Modules"
2. Add new env vars to "Key Configuration" table
3. Add `/compare-algorithms` endpoint to "API Request Format"
4. Update "Frontend State" to include comparison state
