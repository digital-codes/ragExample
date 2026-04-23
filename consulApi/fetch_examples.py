"""
fetch_examples.py – Demonstrates individual and bulk data retrieval from
the Consul Democracy GraphQL API using consul_api.py.

Run:
    python fetch_examples.py
"""

import json
import sys

from consul_api import (
    API_ENDPOINT,
    fetch_all,
    fetch_budget_by_id,
    fetch_budgets,
    fetch_comments,
    fetch_debate_by_id,
    fetch_debates,
    fetch_geozones,
    fetch_geozones_and_tags_bulk,
    fetch_milestones,
    fetch_proposal_by_id,
    fetch_proposal_notifications,
    fetch_proposals,
    fetch_tags,
    fetch_users,
    fetch_vote_by_id,
    fetch_votes,
)


def pp(label: str, data: object) -> None:
    """Pretty-print labelled JSON to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print("=" * 60)
    print(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 1. Individual record lookups
# ---------------------------------------------------------------------------

def demo_individual_lookups() -> None:
    print("\n>>> Fetching individual records by ID <<<", file=sys.stderr)

    proposal = fetch_proposal_by_id(1)
    pp("Proposal id=1 (with tags & votes)", proposal)

    vote = fetch_vote_by_id(1)
    pp("Vote id=1", vote)

    debate = fetch_debate_by_id(1)
    pp("Debate id=1 (with tags & votes)", debate)


# ---------------------------------------------------------------------------
# 2. Full paginated collection fetches (most important: proposals & votes)
# ---------------------------------------------------------------------------

def demo_paginated_collections() -> None:
    print("\n>>> Fetching full paginated collections <<<", file=sys.stderr)

    proposals = fetch_proposals()
    pp(f"All proposals ({len(proposals)} records)", proposals[:3])  # preview first 3

    votes = fetch_votes()
    pp(f"All votes ({len(votes)} records)", votes[:3])

    debates = fetch_debates()
    pp(f"All debates ({len(debates)} records)", debates[:3])

    comments = fetch_comments()
    pp(f"All comments ({len(comments)} records)", comments[:3])

    users = fetch_users()
    pp(f"All users ({len(users)} records)", users[:3])

    budgets = fetch_budgets()
    pp(f"All budgets ({len(budgets)} records)", budgets)

    geozones = fetch_geozones()
    pp(f"All geozones ({len(geozones)} records)", geozones)

    tags = fetch_tags()
    pp(f"All tags ({len(tags)} records)", tags[:10])

    milestones = fetch_milestones()
    pp(f"All milestones ({len(milestones)} records)", milestones[:3])

    notifications = fetch_proposal_notifications()
    pp(
        f"All proposal notifications ({len(notifications)} records)",
        notifications[:3],
    )


# ---------------------------------------------------------------------------
# 3. Bulk multi-resource query (two collections in a single HTTP request)
# ---------------------------------------------------------------------------

def demo_bulk_query() -> None:
    print(
        "\n>>> Fetching two collections in one HTTP request (bulk demo) <<<",
        file=sys.stderr,
    )
    result = fetch_geozones_and_tags_bulk()
    pp(
        f"Geozones ({len(result['geozones'])}) + Tags ({len(result['tags'])}) "
        "– single request",
        result,
    )


# ---------------------------------------------------------------------------
# 4. Fetch ALL resources in sequence
# ---------------------------------------------------------------------------

def demo_fetch_all() -> None:
    print("\n>>> Fetching ALL resources <<<", file=sys.stderr)
    all_data = fetch_all()
    summary = {k: len(v) for k, v in all_data.items()}
    pp("Summary – record counts per resource", summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Target endpoint: {API_ENDPOINT}\n")

    # Uncomment the demos you want to run:

    demo_individual_lookups()
    demo_paginated_collections()
    demo_bulk_query()
    # demo_fetch_all()   # fetches everything – may take a while
