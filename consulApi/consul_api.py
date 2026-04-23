"""
Consul Democracy GraphQL API client.

Retrieves all publicly available data from a Consul Democracy instance using
its GraphQL endpoint.  Supports both individual record lookups (by ID) and
full paginated collection fetches.

Usage
-----
    python consul_api.py                     # fetch all collections to JSON
    python consul_api.py --resource proposals
    python consul_api.py --resource votes
    python consul_api.py --resource debates
    python consul_api.py --resource comments
    python consul_api.py --resource users
    python consul_api.py --resource budgets
    python consul_api.py --resource geozones
    python consul_api.py --resource tags
    python consul_api.py --resource milestones
    python consul_api.py --resource proposal_notifications
    python consul_api.py --id 1 --resource proposals

Dependencies: requests  (pip install requests)
Optionally:   gql[requests] (pip install gql[requests])
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, Generator, List, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_ENDPOINT = "https://consul.ok-lab-karlsruhe.de/graphql"
PAGE_SIZE = 25          # max allowed by the server
MAX_RETRIES = 3
RETRY_DELAY = 2         # seconds between retries

# Some servers reject requests without a recognisable User-Agent header.
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) "
        "Gecko/20100101 Firefox/116.0"
    ),
}

# ---------------------------------------------------------------------------
# GraphQL query definitions
# ---------------------------------------------------------------------------

# --- Proposals (user-contributed – highest priority) -----------------------

PROPOSALS_QUERY = """
{
  proposals(first: %(page_size)s%(after)s) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        title
        summary
        description
        public_created_at
        cached_votes_up
        comments_count
        confidence_score
        hot_score
        geozone_id
        video_url
        retired_reason
        retired_explanation
        retired_at
        public_author { id username }
        geozone { id name }
      }
    }
  }
}
"""

PROPOSAL_BY_ID_QUERY = """
{
  proposal(id: %(id)s) {
    id
    title
    summary
    description
    public_created_at
    cached_votes_up
    comments_count
    confidence_score
    hot_score
    geozone_id
    video_url
    retired_reason
    retired_explanation
    retired_at
    public_author { id username }
    geozone { id name }
    tags {
      edges { node { id name kind taggings_count } }
    }
    votes_for {
      edges { node { id vote_flag votable_id votable_type public_created_at } }
    }
  }
}
"""

# --- Votes (user-contributed) ----------------------------------------------

VOTES_QUERY = """
{
  votes(first: %(page_size)s%(after)s) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        vote_flag
        votable_id
        votable_type
        public_created_at
      }
    }
  }
}
"""

VOTE_BY_ID_QUERY = """
{
  vote(id: %(id)s) {
    id
    vote_flag
    votable_id
    votable_type
    public_created_at
  }
}
"""

# --- Debates ---------------------------------------------------------------

DEBATES_QUERY = """
{
  debates(first: %(page_size)s%(after)s) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        title
        description
        public_created_at
        cached_votes_up
        cached_votes_down
        cached_votes_total
        comments_count
        confidence_score
        hot_score
        public_author { id username }
      }
    }
  }
}
"""

DEBATE_BY_ID_QUERY = """
{
  debate(id: %(id)s) {
    id
    title
    description
    public_created_at
    cached_votes_up
    cached_votes_down
    cached_votes_total
    comments_count
    confidence_score
    hot_score
    public_author { id username }
    tags {
      edges { node { id name kind taggings_count } }
    }
    votes_for {
      edges { node { id vote_flag votable_id votable_type public_created_at } }
    }
  }
}
"""

# --- Comments --------------------------------------------------------------

COMMENTS_QUERY = """
{
  comments(first: %(page_size)s%(after)s) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        body
        ancestry
        commentable_id
        commentable_type
        cached_votes_up
        cached_votes_down
        cached_votes_total
        confidence_score
        public_created_at
        public_author { id username }
      }
    }
  }
}
"""

COMMENT_BY_ID_QUERY = """
{
  comment(id: %(id)s) {
    id
    body
    ancestry
    commentable_id
    commentable_type
    cached_votes_up
    cached_votes_down
    cached_votes_total
    confidence_score
    public_created_at
    public_author { id username }
    votes_for {
      edges { node { id vote_flag votable_id votable_type public_created_at } }
    }
  }
}
"""

# --- Users -----------------------------------------------------------------

USERS_QUERY = """
{
  users(first: %(page_size)s%(after)s) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        username
      }
    }
  }
}
"""

USER_BY_ID_QUERY = """
{
  user(id: %(id)s) {
    id
    username
    public_proposals {
      edges { node { id title public_created_at cached_votes_up } }
    }
    public_debates {
      edges { node { id title public_created_at cached_votes_up } }
    }
  }
}
"""

# --- Budgets ---------------------------------------------------------------

BUDGETS_QUERY = """
{
  budgets(first: %(page_size)s%(after)s) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        name
        phase
      }
    }
  }
}
"""

BUDGET_BY_ID_QUERY = """
{
  budget(id: %(id)s) {
    id
    name
    phase
    investments {
      edges {
        node {
          id
          title
          price
          feasibility
          location
          comments_count
        }
      }
    }
  }
}
"""

# --- Budget investments ----------------------------------------------------

BUDGET_INVESTMENTS_QUERY = """
{
  budgets(first: %(page_size)s%(after)s) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        name
        investments {
          edges {
            node {
              id
              title
              description
              price
              feasibility
              location
              comments_count
              public_author { id username }
            }
          }
        }
      }
    }
  }
}
"""

# --- Geozones --------------------------------------------------------------

GEOZONES_QUERY = """
{
  geozones(first: %(page_size)s%(after)s) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        name
      }
    }
  }
}
"""

GEOZONE_BY_ID_QUERY = """
{
  geozone(id: %(id)s) {
    id
    name
  }
}
"""

# --- Tags ------------------------------------------------------------------

TAGS_QUERY = """
{
  tags(first: %(page_size)s%(after)s) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        name
        kind
        taggings_count
      }
    }
  }
}
"""

TAG_BY_ID_QUERY = """
{
  tag(id: %(id)s) {
    id
    name
    kind
    taggings_count
  }
}
"""

# --- Milestones ------------------------------------------------------------

MILESTONES_QUERY = """
{
  milestones(first: %(page_size)s%(after)s) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        title
        description
        publication_date
      }
    }
  }
}
"""

MILESTONE_BY_ID_QUERY = """
{
  milestone(id: %(id)s) {
    id
    title
    description
    publication_date
  }
}
"""

# --- Proposal notifications ------------------------------------------------

PROPOSAL_NOTIFICATIONS_QUERY = """
{
  proposal_notifications(first: %(page_size)s%(after)s) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        title
        body
        public_created_at
        proposal_id
      }
    }
  }
}
"""

PROPOSAL_NOTIFICATION_BY_ID_QUERY = """
{
  proposal_notification(id: %(id)s) {
    id
    title
    body
    public_created_at
    proposal_id
  }
}
"""

# ---------------------------------------------------------------------------
# Bulk query that fetches several scalar-only resources in one request
# (limited to ≤2 collections per query; here we fetch two leaf resources)
# ---------------------------------------------------------------------------

BULK_GEOZONES_AND_TAGS_QUERY = """
{
  geozones(first: 25) {
    edges { node { id name } }
  }
  tags(first: 25) {
    edges { node { id name kind taggings_count } }
  }
}
"""

# ---------------------------------------------------------------------------
# HTTP / GraphQL helpers
# ---------------------------------------------------------------------------


def _execute_query(
    query: str,
    endpoint: str = API_ENDPOINT,
    retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    """Send a GraphQL query and return the parsed JSON response."""
    payload = json.dumps({"query": query})
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                endpoint, data=payload, headers=HEADERS, timeout=30
            )
            response.raise_for_status()
            result = response.json()
            if "errors" in result:
                raise ValueError(f"GraphQL errors: {result['errors']}")
            return result
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                print(
                    f"  [attempt {attempt}/{retries}] error: {exc} – retrying in {RETRY_DELAY}s",
                    file=sys.stderr,
                )
                time.sleep(RETRY_DELAY)

    raise RuntimeError(
        f"Query failed after {retries} attempts: {last_error}"
    ) from last_error


def _paginate(
    collection_key: str,
    query_template: str,
    endpoint: str = API_ENDPOINT,
) -> Generator[Dict[str, Any], None, None]:
    """
    Yield every node from a paginated GraphQL connection.

    ``query_template`` must contain ``%(page_size)s`` and ``%(after)s``
    placeholders.
    """
    cursor: Optional[str] = None
    page = 0

    while True:
        after_param = f', after: "{cursor}"' if cursor else ""
        query = query_template % {
            "page_size": PAGE_SIZE,
            "after": after_param,
        }
        print(f"  Fetching {collection_key} page {page} …", file=sys.stderr)
        result = _execute_query(query, endpoint=endpoint)
        connection = result["data"][collection_key]
        edges = connection.get("edges", [])
        page_info = connection.get("pageInfo", {})

        for edge in edges:
            yield edge["node"]

        if not page_info.get("hasNextPage"):
            break

        cursor = page_info["endCursor"]
        page += 1


# ---------------------------------------------------------------------------
# Public API – individual resource fetchers
# ---------------------------------------------------------------------------


def fetch_proposals(endpoint: str = API_ENDPOINT) -> List[Dict[str, Any]]:
    """Return all proposals (paginated)."""
    return list(_paginate("proposals", PROPOSALS_QUERY, endpoint))


def fetch_votes(endpoint: str = API_ENDPOINT) -> List[Dict[str, Any]]:
    """Return all votes (paginated)."""
    return list(_paginate("votes", VOTES_QUERY, endpoint))


def fetch_debates(endpoint: str = API_ENDPOINT) -> List[Dict[str, Any]]:
    """Return all debates (paginated)."""
    return list(_paginate("debates", DEBATES_QUERY, endpoint))


def fetch_comments(endpoint: str = API_ENDPOINT) -> List[Dict[str, Any]]:
    """Return all comments (paginated)."""
    return list(_paginate("comments", COMMENTS_QUERY, endpoint))


def fetch_users(endpoint: str = API_ENDPOINT) -> List[Dict[str, Any]]:
    """Return all users (paginated)."""
    return list(_paginate("users", USERS_QUERY, endpoint))


def fetch_budgets(endpoint: str = API_ENDPOINT) -> List[Dict[str, Any]]:
    """Return all budgets (paginated)."""
    return list(_paginate("budgets", BUDGETS_QUERY, endpoint))


def fetch_budget_investments(
    endpoint: str = API_ENDPOINT,
) -> List[Dict[str, Any]]:
    """Return all budget investments, grouped by budget (paginated)."""
    return list(_paginate("budgets", BUDGET_INVESTMENTS_QUERY, endpoint))


def fetch_geozones(endpoint: str = API_ENDPOINT) -> List[Dict[str, Any]]:
    """Return all geozones (paginated)."""
    return list(_paginate("geozones", GEOZONES_QUERY, endpoint))


def fetch_tags(endpoint: str = API_ENDPOINT) -> List[Dict[str, Any]]:
    """Return all tags (paginated)."""
    return list(_paginate("tags", TAGS_QUERY, endpoint))


def fetch_milestones(endpoint: str = API_ENDPOINT) -> List[Dict[str, Any]]:
    """Return all milestones (paginated)."""
    return list(_paginate("milestones", MILESTONES_QUERY, endpoint))


def fetch_proposal_notifications(
    endpoint: str = API_ENDPOINT,
) -> List[Dict[str, Any]]:
    """Return all proposal notifications (paginated)."""
    return list(
        _paginate(
            "proposal_notifications",
            PROPOSAL_NOTIFICATIONS_QUERY,
            endpoint,
        )
    )


# ---------------------------------------------------------------------------
# Public API – single-record fetchers
# ---------------------------------------------------------------------------


def fetch_proposal_by_id(
    record_id: int, endpoint: str = API_ENDPOINT
) -> Dict[str, Any]:
    """Return a single proposal including tags and votes."""
    result = _execute_query(
        PROPOSAL_BY_ID_QUERY % {"id": record_id}, endpoint=endpoint
    )
    return result["data"]["proposal"]


def fetch_vote_by_id(
    record_id: int, endpoint: str = API_ENDPOINT
) -> Dict[str, Any]:
    result = _execute_query(
        VOTE_BY_ID_QUERY % {"id": record_id}, endpoint=endpoint
    )
    return result["data"]["vote"]


def fetch_debate_by_id(
    record_id: int, endpoint: str = API_ENDPOINT
) -> Dict[str, Any]:
    result = _execute_query(
        DEBATE_BY_ID_QUERY % {"id": record_id}, endpoint=endpoint
    )
    return result["data"]["debate"]


def fetch_comment_by_id(
    record_id: int, endpoint: str = API_ENDPOINT
) -> Dict[str, Any]:
    result = _execute_query(
        COMMENT_BY_ID_QUERY % {"id": record_id}, endpoint=endpoint
    )
    return result["data"]["comment"]


def fetch_user_by_id(
    record_id: int, endpoint: str = API_ENDPOINT
) -> Dict[str, Any]:
    result = _execute_query(
        USER_BY_ID_QUERY % {"id": record_id}, endpoint=endpoint
    )
    return result["data"]["user"]


def fetch_budget_by_id(
    record_id: int, endpoint: str = API_ENDPOINT
) -> Dict[str, Any]:
    result = _execute_query(
        BUDGET_BY_ID_QUERY % {"id": record_id}, endpoint=endpoint
    )
    return result["data"]["budget"]


def fetch_geozone_by_id(
    record_id: int, endpoint: str = API_ENDPOINT
) -> Dict[str, Any]:
    result = _execute_query(
        GEOZONE_BY_ID_QUERY % {"id": record_id}, endpoint=endpoint
    )
    return result["data"]["geozone"]


def fetch_tag_by_id(
    record_id: int, endpoint: str = API_ENDPOINT
) -> Dict[str, Any]:
    result = _execute_query(
        TAG_BY_ID_QUERY % {"id": record_id}, endpoint=endpoint
    )
    return result["data"]["tag"]


def fetch_milestone_by_id(
    record_id: int, endpoint: str = API_ENDPOINT
) -> Dict[str, Any]:
    result = _execute_query(
        MILESTONE_BY_ID_QUERY % {"id": record_id}, endpoint=endpoint
    )
    return result["data"]["milestone"]


def fetch_proposal_notification_by_id(
    record_id: int, endpoint: str = API_ENDPOINT
) -> Dict[str, Any]:
    result = _execute_query(
        PROPOSAL_NOTIFICATION_BY_ID_QUERY % {"id": record_id}, endpoint=endpoint
    )
    return result["data"]["proposal_notification"]


# ---------------------------------------------------------------------------
# Bulk fetch – all resources in one call sequence
# ---------------------------------------------------------------------------

#: Maps resource name → (collection fetcher, by-id fetcher)
RESOURCES: Dict[str, Any] = {
    "proposals": (fetch_proposals, fetch_proposal_by_id),
    "votes": (fetch_votes, fetch_vote_by_id),
    "debates": (fetch_debates, fetch_debate_by_id),
    "comments": (fetch_comments, fetch_comment_by_id),
    "users": (fetch_users, fetch_user_by_id),
    "budgets": (fetch_budgets, fetch_budget_by_id),
    "budget_investments": (fetch_budget_investments, None),
    "geozones": (fetch_geozones, fetch_geozone_by_id),
    "tags": (fetch_tags, fetch_tag_by_id),
    "milestones": (fetch_milestones, fetch_milestone_by_id),
    "proposal_notifications": (
        fetch_proposal_notifications,
        fetch_proposal_notification_by_id,
    ),
}


def fetch_all(endpoint: str = API_ENDPOINT) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch every available resource collection and return them in a single
    dictionary keyed by resource name.
    """
    data: Dict[str, List[Dict[str, Any]]] = {}
    for name, (fetcher, _) in RESOURCES.items():
        print(f"Fetching {name} …", file=sys.stderr)
        try:
            data[name] = fetcher(endpoint=endpoint)
            print(
                f"  → {len(data[name])} records retrieved.", file=sys.stderr
            )
        except (requests.RequestException, ValueError, RuntimeError, KeyError) as exc:
            print(f"  [WARNING] Could not fetch {name}: {exc}", file=sys.stderr)
            data[name] = []
    return data


# ---------------------------------------------------------------------------
# Bulk multi-resource query example (two collections in one HTTP request)
# ---------------------------------------------------------------------------


def fetch_geozones_and_tags_bulk(
    endpoint: str = API_ENDPOINT,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Demonstrate fetching two small collections (geozones + tags) in a single
    GraphQL request – the API allows at most two collections per query.
    """
    result = _execute_query(BULK_GEOZONES_AND_TAGS_QUERY, endpoint=endpoint)
    return {
        "geozones": [e["node"] for e in result["data"]["geozones"]["edges"]],
        "tags": [e["node"] for e in result["data"]["tags"]["edges"]],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch data from a Consul Democracy GraphQL API."
    )
    parser.add_argument(
        "--endpoint",
        default=API_ENDPOINT,
        help="GraphQL endpoint URL (default: %(default)s)",
    )
    parser.add_argument(
        "--resource",
        choices=list(RESOURCES.keys()) + ["all"],
        default="all",
        help="Which resource to fetch (default: all)",
    )
    parser.add_argument(
        "--id",
        type=int,
        default=None,
        metavar="ID",
        help="Fetch a single record by ID (requires --resource)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write JSON output to FILE instead of stdout",
    )
    parser.add_argument(
        "--bulk-demo",
        action="store_true",
        help="Show bulk multi-resource query demo (geozones + tags)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.bulk_demo:
        print("Running bulk multi-resource demo …", file=sys.stderr)
        result = fetch_geozones_and_tags_bulk(endpoint=args.endpoint)
        _output(result, args.output)
        return

    if args.id is not None:
        if args.resource == "all":
            parser.error("--id requires --resource to be specified")
        _, by_id_fetcher = RESOURCES[args.resource]
        if by_id_fetcher is None:
            parser.error(
                f"Single-record lookup not supported for '{args.resource}'"
            )
        print(
            f"Fetching {args.resource} id={args.id} …", file=sys.stderr
        )
        result = by_id_fetcher(args.id, endpoint=args.endpoint)
        _output(result, args.output)
        return

    if args.resource == "all":
        result = fetch_all(endpoint=args.endpoint)
    else:
        fetcher, _ = RESOURCES[args.resource]
        print(f"Fetching {args.resource} …", file=sys.stderr)
        result = {args.resource: fetcher(endpoint=args.endpoint)}

    _output(result, args.output)


def _output(data: Any, filepath: Optional[str]) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False)
    if filepath:
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Output written to {filepath}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()