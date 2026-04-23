# consulApi

Python client for the [Consul Democracy](https://consuldemocracy.org/) GraphQL
API.  Retrieves all publicly available data from a Consul instance one-by-one
and in bulk, with automatic pagination.

Target instance: **https://consul.ok-lab-karlsruhe.de/**

---

## Available resources

| Resource | Description |
|---|---|
| `proposals` | User-contributed proposals (most important) |
| `votes` | Votes on proposals, debates and comments |
| `debates` | Debates |
| `comments` | Comments on debates, proposals and other comments |
| `users` | Public user profiles |
| `budgets` | Participatory budgets |
| `budget_investments` | Budget investments (nested inside budgets) |
| `geozones` | Districts / geographic zones |
| `tags` | Tags on debates and proposals |
| `milestones` | Milestones for proposals, investments and processes |
| `proposal_notifications` | Notifications linked to proposals |

---

## Setup

```bash
pip install -r requirements.txt
```

The only mandatory dependency is **`requests`**.  
`gql[requests]` is also listed as an optional GraphQL client library.

---

## Usage

### Command-line

```bash
# Fetch all resources and print as JSON
python consul_api.py

# Fetch a single resource
python consul_api.py --resource proposals
python consul_api.py --resource votes
python consul_api.py --resource debates
python consul_api.py --resource comments
python consul_api.py --resource users
python consul_api.py --resource budgets
python consul_api.py --resource budget_investments
python consul_api.py --resource geozones
python consul_api.py --resource tags
python consul_api.py --resource milestones
python consul_api.py --resource proposal_notifications

# Fetch a single record by ID
python consul_api.py --resource proposals --id 1
python consul_api.py --resource votes --id 42
python consul_api.py --resource debates --id 5

# Write output to a file
python consul_api.py --resource proposals --output proposals.json

# Demonstrate fetching two collections in one HTTP request
python consul_api.py --bulk-demo

# Use a custom endpoint
python consul_api.py --endpoint https://demo.consuldemocracy.org/graphql
```

### Python API

```python
from consul_api import (
    fetch_proposals,       # all proposals (paginated)
    fetch_votes,           # all votes (paginated)
    fetch_debates,         # all debates (paginated)
    fetch_comments,        # all comments (paginated)
    fetch_users,           # all users (paginated)
    fetch_budgets,         # all budgets (paginated)
    fetch_budget_investments,  # investments nested in budgets
    fetch_geozones,        # all geozones (paginated)
    fetch_tags,            # all tags (paginated)
    fetch_milestones,      # all milestones (paginated)
    fetch_proposal_notifications,  # all proposal notifications
    fetch_proposal_by_id,  # single proposal with tags & votes
    fetch_vote_by_id,      # single vote record
    fetch_debate_by_id,    # single debate with tags & votes
    fetch_comment_by_id,   # single comment with votes
    fetch_user_by_id,      # single user with proposals & debates
    fetch_budget_by_id,    # single budget with investments
    fetch_all,             # all resources in sequence
    fetch_geozones_and_tags_bulk,  # two collections in one HTTP request
)

# Fetch all proposals with automatic pagination
proposals = fetch_proposals()
print(f"{len(proposals)} proposals retrieved")

# Fetch a single proposal (includes nested tags and votes)
proposal = fetch_proposal_by_id(1)
print(proposal["title"])

# Fetch all votes
votes = fetch_votes()
print(f"{len(votes)} votes retrieved")

# Fetch everything at once
all_data = fetch_all()
for resource, records in all_data.items():
    print(f"{resource}: {len(records)} records")

# Two collections in one HTTP request (Consul allows max 2 collections/query)
result = fetch_geozones_and_tags_bulk()
print(result["geozones"])
print(result["tags"])
```

### Run the examples script

```bash
python fetch_examples.py
```

---

## API constraints

The Consul Democracy GraphQL API enforces the following limits:

| Constraint | Value |
|---|---|
| Maximum page size | 25 records |
| Maximum query depth | 8 levels |
| Maximum collections per query | 2 |
| Maximum query complexity | 2500 |

The client respects these limits automatically: paginated fetches request 25
records per page, and bulk queries never include more than two collection
fields.

---

## API endpoint

The GraphQL endpoint is available at `/graphql` on any Consul Democracy
instance.  An interactive browser interface (GraphiQL) is available at
`/graphiql`.

- Production: `https://consul.ok-lab-karlsruhe.de/graphql`
- GraphiQL: `https://consul.ok-lab-karlsruhe.de/graphiql`
- Official demo: `https://demo.consuldemocracy.org/graphql`

---

## References

- [Consul Democracy GraphQL documentation](https://github.com/consuldemocracy/consuldemocracy/blob/master/docs/en/features/graphql.md)
- [GraphQL specification](https://graphql.org/learn/)
- [requests library](https://docs.python-requests.org/)
- [gql library](https://gql.readthedocs.io/)
