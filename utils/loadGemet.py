import rdflib
from rdflib.term import _toPythonMapping
from rdflib.namespace import XSD
import sys

# some gemet rdf file has empty datetime fields.
# modify date parser to avoid error
# 1. Save the old parser
old_datetime_parser = _toPythonMapping.get(XSD.dateTime)

# 2. Create a new, lenient parser
def lenient_parse_datetime(value):
    if not value or 'T' not in value:
        # Return None, or the original string, or anything you prefer
        return "2025-01-01T00:00:00Z" # None
    # Otherwise, call the original parser
    if old_datetime_parser:
        return old_datetime_parser(value)
    return None

# 3. Override the mapping
_toPythonMapping[XSD.dateTime] = lenient_parse_datetime


# 1. Create an empty RDFlib Graph
graph = rdflib.Graph()

# gemet: https://www.eionet.europa.eu/gemet/en/exports/rdf/latest
# backbone:   Themes and groups relationships as RDF
# skoscore:    SKOS broader and narrower relations as RDF
# skos: Entire GEMET thesaurus in SKOS format


# Parse the "SKOS broader/narrower" file
# Replace with the actual path/filename you downloaded
#graph.parse("gemet-groups-de.rdf", format="xml")
#graph.parse("gemet-groups-en.rdf", format="xml")

# Parse the German "Labels and definitions" file
#graph.parse("gemet-definitions-de.rdf", format="xml")
#graph.parse("gemet-definitions-en.rdf", format="xml")

# either us language specific files above
# or full core
graph.parse("/home/kugel/Downloads/gemet/gemet-skoscore.rdf", format="xml")

# Parse backbone
graph.parse("/home/kugel/Downloads/gemet/gemet-backbone.rdf", format="xml")

# Parse skos
graph.parse("/home/kugel/Downloads/gemet/gemet-skos.rdf", format="xml")

print(f"Total triples loaded: {len(graph)}")

output_file = "gemet_merged.rdf"
graph.serialize(destination=output_file, format="xml")

print(f"Serialized the merged graph as {output_file}.")



# themes don't have prefLabels ...
# Finding skos:Collection Entities
query_collections = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?collection (COALESCE(?prefLabel, ?rdfsLabel) AS ?label)
WHERE {
  ?collection a skos:Collection.
  OPTIONAL {
    ?collection skos:prefLabel ?prefLabel .
    FILTER(LANG(?prefLabel) = "de")
  }
  OPTIONAL {
    ?collection rdfs:label ?rdfsLabel .
    FILTER(LANG(?rdfsLabel) = "de")
  }
}
ORDER BY ?collection
"""

results = graph.query(query_collections)

print("Query collections")
if not results or len(results) == 0:
    print("No results for query_collections")
for row in results:
    # row.collection, row.label
    print(f"Collection URI: {row.collection} | German label: {row.label}")


# Looking for skos:member or skos:memberOf
query_members = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?collection ?collectionLabel ?member ?memberLabel
WHERE {
  ?collection a skos:Collection .
  ?collection skos:member ?member .
  OPTIONAL {
    ?collection skos:prefLabel ?collectionLabel .
    FILTER(LANG(?collectionLabel) = "de")
  }
  OPTIONAL {
    ?member skos:prefLabel ?memberLabel .
    FILTER(LANG(?memberLabel) = "de")
  }
}
LIMIT 50
"""

print("Query members")
if not results or len(results) == 0:
    print("No results for query_members")

for row in graph.query(query_members):
    print("Collection:", row.collection, row.collectionLabel or "")
    print("  => Member:", row.member, row.memberLabel or "")
    print("---")


## 

# Example SPARQL query to find broader-narrower pairs with German labels
query = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?child ?childLabel ?parent ?parentLabel
WHERE {
  ?child skos:broader ?parent .
  OPTIONAL {
    ?child skos:prefLabel ?childLabel .
    FILTER (LANG(?childLabel) = "de")
  }
  OPTIONAL {
    ?parent skos:prefLabel ?parentLabel .
    FILTER (LANG(?parentLabel) = "de")
  }
}
LIMIT 20
"""

results = graph.query(query)
print("Borad/narrow relations")
print("\nSome broader/narrower relationships (German labels if available):\n")
for row in results:
    # row.child, row.childLabel, row.parent, row.parentLabel
    print(f"- Child:   {row.childLabel or row.child}")
    print(f"  Broader: {row.parentLabel or row.parent}")
    print("---")


## ##########
# Example: Retrieve and print all German prefLabels
SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
print("rdflib search prefLabels")
for s, p, o in graph.triples((None, SKOS.prefLabel, None)):
    # Check if this label is in German
    if o.language == "de":
        print(f"Concept: {s}\n  German prefLabel: {o}\n")


# Example: Retrieve and print all German rdfsLabels
RDFS = rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#")
print("rdflib search rdfslabels")
for s, p, o in graph.triples((None, RDFS.label, None)):
    # Check if this label is in German
    if o.language == "de":
        print(f"Concept: {s}\n  German rdfsLabel: {o}\n")
    if o.language == "en":
        print(f"Concept: {s}\n  English rdfsLabel: {o}\n")


