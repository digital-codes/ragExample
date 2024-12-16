# ragExample
Example for retrieval augmented genereration (RAG)

## Tools

RAG fundamentals: combine LLM with knowledge base

 * Searchable knowledge base: Text search, **vector database**, knowledge graph
 * LLM: locally installed e.g. via Ollama, **OpenAI**

### Search engines

Tested [chromadb](https://docs.trychroma.com/) and [elasticsearch](https://www.elastic.co/)

Install elasticsearch from [here](https://www.elastic.co/downloads/elasticsearch). Or use docker.

Install chromadb as documented [here](https://docs.trychroma.com/getting-started)

Elasticsearch provided better results, might be related to embedding algorithm 
(default on chromadb, transformer on elastic. chroma could use same embedder as well ...)

Elasticsearch needs more than 8GB of memory (dies on cloud VM). 
Chromadb works well on same machine (in principle)

#### Embeddings

Embedding options for search engines to be investigated. So far, built-in embedder from chromadb vs 
[sentence-transformer](https://huggingface.co/sentence-transformers) with 
[deepset/gbert-base](https://huggingface.co/deepset/gbert-base) for German. 

**Use something else for non-German application like [all-MiniLM-L6-v1](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v1)**


#### Indexing

Documents are indexed using embeddings. Make sure to store text content and useful metadata as well. Chunking (break large documents into pieces) recommended, chunk size effects to be investigated (currently 1000 words, 50 overlapp). 

**Use same embedding for indexing and query**

### LLMs

Tested locally [Ollama](https://ollama.com/) with [llama3.2, 3B](https://ollama.com/library/llama3.2) 
Works but very slow without GPU

Tested with OpenAI API. Works. *Inital testing, maybe 50 trial queries ~ 50k tokens, 0.30€*

Would also like to test [*LLäMmlein*](https://arxiv.org/abs/2411.11171) from [Uni Würzburg](https://www.informatik.uni-wuerzburg.de/datascience/projects/nlp/llammlein/), available [here](https://huggingface.co/collections/LSX-UniWue/llammlein-6732ff41f3705c686e605762)


### Data Sources 

 1. [Ratsinformation Karlsruhe](https://sitzungskalender.karlsruhe.de/db/ratsinformation/start)
    All documents since 2014 downloaded. Text and metadata extraction still incomplete (e.g. missing titles). Tests on 2024 subset provide reasonable results.
 2. [Klimaschutzkonzeppt Karlsruhe](https://www.karlsruhe.de/umwelt-klima/klimaschutz-klimaanpassung/klimaschutzaktivitaeten/klimaschutzziele-der-stadt-karlsruhe) 

#### Examples

*Using elasticsearch for Ratsinfomation*

massnahmen zum klimaschutz
> (rag) rag@ragtest:~/temp/okl/ka-rat$ python ragElasticOpenai.py  
Connected to ES! 
🔍 Enter your query:  massnahmen zum klimaschutz  
Die Klimaschutzmaßnahmen umfassen:  
    - Implementierung von Nachtlüftungskonzepten  zur sommerlichen Auskühlung.  
    - Sanierung und Dämmung von Flachdächern sowie extensive Dach- und Fassadenbegrünungen.  
    - Ausbau von Photovoltaikanlagen zur Nutzung solarer Energie.   
    - Umrüstung der Beleuchtung auf energieeffiziente LED-Technologien.   
    - Retention von Niederschlagswasser zur Wasserbewirtschaftung.   
    - Reduzierung und Elektrifizierung des städtischen Fahrzeugbestands.   
    - Nachhaltiges Bauen mit klimafreundlichen Materialien und verbesserten Gebäudehüllen.  
    - Ausbau von Nahwärmenetzen und Förderung erneuerbarer Energien in der Energieinfrastruktur.   
    Diese Maßnahmen zielen darauf ab, den Energieverbrauch zu senken, Emissionen zu reduzieren und die Anpassung an den Klimawandel zu fördern.   
    ['2024-termin-9545-top3-link2.json_chunk_26', '2024-termin-9526-top1-link3.json_chunk_20', '2024-termin-9776-top10-link1.json_chunk_0']
    [1.5776099, 1.5662116, 1.5628055]
    [{'filename': '2024-termin-9545-top3-link2.json', 'indexdate': '2024-12-14T16:18:26.359981', 'dcterms:modified': '2024-05-02T12:14:10Z', 'dc:language': 'de-DE'}, {'filename': '2024-termin-9526-top1-link3.json', 'indexdate': '2024-12-14T16:18:49.879155', 'dcterms:modified': '2022-05-31T11:26:58Z'}, {'filename': '2024-termin-9776-top10-link1.json', 'indexdate': '2024-12-14T16:20:41.252193', 'dcterms:modified': '2024-08-28T07:36:59Z', 'dc:language': 'de-DE'}]


fahrtkosten zur schülerbeförderung   
> (rag) rag@ragtest:~/temp/okl/ka-rat$ python ragElasticOpenai.py   
Connected to ES!   
🔍 Enter your query: fahrtkosten zur schülerbeförderung   
Die aktuellen Fahrtkosten für die Schülerbeförderung zu Grundschulförderklassen werden im Kontext der Anfrage kritisch betrachtet. Der geplante Wegfall der kostenlosen Beförderung („freigestellter Schülerverkehr“) würde bedeuten, dass Eltern ihre Kinder selbst transportieren müssen. Dies könnte zusätzliche finanzielle Belastungen verursachen, insbesondere für berufstätige Eltern. Zudem müssten möglicherweise alternative Lösungen wie verlängerte Kita-Betreuung oder zusätzliche Zuschüsse eingeführt werden, um die Bildungschancen der betroffenen Kinder zu sichern. Die Stadtverwaltung bewertet die Kosten-Nutzen-Aspekte und deren Auswirkungen auf die betroffenen Familien und Bildungseinrichtungen.   
['2024-termin-9531-top28-link1.json_chunk_0', '2024-termin-9528-top27-link1.json_chunk_0', '2024-termin-9827-top2-link1.json_chunk_0']
[1.5818721, 1.5772125, 1.5701231]
[{'filename': '2024-termin-9531-top28-link1.json', 'indexdate': '2024-12-14T16:22:13.880498', 'dcterms:modified': '2024-09-12T11:38:35Z', 'dc:language': 'de-DE'}, {'filename': '2024-termin-9528-top27-link1.json', 'indexdate': '2024-12-14T16:21:00.291752', 'dcterms:modified': '2024-07-03T11:41:57Z', 'dc:language': 'de-DE'}, {'filename': '2024-termin-9827-top2-link1.json', 'indexdate': '2024-12-14T16:18:45.329838', 'dcterms:modified': '2024-04-30T11:22:58Z', 'dc:language': 'de-DE'}]

erinnerungswege palmbach  
> (rag) rag@ragtest:~/temp/okl/ka-rat$ python ragElasticOpenai.py   
Connected to ES!   
🔍 Enter your query: erinnerungswege palmbach   
Die Erinnerungswege in Palmbach sind Gedenkpfade, die historische Ereignisse oder Persönlichkeiten würdigen. Sie sind mit speziellen Blindenleitsystemen ausgestattet, um Barrierefreiheit zu gewährleisten. Im Rahmen des Jahresbauprogramms 2024 plant die Stadt Karlsruhe möglicherweise Renovierungen oder Erweiterungen dieser Wege, einschließlich verbesserter Pflasterungen und Grünflächen. Änderungen am Kantstein und an der Rinnenführung könnten ebenfalls vorgesehen sein, um die Wege an die städtischen Infrastrukturprojekte anzupassen. Für detaillierte Informationen empfiehlt sich eine Kontaktaufnahme mit dem Tiefbauamt oder dem Baustellenmanagement der Stadt Karlsruhe.   
['2024-termin-9531-top23-link4.json_chunk_0', '2024-termin-9552-top2-link3.json_chunk_0', '2024-termin-9524-top20-link5.json_chunk_0']
[1.6132288, 1.5614281, 1.5602466]
[{'filename': '2024-termin-9531-top23-link4.json', 'indexdate': '2024-12-14T16:15:16.314280', 'dcterms:modified': '2024-06-07T15:26:48Z', 'dc:language': 'de-DE'}, {'filename': '2024-termin-9552-top2-link3.json', 'indexdate': '2024-12-14T16:15:27.478850', 'dcterms:modified': '2024-09-06T09:48:45Z'}, {'filename': '2024-termin-9524-top20-link5.json', 'indexdate': '2024-12-14T16:15:35.473790', 'dcterms:modified': '2024-05-14T16:12:18Z'}]

probleme mit wertstoffsammlung 
> (rag) rag@ragtest:~/temp/okl/ka-rat$ python ragElasticOpenai.py   
Connected to ES!   
🔍 Enter your query: probleme mit wertstoffsammlung   
In Karlsruhe gibt es erhebliche Probleme mit der Wertstoffsammlung:   
    1. **Falsche Entsorgung**: Viele Pfandflaschen werden trotz Pfandsystem in Restmüll geworfen, besonders in der Innenstadt.   
    2. **Planlosigkeit der Verwaltung**: Verwaltungsvorlagen zeigen mangelnde konkrete Konzepte und effektive Maßnahmen.   
    3. **Versprechen gebrochen**: Zusicherungen zur Beibehaltung der aktuellen Wertstofftonnen wurden nicht eingehalten, was zu Verunsicherung und Frustration bei Bürgern führt.   
    4. **Arbeitsbedingungen**: Hoher Krankenstand bei Müllarbeitern aufgrund schlechter Arbeitsbedingungen und fehlender Anpassungen.   
    5. **Unzureichende Infrastruktur**: Fehlende oder ineffiziente Sammelstellen erschweren das Recycling.      
Diese Probleme erfordern dringend koordinierte und nachhaltige Lösungen.   
['2024-termin-9524-top20-link1.json_chunk_0', '2024-termin-9531-top20-link3.json_chunk_2', '2024-termin-9532-top37-link1.json_chunk_0']
[1.5981559, 1.5930581, 1.5916256]
[{'filename': '2024-termin-9524-top20-link1.json', 'indexdate': '2024-12-14T16:19:27.922537', 'dcterms:modified': '2024-04-30T08:56:44Z', 'dc:language': 'de-DE'}, {'filename': '2024-termin-9531-top20-link3.json', 'indexdate': '2024-12-14T16:17:01.698201', 'dcterms:modified': '2024-10-14T05:52:05Z', 'dc:language': 'de-DE'}, {'filename': '2024-termin-9532-top37-link1.json', 'indexdate': '2024-12-14T16:21:14.050966', 'dcterms:modified': '2024-10-09T09:59:10Z', 'dc:language': 'de-DE'}]

*Using Chromadb for Klimaschutzkonzept* Brackets indicate selected action topics from search phase

 Wie hoch ist das Ausbauziel für Photovoltaik auf städtischen Liegenschaften?
 
> Das Ausbauziel für Photovoltaik auf städtischen Liegenschaften liegt bei ca. 10.000 kWp bis 2040, wobei bis 2030 bereits 5.000 kWp erreicht werden sollen. Dies entspricht einer jährlichen CO2-Einsparung von rund 1.100 t/a ab 2030.  
[['B_1_2', 'E_1_3', 'A_4_1']]


Wie will die Stadt den Fahrradverkehr stärken?

> Die Stadt Karlsruhe will den Fahrradverkehr stärken, indem sie ein Leitprojekt "Öffentlicher Raum und Mobilität" im Rahmen des IQ-Korridorthemas "Zukunft Innenstadt" erarbeitet. Dieses Konzept soll alle Mobilitätsarten und deren Wegebeziehungen in der Innenstadt integriert behandeln und eine hohe Priorität auf der Reduzierung des MIV (motorisierter Individualverkehr) und der Freihaltung insbesondere von Fußgängerbereichen vom Kfz-Verkehr legen. Konkrete Vorschläge sollen auch hinsichtlich Einfuhrbeschränkungen und Reduzierungsmöglichkeiten für den Lieferverkehr erarbeitet werden. Zudem sind erste Schritte zur Regulation des MIV im Innenstadtbereich bereits in Umsetzung oder konkret geplant, wie die Sperrung für den Kfz-Verkehr in der Erbprinzenstraße. 
[['E_1_1', 'D_1_2', 'E_1_2']]


Auf welche Energieträger setzt die Stadt bei der Energiewende?  

> Die Stadt Karlsruhe setzt bei der Energiewende auf Erneuerbare Energien und plant, den Anteil erneuerbarer Energien an der Wärmeversorgung durch den Energieträgerwechsel zu erhöhen. Zusätzlich fördert sie die Elektromobilität durch den Ausbau der Ladeinfrastruktur im öffentlichen Raum. Sie strebt an, dass jedes dritteAuto klimaneutral fährt. Die Stadtwerke Karlsruhe spielen dabei eine zentrale Rolle in der Umsetzung dieser Maßnahmen.  
[['E_1_2', 'B_2_2', 'D_3_3']]


Wann soll Klimaneutralität erreicht werden?

> Laut dem Kontext sollen sowohl die Stadt Karlsruhe als auch die Landesregierung bis 2040 eine weitgehend klimaneutrale Verwaltung anstreben. Ergänzend dazu gibt es Bestrebungen am Karlsruher Institut für Technologie (KIT) und in der Hochschule Karlsruhe Wirtschaft und Technik (HsKA), eigene Initiativen zur Klimaneutralität zu starten.  
[['E_3_4', 'C_1_1', 'C_2_1']]


 Welche Angebote im Klimaschutz gibt es für die Wirtschaft / Unternehmen?
 
> In Karlsruhe werden Unternehmen durch das Klimaschutzmanagement für den Bereich Wirtschaft unterstützt. Dazu gehört ein "Kümmerer" für Energieeinsparpotenziale, eine Andockung an Projektstellen für Revitalisierung von Gewerbegebieten und gebietsbezogene Arbeit. Betreuung der Klimaallianz und Koordinierung neuer Netzwerke sind weitere Aufgaben. Es wird eine enge Zusammenarbeit mit Beratungs- und Fortbildungsakteuren angestrebt. Die Maßnahme schafft keine direkte CO2-Einsparung, ist aber Voraussetzung für projektbezogene Einsparungen.  
[['E_3_3', 'E_3_4', 'C_1_1']]


Wer berät die Bürger*innen bei Sanierungsmaßnahmen?

> Bürger*innen werden in Sanierungsmaßnahmen durch ein unabhängiges und kostenfreies Beratungs- und Sanierungsmanagement beraten. Das Bewusstsein für den persönlichen Beitrag zur klimaneutralen Stadt wird damit geschaffen.  
[['B_2_2', 'E_1_2', 'D_1_2']]



