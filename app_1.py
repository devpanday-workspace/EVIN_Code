import pandas as pd
import networkx as nx
from sqlalchemy import create_engine
import ffs  

DB_URL = (
    "postgresql://gdelt_reader:Access-301274@ep-old-sun-ahv7dfvy-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
)

engine = create_engine(DB_URL)

def fetch_events(
    start_date: int,
    end_date: int,
    limit: int | None = None
) -> pd.DataFrame:
    """
    Load GDELT events from PostgreSQL.
    Dates should be integers in YYYYMMDD format.
    """

    query = f"""
SELECT
    globaleventid      AS "GlobalEventID",
    sqldate            AS "Day",

    actor1code         AS "Actor1Code",
    actor1name         AS "Actor1Name",
    actor1countrycode  AS "Actor1CountryCode",

    actor1type1code    AS "Actor1Type1Code",
    actor1type2code    AS "Actor1Type2Code",
    actor1type3code    AS "Actor1Type3Code",

    actor1geo_countrycode AS "Actor1GeoCountryCode",

    actor2code         AS "Actor2Code",
    actor2name         AS "Actor2Name",
    actor2countrycode  AS "Actor2CountryCode",

    actor2type1code    AS "Actor2Type1Code",
    actor2type2code    AS "Actor2Type2Code",
    actor2type3code    AS "Actor2Type3Code",

    actor2geo_countrycode AS "Actor2GeoCountryCode",

    eventcode          AS "EventCode",
    eventbasecode      AS "EventBaseCode",
    eventrootcode      AS "EventRootCode",

    quadclass          AS "QuadClass",
    goldsteinscale     AS "GoldsteinScale",

    nummentions        AS "NumMentions",
    numsources         AS "NumSources",
    numarticles        AS "NumArticles",

    avgtone            AS "AvgTone",
    sourceurl          AS "SOURCEURL"

FROM public.gdelt_events
WHERE sqldate BETWEEN {start_date} AND {end_date}
    """

    if limit:
        query += f"\nLIMIT {limit}"

    return pd.read_sql(query, engine)

def load_cameo_lookup(path: str) -> dict:
    lookup = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or ":" not in line:
                continue

            code, label = line.split(":", 1)

            lookup[code.strip()] = label.strip()

    return lookup

def extract_triples(
    df: pd.DataFrame,
    cameo_lookup: dict | None = None
) -> pd.DataFrame:
    """
    Convert raw GDELT events into:
        subject --action--> object
    """

    triples = df.dropna(
        subset=["Actor1Name", "Actor2Name", "EventCode"]
    ).copy()

    triples["action"] = triples["EventCode"].astype(str)

    if cameo_lookup:
        triples["action"] = (
            triples["action"]
            .map(cameo_lookup)
            .fillna(triples["action"])
        )

    triples = triples.rename(
        columns={
            "Actor1Name": "subject",
            "Actor2Name": "object"
        }
    )

    return triples[
        [
    "subject",
    "object",
    "action",

    "Actor1Code",
    "Actor1CountryCode",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",
    "Actor1GeoCountryCode",

    "Actor2Code",
    "Actor2CountryCode",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",
    "Actor2GeoCountryCode",

    "EventCode",
    "EventRootCode",
    "QuadClass",

    "GoldsteinScale",
    "AvgTone",
    "NumMentions",

    "Day",
    "SOURCEURL"
]
        
    ]

def build_chain_graph(
    triples: pd.DataFrame
) -> nx.MultiDiGraph:
    """
    Build Actor -> Action -> Actor graph.
    """

    G = nx.MultiDiGraph()

    for _, row in triples.iterrows():

        G.add_edge(
            row["subject"],
            row["object"],
            action=row["action"],
            day=row["Day"],
            goldstein=row["GoldsteinScale"],
            tone=row["AvgTone"],
            mentions=row["NumMentions"]
        )

    return G

def find_chains(
    G: nx.MultiDiGraph,
    start_actor: str,
    max_hops: int = 3
):
    """
    DFS traversal producing:
        Actor -> Action -> Actor chains
    """

    chains = []

    def dfs(node, path, depth):

        if depth >= max_hops:
            chains.append(list(path))
            return

        if node not in G:
            return

        for nbr in G.successors(node):

            edge_dict = G.get_edge_data(node, nbr)

            for _, edge in edge_dict.items():

                path.append(
                    (
                        node,
                        edge["action"],
                        nbr,
                        edge["day"]
                    )
                )

                dfs(
                    nbr,
                    path,
                    depth + 1
                )

                path.pop()

    dfs(start_actor, [], 0)

    return chains

def show_database_columns():
    """
    Print all columns in gdelt_events.
    Useful for debugging schema issues.
    """

    query = """
    SELECT
        column_name,
        data_type
    FROM information_schema.columns
    WHERE table_schema='public'
    AND table_name='gdelt_events'
    ORDER BY ordinal_position
    """

    print(pd.read_sql(query, engine))

if __name__ == "__main__":

    print("Loading events...")
    
    print(pd.read_sql("""
    SELECT
        MIN(sqldate) AS min_date,
        MAX(sqldate) AS max_date,
        COUNT(*) AS total_rows
    FROM public.gdelt_events
    """, engine))

    events = fetch_events(
        start_date=20260621,
        end_date=20260622,
        limit=10000
    )

    print()
    print("Rows Loaded:", len(events))
    print()

    # print(events.head())
    triples = extract_triples(events)

    print("Triples Extracted:", len(triples))
    # print(triples.head())
    cameo_lookup = load_cameo_lookup("cameo.txt")
    semantic_results = ffs.process_triples_dataframe(triples,cameo_lookup = cameo_lookup)
    print(
        f"Semantic Results: "
        f"{len(semantic_results)}"
    )
    sem_count= {}
    for result in semantic_results:

        applicable = any(
        r.applicable
        for r in result["sphere_results"].values()
    )
        # if applicable:
        #     print()
        #     print(
        #         result["subject"],
        #         "->",
        #         result["object"]
        #     )
        #     print(
        #         "Event:",result["event_code"]
        #     )
        # for sphere_name, sphere_result in (
        #     result["sphere_results"].items()
        # ):
        #     if sphere_result.applicable:
        #         print(
        #             sphere_name,
        #             sphere_result.score,
        #             sphere_result.confidence
        #         )
        #         print(
        #             "Event:",
        #             result["event_code"],
        #             "Sphere:",
        #             sphere_name,
        #             "Score:",
        #             sphere_result.score
        #             )
        for name, r in result["sphere_results"].items():
            if r.applicable:
                sem_count[name] = sem_count.get(name, 0) + 1
                
    print("\nSphere Counts")
    print(sem_count)            

    G = build_chain_graph(triples)

    print(
        f"Graph contains "
        f"{G.number_of_nodes()} actors and "
        f"{G.number_of_edges()} actions"
    )

    if len(triples) > 0:

        top_actor = (
            triples["subject"]
            .value_counts()
            .index[0]
        )
        print("Top actor:", top_actor)
        chains = find_chains(
            G,
            top_actor,
            max_hops=2
        )
        print(
            f"Found {len(chains)} chains"
        )
        # print()
        # for chain in chains[:10]:
        #     print(chain)