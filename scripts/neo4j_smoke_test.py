"""Smoke-test the Neo4j connection.

Verifies the driver can reach the instance defined in docker-compose.yml
and runs a trivial write/read round-trip.

Usage:
  docker compose up -d
  python scripts/neo4j_smoke_test.py
"""

import os
from neo4j import GraphDatabase

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "medragpass")


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    with driver.session() as session:
        result = session.run(
            "MERGE (n:SmokeTest {name: $name}) RETURN n.name AS name",
            name="hello",
        )
        record = result.single()
        print(f"Neo4j replied: {record['name']}")

        session.run("MATCH (n:SmokeTest {name: $name}) DELETE n", name="hello")

    driver.close()
    print("Connection OK.")


if __name__ == "__main__":
    main()
