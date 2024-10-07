from neo4j import GraphDatabase


def get_compound_info(tx, compound_id):
    query = """
    MATCH (c:Compound {compound_id: $compound_id})
    RETURN c AS compound
    """
    result = tx.run(query, compound_id=compound_id)
    record = result.single()
    if record:
        compound = record["compound"]
        print(f"Compound: {compound['compound_id']}")
        print("Properties:")
        for key, value in compound.items():
            print(f"  {key}: {value}")
    else:
        print(f"No compound found with id: {compound_id}")


def find_reactions_with_similar_product_compounds(
    uri, user, password, target_compound_id, similarity_threshold=None, limit=1000
):
    def get_reactions(tx):
        query = (
            "MATCH (c:Compound {compound_id: $target_compound_id})-[sim:CHEMICAL_SIMILARITY]-(similar:Compound) "
            "MATCH (similar)-[:PRODUCT_OF]->(r:Reaction) "
            "WHERE toFloat(sim.distance) <= $similarity_threshold "
            "RETURN r.reaction_id AS reaction_id, "
            "       r.name AS reaction_name, "
            "       similar.compound_id AS similar_compound_id, "
            "       similar.name AS similar_compound_name, "
            "       similar.smiles AS similar_compound_smiles, "
            "       toFloat(sim.distance) AS similarity_distance "
            "ORDER BY similarity_distance ASC "
            "LIMIT $limit"
        )
        result = tx.run(
            query,
            target_compound_id=target_compound_id,
            similarity_threshold=similarity_threshold,
            limit=limit,
        )
        return [dict(record) for record in result]

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        reactions = session.execute_read(get_reactions)
    driver.close()
    return reactions


def find_proteins_by_ec_number(
    uri,
    user,
    password,
    ec_number,
    min_temp=None,
    max_temp=None,
    min_lat=None,
    max_lat=None,
    min_lon=None,
    max_lon=None,
    limit=1000,
):
    def get_proteins(tx):
        query = (
            "MATCH (p:Protein)-[:CATALYZES]->(r:Reaction) "
            "WHERE $ec_number IN p.ec_numbers "
            "OPTIONAL MATCH (m:Genome)-[:CONTAINS]->(p) "
            "OPTIONAL MATCH (m)-[:ORIGINATED_FROM]->(s:Sample) "
            "WHERE "
            "   ($min_temp IS NULL OR s.temperature >= $min_temp) AND "
            "   ($max_temp IS NULL OR s.temperature <= $max_temp) AND "
            "   ($min_lat IS NULL OR s.latitude >= $min_lat) AND "
            "   ($max_lat IS NULL OR s.latitude <= $max_lat) AND "
            "   ($min_lon IS NULL OR s.longitude >= $min_lon) AND "
            "   ($max_lon IS NULL OR s.longitude <= $max_lon) "
            "RETURN "
            "   p.protein_id AS protein_id, "
            "   p.name AS protein_name, "
            "   r.reaction_id AS reaction_id, "
            "   r.name AS reaction_name, "
            "   m.genome_id AS genome_id, "
            "   s.biosample_id AS sample_id, "
            "   s.temperature AS sample_temperature, "
            "   s.latitude AS sample_latitude, "
            "   s.longitude AS sample_longitude "
            "LIMIT $limit"
        )
        result = tx.run(
            query,
            ec_number=ec_number,
            min_temp=min_temp,
            max_temp=max_temp,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            limit=limit,
        )
        return [dict(record) for record in result]

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        proteins = session.execute_read(get_proteins)

    driver.close()
    return proteins


def find_similar_compounds_and_reactions(
    uri,
    user,
    password,
    target_smiles,
    similarity_threshold,
    min_temp=None,
    max_temp=None,
    min_salinity=None,
    max_salinity=None,
    min_lat=None,
    max_lat=None,
    min_lon=None,
    max_lon=None,
    limit=1000,
):
    def get_similar_compounds(tx):
        query = (
            "MATCH (target:Compound {smiles: $target_smiles}) "
            "MATCH (target)-[sim:CHEMICAL_SIMILARITY]->(c:Compound)-[:SUBSTRATE_OF]->(r:Reaction)<-[:CATALYZES]-(p:Protein) "
            "WHERE sim.distance <= $similarity_threshold "
            "OPTIONAL MATCH (m:Genome)-[:CONTAINS]->(p) "
            "OPTIONAL MATCH (m)-[:ORIGINATED_FROM]->(s:Sample) "
            "WHERE "
            "   ($min_temp IS NULL OR s.temperature >= $min_temp) AND "
            "   ($max_temp IS NULL OR s.temperature <= $max_temp) AND "
            "   ($min_salinity IS NULL OR s.salinity >= $min_salinity) AND "
            "   ($max_salinity IS NULL OR s.salinity <= $max_salinity) AND "
            "   ($min_lat IS NULL OR s.latitude >= $min_lat) AND "
            "   ($max_lat IS NULL OR s.latitude <= $max_lat) AND "
            "   ($min_lon IS NULL OR s.longitude >= $min_lon) AND "
            "   ($max_lon IS NULL OR s.longitude <= $max_lon) "
            "RETURN "
            "   target.compound_id AS target_compound_id, "
            "   target.name AS target_compound_name, "
            "   c.compound_id AS similar_compound_id, "
            "   c.name AS similar_compound_name, "
            "   c.smiles AS similar_compound_smiles, "
            "   sim.distance AS similarity_distance, "
            "   r.reaction_id AS reaction_id, "
            "   r.name AS reaction_name, "
            "   p.protein_id AS protein_id, "
            "   p.name AS protein_name, "
            "   m.genome_id AS genome_id, "
            "   s.biosample_id AS sample_id, "
            "   s.temperature AS sample_temperature, "
            "   s.salinity AS sample_salinity, "
            "   s.latitude AS sample_latitude, "
            "   s.longitude AS sample_longitude "
            "ORDER BY similarity_distance ASC "
            "LIMIT $limit"
        )
        result = tx.run(
            query,
            target_smiles=target_smiles,
            similarity_threshold=similarity_threshold,
            min_temp=min_temp,
            max_temp=max_temp,
            min_salinity=min_salinity,
            max_salinity=max_salinity,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            limit=limit,
        )
        return [dict(record) for record in result]

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        similar_compounds = session.execute_read(get_similar_compounds)

    driver.close()
    return similar_compounds
