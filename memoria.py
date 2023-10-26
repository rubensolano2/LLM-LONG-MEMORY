from neo4j import GraphDatabase
from claves import neo4j
from claves import uri

class VeraDatabaseManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def iniciar_conversacion(self, fecha, sentimiento, texto_vera, texto_usuario, rank):
        with self.driver.session() as session:
            return session.execute_write(self._iniciar_conversacion, fecha, sentimiento, texto_vera, texto_usuario, rank)

    def _iniciar_conversacion(self, tx, fecha, sentimiento, texto_vera, texto_usuario, rank):
        query = """
        CREATE (c:Conversacion {
            fecha: $fecha,
            sentimiento: $sentimiento,
            vera: $texto_vera,
            usuario: $texto_usuario,
            rank: $rank
        })
        RETURN id(c) as element_id
        """
        result = tx.run(query, fecha=fecha, sentimiento=sentimiento, texto_vera=texto_vera, texto_usuario=texto_usuario, rank=rank)
        return result.single()['element_id']

    def agregar_tematica(self, conversacion_element_id, tematica):
        with self.driver.session() as session:
            return session.execute_write(self._agregar_tematica, conversacion_element_id, tematica)

    def _agregar_tematica(self, tx, conversacion_element_id, tematica):
        query = """
        MATCH (c:Conversacion) WHERE id(c) = $conversacion_element_id
        CREATE (t:Tematica {tematica: $tematica})-[:TEMATICA_DE]->(c)
        RETURN t
        """
        result = tx.run(query, conversacion_element_id=conversacion_element_id, tematica=tematica).single()
        return result[0] if result else None

    def agregar_resumen(self, conversacion_element_id, resumen):
        with self.driver.session() as session:
            return session.execute_write(self._agregar_resumen, conversacion_element_id, resumen)

    def _agregar_resumen(self, tx, conversacion_element_id, resumen):
        query = """
        MATCH (c:Conversacion) WHERE id(c) = $conversacion_element_id
        CREATE (r:Resumen {resumen: $resumen})-[:RESUMEN_DE]->(c)
        RETURN r
        """
        result = tx.run(query, conversacion_element_id=conversacion_element_id, resumen=resumen)
        return result.single()[0] if result else None

    def finalizar_conversacion(self, conversacion_element_id, fecha_fin):
        with self.driver.session() as session:
            session.execute_write(self._finalizar_conversacion, conversacion_element_id, fecha_fin)

    def _finalizar_conversacion(self, tx, conversacion_element_id, fecha_fin):
        query = """
        MATCH (c:Conversacion) WHERE id(c) = $conversacion_element_id
        SET c.fecha_fin = $fecha_fin
        """
        tx.run(query, conversacion_element_id=conversacion_element_id, fecha_fin=fecha_fin)


vera_db_manager = VeraDatabaseManager(uri=uri, user="neo4j", password=neo4j)

