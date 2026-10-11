# LLM Long Memory — Vera

Asistente de voz en Python que almacena conversaciones en Neo4j y recupera contexto mediante similitud textual y embeddings antes de generar una respuesta.

## Arquitectura

```text
Micrófono → transcripción → consulta y recuperación de contexto
                                      ↕
                                    Neo4j
                                      ↓
                      LLM → síntesis de voz → reproducción
```

## Código implementado

| Archivo | Responsabilidad |
| --- | --- |
| [MAIN-CODE.py](MAIN-CODE.py) | Grabación, transcripción, llamadas al LLM, síntesis de voz y reproducción de audio. |
| [memory.py](memory.py) | Escritura de conversaciones, temáticas y resúmenes en Neo4j. |
| [search_algorithm.py](search_algorithm.py) | Recuperación con TF-IDF, Sentence Transformers y similitud coseno. |

`TextComparer` combina los scores mediante `maximum(0.6 × TF-IDF, 0.4 × similitud semántica)`. Es una heurística de recuperación; no incluye un benchmark de relevancia ni garantiza recordar todos los detalles.

El esquema del código utiliza nodos `Conversacion`, `Tematica` y `Resumen`, con relaciones `TEMATICA_DE` y `RESUMEN_DE`. Las conversaciones persisten en la base de datos entre ejecuciones.

## Preparación

Este proyecto utiliza interfaces antiguas de los SDK de OpenAI y ElevenLabs (`openai.ChatCompletion`, `openai.Audio` e imports de ElevenLabs de su API anterior). Instalar las versiones actuales sin adaptar el código no es un procedimiento validado.

1. Crea un entorno Python aislado y proporciona una instancia de Neo4j.
2. Instala versiones compatibles de `openai`, `elevenlabs`, `neo4j`, `keyboard`, `sounddevice`, `numpy`, `scipy`, `requests`, `pygame`, `nltk`, `scikit-learn`, `sentence-transformers` y `pydub`. La conversión MP3 requiere FFmpeg disponible en el sistema. No se incluye un lockfile de dependencias.
3. Copia `claves.example.py` a `claves.py` y configura las variables de entorno indicadas. `claves.py` y los archivos `.env` quedan excluidos de Git.
4. Ejecuta `python MAIN-CODE.py` desde la raíz. La interacción se activa mediante `Ctrl + Alt`; requiere micrófono y salida de audio.

Las llamadas a los proveedores externos requieren credenciales propias y pueden tener coste. No incluyas conversaciones personales en commits ni en demostraciones públicas.

## Estado y evaluación

El repositorio conserva una implementación experimental de integración de voz, grafos y recuperación de contexto. No incluye pruebas integrales, mediciones de latencia ni evaluación de calidad de recuperación. Las demos históricas del README anterior no forman parte de un conjunto de evaluación reproducible.

## Siguientes mejoras

- Migrar los clientes de proveedores a SDK compatibles y bloquear dependencias.
- Separar grabación, recuperación y generación para probarlas de forma independiente.
- Evaluar la recuperación con consultas y conversaciones sintéticas.
- Gestionar errores de red, sesiones de base de datos y contextos vacíos.

## Licencia

[GNU GPL v3](LICENSE).
