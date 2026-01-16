# Sesión 02 — Embeddings & RAG (guía + actividad)

## Objetivo de la sesión
Al final de esta hora, deberías poder:
- explicar qué es un **embedding** y para qué sirve,
- generar embeddings para textos,
- medir similitud (cosine / dot product) y recuperar *top-k* fragmentos,
- construir un **mini-RAG**: *buscar primero, responder después*.

> Esta sesión es **fundacional**: lo aprendido aplica a muchos casos de uso (chatbots internos, búsqueda semántica, clasificación asistida, etc.).

---

## 1) Conceptos clave (10–15 min)

### 1.1 ¿Qué es un embedding?
Un embedding es una **representación numérica** del significado de un texto.
- es un **vector** (lista de números),
- vive en un **espacio vectorial**,
- textos con significado parecido tienden a quedar **cerca** en ese espacio.

**Regla importante:** solo puedes comparar embeddings si fueron generados con el **mismo modelo** (mismo espacio).

### 1.2 Similitud: cosine similarity y dot product
La comparación más común es **cosine similarity** (intuición: “ángulo entre vectores”).

Muchos modelos de embeddings producen vectores **normalizados**, y entonces:
- cosine similarity se simplifica a un **dot product** (producto punto)
- esto es eficiente y escala bien.

### 1.3 ¿Qué es RAG?
**RAG = Retrieval-Augmented Generation**

En vez de pedirle al LLM que “sepa” tus documentos, tú haces:
1) **Retrieval**: encuentras los fragmentos relevantes (embeddings + búsqueda)
2) **Generation**: le pasas esos fragmentos al LLM para que responda usando ese contexto

Frase útil:
> **Search first, then answer.**

RAG no es fine-tuning: no cambia el modelo, cambia el **contexto**.

---

## 2) Actividad guiada (35–45 min)

### Preparación
Asegúrate de tener:
- entorno activo (`.venv`),
- `.env` con `OPENAI_API_KEY`,
- el repo abierto en VS Code.


### Paso A — Generar embeddings (10 min)
Ejecuta:
```bash
python examples/01_generate_embeddings.py
```

Qué deberías ver:
- impresión del tamaño del vector,
- vista previa de los primeros valores,
- confirmación de que la API funciona para embeddings.


### Paso B — Búsqueda semántica simple (15 min)
Ejecuta:
```bash
python examples/02_similarity_search.py
```

Este script:
1) lee varios documentos cortos,
2) los parte en fragmentos (chunking simple),
3) genera embeddings por fragmento,
4) genera embedding de una pregunta,
5) calcula similitudes y muestra el top-k.

Qué observar:
- los fragmentos recuperados deberían “tener sentido” con tu pregunta,
- el top-1 y top-3 deberían ser razonables.


### Paso C — Mini-RAG (10–15 min)
Ejecuta:
```bash
python examples/03_mini_rag.py
```

Este script hace lo mismo que el Paso B, pero además:
- arma un prompt que incluye los fragmentos recuperados,
- le pide al LLM responder usando **solo ese contexto**,
- imprime la respuesta.

Qué observar:
- la respuesta debería referenciar el contenido recuperado,
- si la pregunta no está cubierta por el contexto, el modelo debería decirlo (según la instrucción).

---

## 3) Ejercicios rápidos (si queda tiempo)

1) Cambia `TOP_K` de 3 a 5 y compara resultados.
2) Cambia el chunking (p.ej. por párrafos vs. tamaño fijo) y mira cómo cambia el top-k.
3) Escribe una pregunta “difícil” (poca evidencia en los docs) y observa cómo se comporta el mini-RAG.

---

## 4) Puente a las siguientes sesiones

Lo que viene después suele ser:
- chunking más robusto (tokens, solapamiento),
- almacenar embeddings (archivos / DB vectorial),
- trazabilidad (citas, IDs de chunks),
- evaluación: ¿recuperamos lo correcto? ¿la respuesta usa el contexto?

Este repositorio ya te deja con el núcleo técnico para avanzar.
