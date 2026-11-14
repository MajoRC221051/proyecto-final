"""Interfaz del modelo final del Proyecto 2. Ejecutar: streamlit run app.py."""

from io import BytesIO
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted


MODEL_PATH = Path(__file__).resolve().parent / "model_svc_tfidf.joblib"
LABELS = {0: "No se detecta infracción", 1: "Posible infracción"}


@st.cache_resource(show_spinner="Cargando el modelo…")
def load_model(path: str, modified_ns: int, size: int) -> Pipeline:
    # La fecha y el tamaño invalidan la caché si se reemplaza el modelo.
    # Cargar únicamente el archivo propio exportado desde el notebook.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise ValueError("Se necesita el Pipeline completo, incluido TF-IDF.")
    if not isinstance(model.named_steps.get("tfidf"), TfidfVectorizer):
        raise ValueError("El Pipeline debe contener el paso 'tfidf'.")
    if not isinstance(model.named_steps.get("clf"), LinearSVC):
        raise ValueError("El clasificador debe ser el LinearSVC del notebook.")
    check_is_fitted(model.named_steps["tfidf"])
    check_is_fitted(model.named_steps["clf"])
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Las clases esperadas son 0 = no infracción y 1 = infracción.")
    model.predict(["Test comment"])
    return model


def predict_texts(model: Pipeline, texts: list[str]) -> pd.DataFrame:
    """Conserva el texto original: el Pipeline entrenado aplica TF-IDF."""
    labels = model.predict(texts).astype(int)
    margins = np.asarray(model.decision_function(texts)).reshape(-1)
    return pd.DataFrame(
        {
            "prediccion": labels,
            "resultado": [LABELS[int(label)] for label in labels],
            "margen_svm": margins,
        }
    )


def predict_csv(model: Pipeline, frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mantiene todas las filas y deja sin clasificar los textos vacíos."""
    output = frame.copy().reset_index(drop=True)
    texts = output[column].fillna("").astype(str)
    valid = texts.str.strip().ne("")
    # Evitar sobrescribir columnas que ya existen en el CSV original.
    generated = ["prediccion", "resultado", "margen_svm"]
    for name in generated:
        if name in output.columns:
            raise ValueError(f"Renombra la columna existente '{name}' antes de cargar el CSV.")
    output["prediccion"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["resultado"] = "Texto vacío: sin analizar"
    output["margen_svm"] = np.nan
    # Lotes pequeños para limitar el uso de memoria durante la inferencia.
    indices = output.index[valid]
    for start in range(0, len(indices), 500):
        selected = indices[start : start + 500]
        predictions = predict_texts(model, texts.loc[selected].tolist())
        for name in generated:
            output.loc[selected, name] = predictions[name].to_numpy()
    return output


def main() -> None:
    st.set_page_config(page_title="Proyecto 2 · Moderación", page_icon="💬", layout="wide")
    st.title("Moderación de comentarios")
    st.write("Clasificación de comentarios de Reddit con el modelo del Proyecto 2.")
    with st.sidebar:
        st.header("Proyecto 2")
        st.write("Daniel Machic · María José Ramírez")
        st.caption("Modelo: LinearSVC + TF-IDF")
        st.info("0: no se detecta infracción\n\n1: posible infracción")
        st.caption(
            "La predicción requiere revisión humana. Este modelo utiliza el comentario; "
            "no recibe una regla ni el contexto del subreddit como entradas. "
            "Usa preferentemente texto en el idioma del entrenamiento."
        )

    if not MODEL_PATH.is_file():
        st.warning("Falta el modelo entrenado: model_svc_tfidf.joblib")
        st.write(
            "Después de entrenar en tu notebook, exporta el Pipeline completo "
            "y guarda el archivo en la misma carpeta que app.py."
        )
        st.code('import joblib\njoblib.dump(best_svc, "model_svc_tfidf.joblib")', language="python")
        st.caption("En Colab, descarga el archivo desde el panel Archivos.")
        st.stop()

    try:
        stat = MODEL_PATH.stat()
        model = load_model(str(MODEL_PATH), stat.st_mtime_ns, stat.st_size)
    except InconsistentVersionWarning:
        st.error(
            "La versión de scikit-learn difiere de la usada al entrenar. "
            "Fija las versiones del notebook en requirements.txt y vuelve a desplegar."
        )
        st.stop()
    except Exception:
        st.error(
            "No se pudo cargar el Pipeline. Exporta best_svc desde el notebook y "
            "verifica que Python y las dependencias coincidan con el entrenamiento."
        )
        st.stop()

    single, batch = st.tabs(["Un comentario", "Archivo CSV"])
    with single:
        with st.form("comment_form"):
            comment = st.text_area(
                "Comentario", height=170,
                placeholder="Escribe el comentario que deseas analizar…",
            )
            submitted = st.form_submit_button("Analizar comentario", type="primary")
        if submitted:
            if not comment.strip():
                st.warning("Escribe un comentario antes de analizarlo.")
            else:
                try:
                    result = predict_texts(model, [comment]).iloc[0]
                    if result["prediccion"] == 1:
                        st.warning("Posible infracción de las reglas")
                    else:
                        st.success("No se detecta infracción")
                    st.metric("Margen de decisión SVM", f'{result["margen_svm"]:.3f}')
                    st.caption(
                        "Un margen positivo corresponde a la clase 1 y uno negativo a la clase 0. "
                        "Es una puntuación del modelo, no una probabilidad ni un porcentaje de confianza."
                    )
                except Exception:
                    st.error("No se pudo analizar el comentario. Verifica el modelo exportado.")

    with batch:
        st.write("Carga un CSV, selecciona la columna de comentarios y descarga los resultados.")
        uploaded = st.file_uploader("Archivo de comentarios", type=["csv"])
        if uploaded is not None:
            separator = st.selectbox("Separador", [",", ";", "\t"], format_func=lambda x: "Tabulación" if x == "\t" else x)
            try:
                frame = pd.read_csv(BytesIO(uploaded.getvalue()), sep=separator, encoding="utf-8-sig")
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
                st.error("No se pudo leer el CSV. Usa UTF-8 y verifica el separador.")
                return
            if frame.empty:
                st.warning("El CSV no contiene filas.")
                return
            if len(frame) > 50_000:
                st.warning("Carga como máximo 50,000 filas por archivo.")
                return
            columns = frame.columns.tolist()
            preferred = next((c for c in ["body", "text", "comment_text", "comment"] if c in columns), columns[0])
            column = st.selectbox("Columna de comentarios", columns, index=columns.index(preferred))
            st.dataframe(frame.head(10), hide_index=True)
            # Se recalcula con el archivo y la selección actuales: evita descargas obsoletas.
            if st.checkbox("Analizar este archivo"):
                try:
                    with st.spinner("Analizando comentarios…"):
                        results = predict_csv(model, frame, column)
                except ValueError as error:
                    st.error(str(error))
                    return
                except Exception:
                    st.error("No se pudo procesar el archivo. Verifica los datos y el modelo.")
                    return
                total = int(results["prediccion"].notna().sum())
                flagged = int(results["prediccion"].eq(1).sum())
                left, middle, right = st.columns(3)
                left.metric("Comentarios analizados", total)
                middle.metric("Posibles infracciones", flagged)
                right.metric("Filas vacías", len(results) - total)
                st.dataframe(results.head(100), hide_index=True)
                st.caption("Vista previa: primeras 100 filas. El CSV descargado incluye todas las filas.")
                st.download_button(
                    "Descargar predicciones",
                    data=results.to_csv(index=False).encode("utf-8-sig"),
                    file_name="predicciones.csv", mime="text/csv",
                )


if __name__ == "__main__":
    main()
