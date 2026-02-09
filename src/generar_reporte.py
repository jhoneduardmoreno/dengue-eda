"""
Generador de Reporte PDF: Análisis Exploratorio de Datos - Dengue en Colombia.

Ensambla un PDF profesional con las gráficas de results_graphs/comparative/
y texto de análisis epidemiológico interpretativo.

Ejecución:
    python src/generar_reporte.py

Genera:
    reporte_dengue_colombia.pdf (en la raíz del proyecto)
"""

import os
from pathlib import Path
from fpdf import FPDF


# ============================================================================
# Rutas y constantes
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPHS_DIR = PROJECT_ROOT / "results_graphs" / "comparative"
OUTPUT_PDF = PROJECT_ROOT / "reporte_dengue_colombia.pdf"

GRAFICOS = [
    "comparativo dengue regular vs grave por año.png",
    "curva epidemica dengue regular vs grave.png",
    "casos por departamento dengue regular vs grave.png",
    "proporcion dengue grave por departamento y años.png",
    "distribucion grupo etario dengue regular vs grave.png",
    "plan estacional promedio.png",
    "tasa de incidencia.png",
    "casos vs temperatura.png",
    "correalacion dengue vs variables climaticas.png",
    "visualizacion datos nulos.png",
]

# Colores (RGB)
COLOR_PRIMARIO = (0, 71, 133)       # Azul oscuro
COLOR_SECUNDARIO = (41, 128, 185)   # Azul medio
COLOR_FONDO_SECCION = (230, 240, 250)
COLOR_TEXTO = (33, 33, 33)
COLOR_GRIS = (120, 120, 120)


# ============================================================================
# Textos del reporte
# ============================================================================
TEXTOS = {
    # --- Portada ---
    "titulo_principal": "Análisis Exploratorio de Datos:\nDengue en Colombia",
    "universidad": "Universidad Autónoma de Occidente",
    "maestria": "Maestría en Inteligencia Artificial",
    "materia": "Desarrollo de Soluciones con IA",
    "autores": [
        "Jhon Edwar Salazar",
        "Santiago Castaño Orozco",
        "David Alejandro Burbano Getial",
    ],
    "fecha": "Febrero 2025",

    # --- Sección 1: Problema y contexto ---
    "contexto_titulo": "1. Problema y Contexto",
    "contexto_texto": (
        "El dengue constituye uno de los principales problemas de salud pública en "
        "Colombia. Como enfermedad transmitida por el mosquito Aedes aegypti, su "
        "dinámica está estrechamente vinculada a factores climáticos, geográficos y "
        "socioeconómicos que favorecen la proliferación del vector. Colombia, por su "
        "ubicación tropical y la diversidad de pisos térmicos, presenta condiciones "
        "ideales para la transmisión del virus en gran parte de su territorio, "
        "especialmente en zonas por debajo de los 1.800 metros sobre el nivel del mar.\n\n"
        "El Sistema Nacional de Vigilancia en Salud Pública (SIVIGILA), operado por el "
        "Instituto Nacional de Salud (INS), registra de manera sistemática los casos de "
        "dengue en el país mediante los eventos 210 (dengue) y 220 (dengue grave). Estos "
        "datos, disponibles por semana epidemiológica, permiten analizar la evolución "
        "temporal, la distribución geográfica y las características demográficas de la "
        "enfermedad.\n\n"
        "El presente análisis exploratorio de datos (EDA) busca integrar la información "
        "de SIVIGILA con proyecciones poblacionales del DANE y variables climáticas "
        "obtenidas de Google Earth Engine (temperatura, precipitación, humedad) para "
        "comprender los patrones del dengue en Colombia en los años 2010, 2016, 2019, "
        "2022 y 2024. Estos años fueron seleccionados por representar contextos "
        "epidemiológicos diversos: años de brotes intensos, periodos interepidémicos y "
        "la dinámica post-pandemia de COVID-19."
    ),

    # --- Sección 2: Pregunta de negocio ---
    "pregunta_titulo": "2. Pregunta de Negocio",
    "pregunta_texto": (
        "¿Cuáles son los patrones espaciales, temporales y demográficos del dengue "
        "en Colombia, y qué relación existe entre las variables climáticas "
        "(temperatura, precipitación y humedad) y la incidencia de la enfermedad?\n\n"
        "Esta pregunta guía el análisis exploratorio y permite identificar factores "
        "de riesgo, poblaciones vulnerables y dinámicas estacionales que podrían "
        "informar estrategias de prevención y control vectorial basadas en evidencia."
    ),

    # --- Sección 3: Descripción de datos ---
    "datos_titulo": "3. Descripción de los Datos",
    "datos_texto": (
        "El análisis integra tres fuentes de datos complementarias:\n\n"
        "Datos epidemiológicos (SIVIGILA): Registros individuales de casos de dengue "
        "(evento 210) y dengue grave (evento 220) para los años 2010, 2016, 2019, "
        "2022 y 2024. Cada registro contiene información demográfica del paciente "
        "(edad, sexo, etnia), ubicación geográfica (departamento, municipio), "
        "clasificación del caso, fechas de notificación e inicio de síntomas, y "
        "estado final del caso. El dengue regular (cod. 210) cuenta con datos para "
        "2010, 2016, 2022 y 2024, mientras que el dengue grave (cod. 220) incluye "
        "adicionalmente el año 2019.\n\n"
        "Proyecciones poblacionales (DANE): Proyecciones de población a nivel "
        "municipal para el periodo 2005-2020, utilizadas para calcular tasas de "
        "incidencia por cada 100.000 habitantes, lo que permite comparaciones "
        "normalizadas entre departamentos de distinto tamaño poblacional.\n\n"
        "Variables climáticas (Google Earth Engine): Datos de temperatura superficial, "
        "precipitación acumulada y humedad relativa a nivel departamental, extraídos "
        "de sensores satelitales mediante la plataforma Google Earth Engine. Estas "
        "variables permiten explorar la correlación entre condiciones climáticas y la "
        "dinámica de transmisión del dengue."
    ),

    # --- Sección 4: Análisis EDA (interpretaciones por gráfico) ---
    "eda_titulo": "4. Análisis Exploratorio de Datos",
    "eda_intro": (
        "A continuación se presenta el análisis detallado de los principales hallazgos, "
        "organizados en diez visualizaciones que cubren aspectos de volumen, "
        "estacionalidad, distribución geográfica, perfil demográfico, incidencia "
        "normalizada, factores climáticos y calidad de los datos."
    ),

    "analisis": [
        # 1. Comparativo regular vs grave por año
        {
            "subtitulo": "4.1 Comparativo Dengue Regular vs Grave por Año",
            "texto": (
                "Esta visualización presenta el volumen total de casos de dengue "
                "regular y dengue grave reportados en cada año de estudio. Se observa "
                "una marcada variabilidad interanual en el número de casos, reflejo "
                "de los ciclos epidémicos característicos del dengue en regiones "
                "endémicas.\n\n"
                "El año 2010 destaca como un periodo de intensa actividad epidémica, "
                "con un volumen significativamente mayor de casos tanto de dengue "
                "regular como grave. Por otro lado, los años 2016 y 2022 presentan "
                "niveles intermedios, mientras que 2024 muestra un repunte que "
                "podría estar asociado a condiciones climáticas favorables para el "
                "vector. La proporción de casos graves respecto al total se mantiene "
                "relativamente estable a lo largo de los años, lo que sugiere que "
                "la severidad de la enfermedad no ha cambiado significativamente en "
                "el periodo analizado.\n\n"
                "Este panorama general permite contextualizar los análisis "
                "posteriores y resalta la importancia de mantener sistemas de "
                "vigilancia robustos en años tanto epidémicos como interepidémicos."
            ),
        },
        # 2. Curva epidémica
        {
            "subtitulo": "4.2 Curva Epidémica: Dengue Regular vs Grave",
            "texto": (
                "La curva epidémica muestra la distribución de casos por semana "
                "epidemiológica, permitiendo identificar patrones de estacionalidad. "
                "Se aprecia un patrón bimodal en varios años, con picos que "
                "frecuentemente coinciden con las temporadas de lluvia en Colombia "
                "(abril-junio y octubre-noviembre), periodos en los que las "
                "condiciones de temperatura y humedad favorecen la proliferación "
                "del Aedes aegypti.\n\n"
                "La comparación entre dengue regular y grave revela que los picos "
                "de casos graves tienden a seguir, con un rezago de pocas semanas, "
                "a los picos del dengue regular. Este comportamiento es "
                "epidemiológicamente consistente, ya que el dengue grave suele "
                "manifestarse como complicación del cuadro inicial. La intensidad "
                "de los picos varía entre años, con 2010 presentando las mayores "
                "amplitudes.\n\n"
                "El análisis de la curva epidémica es fundamental para la "
                "planificación de recursos hospitalarios y campañas de control "
                "vectorial preventivas."
            ),
        },
        # 3. Casos por departamento
        {
            "subtitulo": "4.3 Casos por Departamento: Regular vs Grave",
            "texto": (
                "La distribución geográfica de casos evidencia una alta "
                "concentración en departamentos del Valle del Cauca, "
                "Santander, Norte de Santander, Tolima y Meta, que "
                "consistentemente reportan los mayores volúmenes de casos. "
                "Estos departamentos comparten características propicias para "
                "la transmisión: altitudes bajas, temperaturas elevadas, "
                "urbanización creciente y, en algunos casos, problemas de "
                "acceso a agua potable que favorecen el almacenamiento de "
                "agua y la creación de criaderos del vector.\n\n"
                "La comparación entre dengue regular y grave por departamento "
                "muestra que la distribución geográfica de casos graves no "
                "siempre es proporcional al volumen de casos regulares. "
                "Algunos departamentos presentan proporciones de gravedad "
                "superiores al promedio nacional, lo que podría estar "
                "relacionado con factores como serotipos circulantes, acceso "
                "a servicios de salud o comorbilidades de la población.\n\n"
                "Esta información es esencial para la focalización de "
                "intervenciones y la asignación eficiente de recursos en "
                "las regiones más afectadas."
            ),
        },
        # 4. Proporción grave por departamento y año
        {
            "subtitulo": "4.4 Proporción de Dengue Grave por Departamento y Año",
            "texto": (
                "El mapa de calor de la proporción de dengue grave por "
                "departamento y año permite identificar patrones "
                "espaciotemporales de severidad. Se observa que ciertos "
                "departamentos mantienen proporciones de gravedad "
                "consistentemente altas a lo largo del periodo, mientras "
                "que otros muestran picos aislados que podrían corresponder "
                "a cambios en los serotipos circulantes o a mejoras en la "
                "capacidad diagnóstica.\n\n"
                "Los departamentos con proporciones de gravedad elevadas de "
                "forma sostenida merecen atención especial en términos de "
                "fortalecimiento de la capacidad hospitalaria y formación "
                "del personal de salud en manejo clínico del dengue grave. "
                "Las variaciones interanuales sugieren que factores como la "
                "inmunidad poblacional y la circulación de serotipos juegan "
                "un rol importante en la proporción de casos que evolucionan "
                "a formas graves.\n\n"
                "Este análisis combinado de espacio y tiempo ofrece una "
                "perspectiva más completa que el análisis de cada dimensión "
                "por separado."
            ),
        },
        # 5. Distribución grupo etario
        {
            "subtitulo": "4.5 Distribución por Grupo Etario: Regular vs Grave",
            "texto": (
                "El perfil demográfico por grupos de edad revela diferencias "
                "importantes entre el dengue regular y el dengue grave. "
                "El dengue regular afecta predominantemente a adultos jóvenes "
                "(15-29 años), grupo que presenta mayor exposición al vector "
                "por su movilidad y actividades laborales o educativas en "
                "zonas urbanas.\n\n"
                "En contraste, el dengue grave muestra una distribución más "
                "uniforme entre grupos etarios, con una proporción "
                "relativamente mayor en niños (0-14 años) y adultos mayores "
                "(60+). Estos grupos son particularmente vulnerables: los "
                "niños por la posibilidad de infecciones secundarias con "
                "serotipos distintos y los adultos mayores por comorbilidades "
                "que complican el manejo clínico.\n\n"
                "Esta caracterización demográfica es crucial para diseñar "
                "estrategias de comunicación diferenciadas y protocolos "
                "de atención temprana focalizados en las poblaciones de "
                "mayor riesgo."
            ),
        },
        # 6. Patrón estacional promedio
        {
            "subtitulo": "4.6 Patrón Estacional Promedio",
            "texto": (
                "El patrón estacional promedio consolida la información de "
                "todos los años estudiados para identificar el ciclo anual "
                "típico del dengue en Colombia. Se observa que los casos "
                "tienden a incrementarse a partir de las semanas 10-12 "
                "(marzo), alcanzando un primer pico entre las semanas 16-22 "
                "(abril-mayo), seguido de un descenso parcial y un segundo "
                "pico hacia las semanas 40-48 (octubre-noviembre).\n\n"
                "Este patrón bimodal está estrechamente relacionado con el "
                "régimen de lluvias bimodal de la región andina y del "
                "interior de Colombia. Las temporadas húmedas proveen las "
                "condiciones necesarias para la oviposición y desarrollo "
                "larvario del Aedes aegypti, generando un incremento en la "
                "densidad del vector y, consecuentemente, en la transmisión "
                "del virus.\n\n"
                "La identificación de este patrón promedio permite establecer "
                "ventanas temporales óptimas para intervenciones preventivas, "
                "como jornadas de eliminación de criaderos y campañas de "
                "fumigación, idealmente 4-6 semanas antes de los picos "
                "esperados."
            ),
        },
        # 7. Tasa de incidencia
        {
            "subtitulo": "4.7 Tasa de Incidencia por Departamento",
            "texto": (
                "La tasa de incidencia, expresada como casos por cada "
                "100.000 habitantes, ofrece una medida normalizada que "
                "permite comparaciones justas entre departamentos de "
                "distinto tamaño poblacional. A diferencia del conteo "
                "absoluto de casos, esta métrica revela que algunos "
                "departamentos con menor población pero alta transmisión "
                "presentan las tasas más elevadas del país.\n\n"
                "Departamentos como Arauca, Meta, Casanare y Norte de "
                "Santander frecuentemente lideran las tasas de incidencia, "
                "reflejo de condiciones ecológicas y socioeconómicas "
                "particularmente favorables para la transmisión. En "
                "contraste, departamentos con grandes centros urbanos "
                "como Bogotá (ubicado a más de 2.600 msnm) presentan "
                "tasas mínimas o nulas dada la ausencia del vector en "
                "altitudes elevadas.\n\n"
                "La tasa de incidencia es el indicador recomendado por "
                "la OMS para la comparación entre regiones y la "
                "priorización de intervenciones en salud pública."
            ),
        },
        # 8. Casos vs temperatura
        {
            "subtitulo": "4.8 Casos de Dengue vs Temperatura",
            "texto": (
                "La relación entre temperatura y número de casos de "
                "dengue es un factor epidemiológico bien documentado. "
                "Esta visualización explora la asociación entre la "
                "temperatura media y la incidencia de dengue a nivel "
                "departamental. Se observa una tendencia positiva: "
                "departamentos con temperaturas medias más elevadas "
                "tienden a reportar mayor número de casos.\n\n"
                "El rango óptimo de temperatura para el desarrollo del "
                "Aedes aegypti se sitúa entre 25°C y 30°C, lo cual "
                "coincide con las condiciones climáticas de las zonas "
                "bajas y cálidas de Colombia. Temperaturas por debajo "
                "de 20°C limitan significativamente la supervivencia y "
                "capacidad vectorial del mosquito, explicando la ausencia "
                "de transmisión en zonas andinas de mayor altitud.\n\n"
                "Esta evidencia respalda la importancia de integrar "
                "variables climáticas en los modelos predictivos de "
                "dengue y en los sistemas de alerta temprana."
            ),
        },
        # 9. Correlación variables climáticas
        {
            "subtitulo": "4.9 Correlación del Dengue con Variables Climáticas",
            "texto": (
                "La matriz de correlación entre casos de dengue y "
                "variables climáticas (temperatura, precipitación y "
                "humedad relativa) cuantifica la fuerza y dirección de "
                "estas asociaciones. La temperatura presenta la "
                "correlación positiva más fuerte con la incidencia de "
                "dengue, seguida por la precipitación acumulada.\n\n"
                "La humedad relativa muestra una relación más compleja, "
                "con correlaciones que varían según la región y el "
                "periodo analizado. Valores extremos de humedad (tanto "
                "muy altos como muy bajos) pueden limitar la "
                "transmisión: la sequedad extrema reduce los criaderos "
                "disponibles, mientras que lluvias torrenciales pueden "
                "arrastrar las larvas.\n\n"
                "Estos hallazgos son consistentes con la literatura "
                "internacional y refuerzan la relevancia de monitorear "
                "múltiples variables climáticas de forma simultánea "
                "para anticipar brotes de dengue. La integración de "
                "datos satelitales de Google Earth Engine permite "
                "realizar este análisis con cobertura nacional y "
                "resolución temporal adecuada."
            ),
        },
        # 10. Visualización datos nulos
        {
            "subtitulo": "4.10 Calidad de Datos: Visualización de Datos Nulos",
            "texto": (
                "La evaluación de la calidad de los datos es un paso "
                "fundamental en cualquier análisis exploratorio. Esta "
                "visualización muestra el patrón de valores nulos "
                "(datos faltantes) en las diferentes variables del "
                "dataset consolidado. Se identifican variables con "
                "alta completitud (como año, semana, departamento) "
                "junto a otras con proporciones significativas de "
                "valores faltantes.\n\n"
                "Las variables con mayor proporción de nulos suelen "
                "corresponder a campos de registro clínico cuyo "
                "diligenciamiento depende de la evolución del caso "
                "(como fecha de hospitalización o fecha de defunción) "
                "o a campos que fueron introducidos o modificados en "
                "versiones posteriores de la ficha de notificación "
                "de SIVIGILA.\n\n"
                "El conocimiento de estos patrones de datos faltantes "
                "es esencial para interpretar correctamente los "
                "resultados del análisis y para tomar decisiones "
                "informadas sobre estrategias de imputación o "
                "exclusión de variables en análisis posteriores. "
                "Para este EDA, las variables con alta completitud "
                "fueron priorizadas en el análisis."
            ),
        },
    ],

    # --- Sección 5: Conclusiones ---
    "conclusiones_titulo": "5. Conclusiones",
    "conclusiones_texto": (
        "El análisis exploratorio de datos del dengue en Colombia para los años "
        "2010, 2016, 2019, 2022 y 2024 permite establecer las siguientes "
        "conclusiones principales:\n\n"
        "1. Variabilidad interanual significativa: El dengue en Colombia presenta "
        "ciclos epidémicos marcados, con años de alta incidencia (como 2010 y 2024) "
        "alternados con periodos de menor actividad. Esta variabilidad responde a la "
        "interacción de factores como la inmunidad poblacional, los serotipos "
        "circulantes y las condiciones climáticas.\n\n"
        "2. Estacionalidad definida: Se identificó un patrón estacional bimodal "
        "consistente con el régimen de lluvias colombiano, con picos de transmisión "
        "durante las temporadas húmedas. Este patrón ofrece ventanas de oportunidad "
        "para intervenciones preventivas focalizadas.\n\n"
        "3. Concentración geográfica: Los casos se concentran en departamentos de "
        "tierras bajas y cálidas, con el Valle del Cauca, Santander, Norte de "
        "Santander, Tolima y Meta como los territorios más afectados. Las tasas de "
        "incidencia normalizadas revelan que departamentos menos poblados de la "
        "Orinoquía y el nororiente pueden tener cargas relativas superiores.\n\n"
        "4. Perfil demográfico diferencial: Mientras el dengue regular afecta "
        "principalmente a adultos jóvenes, el dengue grave impacta de forma "
        "proporcionalmente mayor a niños y adultos mayores, poblaciones que "
        "requieren vigilancia y manejo clínico especializados.\n\n"
        "5. Relación clima-dengue: La temperatura es la variable climática con mayor "
        "asociación con la incidencia del dengue. La integración de datos climáticos "
        "satelitales de Google Earth Engine demostró ser una herramienta valiosa para "
        "el análisis epidemiológico a escala nacional.\n\n"
        "Estos hallazgos proporcionan una base sólida para el desarrollo de modelos "
        "predictivos y sistemas de alerta temprana que integren variables "
        "epidemiológicas y climáticas para la prevención del dengue en Colombia."
    ),

    # --- Sección 6: Repositorios ---
    "repositorios_titulo": "6. Repositorios",
    "repositorios_texto": (
        "El código fuente, los notebooks de análisis y las visualizaciones generadas "
        "se encuentran disponibles en el repositorio Git del proyecto. Este repositorio "
        "incluye:\n\n"
        "- Notebooks de Jupyter con el análisis exploratorio completo\n"
        "- Scripts de Python para procesamiento de datos y generación de reportes\n"
        "- Datos climáticos procesados desde Google Earth Engine\n"
        "- Gráficas generadas en formato PNG de alta resolución\n\n"
        "El repositorio sigue una estructura organizada que facilita la "
        "reproducibilidad del análisis y la colaboración entre miembros del equipo."
    ),

    # --- Sección 7: Trabajo en equipo ---
    "equipo_titulo": "7. Trabajo en Equipo",
    "equipo_intro": (
        "A continuación se detalla la distribución de responsabilidades entre los "
        "miembros del equipo para el desarrollo de este proyecto:"
    ),
    "equipo_tabla": [
        ["Integrante", "Responsabilidades"],
        ["Jhon Edwar Salazar", "Recolección de datos, EDA dengue regular y grave, gráficos comparativos, generación del reporte"],
        ["Santiago Castaño Orozco", "Extracción de variables climáticas (GEE), análisis de correlación clima-dengue"],
        ["David Alejandro Burbano Getial", "Datos DANE, cálculo de tasas de incidencia, análisis demográfico"],
    ],
}


# ============================================================================
# Clase ReporteDengue
# ============================================================================
class ReporteDengue(FPDF):
    """Generador de PDF para el reporte de Dengue en Colombia."""

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=25)
        self.set_margins(left=20, top=20, right=20)

    # ----- Header / Footer -----
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*COLOR_GRIS)
        self.cell(
            0, 8,
            "Análisis Exploratorio de Datos: Dengue en Colombia",
            align="L",
        )
        self.ln(4)
        self.set_draw_color(*COLOR_SECUNDARIO)
        self.set_line_width(0.3)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(6)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*COLOR_GRIS)
        self.cell(0, 10, f"Página {self.page_no() - 1}", align="C")

    # ----- Portada -----
    def portada(self):
        self.add_page()

        # Banda decorativa superior
        self.set_fill_color(*COLOR_PRIMARIO)
        self.rect(0, 0, 210, 8, "F")

        # Título principal
        self.ln(45)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(*COLOR_PRIMARIO)
        self.multi_cell(0, 14, TEXTOS["titulo_principal"], align="C")

        # Línea decorativa
        self.ln(8)
        self.set_draw_color(*COLOR_SECUNDARIO)
        self.set_line_width(1)
        self.line(50, self.get_y(), 160, self.get_y())
        self.ln(12)

        # Universidad, maestría, materia
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*COLOR_TEXTO)
        self.cell(0, 8, TEXTOS["universidad"], align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "I", 12)
        self.cell(0, 8, TEXTOS["maestria"], align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, TEXTOS["materia"], align="C", new_x="LMARGIN", new_y="NEXT")

        # Autores
        self.ln(20)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, "Autores:", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 12)
        for autor in TEXTOS["autores"]:
            self.cell(0, 7, autor, align="C", new_x="LMARGIN", new_y="NEXT")

        # Fecha
        self.ln(15)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(*COLOR_GRIS)
        self.cell(0, 8, TEXTOS["fecha"], align="C")

        # Banda decorativa inferior
        self.set_fill_color(*COLOR_PRIMARIO)
        self.rect(0, 289, 210, 8, "F")

    # ----- Componentes reutilizables -----
    def titulo_seccion(self, texto):
        """Título H1 con fondo de color."""
        self.ln(6)
        self.set_fill_color(*COLOR_FONDO_SECCION)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*COLOR_PRIMARIO)
        self.cell(0, 12, f"  {texto}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def subtitulo(self, texto):
        """Título H2."""
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*COLOR_SECUNDARIO)
        self.cell(0, 9, texto, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def parrafo(self, texto):
        """Texto justificado."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*COLOR_TEXTO)
        self.multi_cell(0, 5.5, texto, align="J")
        self.ln(2)

    def agregar_grafico(self, nombre_archivo, subtitulo_texto, analisis_texto):
        """Agrega un gráfico centrado con su título y análisis."""
        ruta_imagen = GRAPHS_DIR / nombre_archivo
        if not ruta_imagen.exists():
            self.parrafo(f"[Imagen no encontrada: {nombre_archivo}]")
            return

        # Subtítulo
        self.subtitulo(subtitulo_texto)

        # Verificar si queda espacio suficiente para la imagen
        espacio_disponible = 297 - self.get_y() - 25  # margen inferior
        if espacio_disponible < 80:
            self.add_page()

        # Insertar imagen centrada (ancho útil ~170mm)
        ancho_imagen = 165
        x_imagen = (210 - ancho_imagen) / 2
        self.image(str(ruta_imagen), x=x_imagen, w=ancho_imagen)
        self.ln(6)

        # Texto de análisis
        self.parrafo(analisis_texto)

    def agregar_tabla(self, datos):
        """Tabla simple con encabezado destacado."""
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(*COLOR_PRIMARIO)
        self.set_text_color(255, 255, 255)

        # Anchos de columna
        col_widths = [55, 115]

        # Encabezado
        for i, header in enumerate(datos[0]):
            self.cell(col_widths[i], 8, f"  {header}", border=1, fill=True)
        self.ln()

        # Filas de datos
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*COLOR_TEXTO)
        fill = False
        for fila in datos[1:]:
            if fill:
                self.set_fill_color(245, 248, 252)
            else:
                self.set_fill_color(255, 255, 255)

            x_start = self.get_x()
            y_start = self.get_y()

            # Calcular la altura necesaria para la celda de responsabilidades
            # usando multi_cell ficticio
            self.set_font("Helvetica", "", 9)
            # Medir alto del texto largo
            n_lines = len(self.multi_cell(
                col_widths[1] - 2, 5.5, fila[1],
                align="L", dry_run=True, output="LINES",
            ))
            row_height = max(8, n_lines * 5.5 + 2)

            # Dibujar primera columna (nombre)
            self.set_xy(x_start, y_start)
            self.set_font("Helvetica", "B", 9)
            self.cell(col_widths[0], row_height, f"  {fila[0]}", border=1, fill=fill)

            # Dibujar segunda columna (responsabilidades)
            self.set_xy(x_start + col_widths[0], y_start)
            self.set_font("Helvetica", "", 9)
            # Borde manual
            self.rect(x_start + col_widths[0], y_start, col_widths[1], row_height)
            if fill:
                self.set_fill_color(245, 248, 252)
                self.rect(
                    x_start + col_widths[0] + 0.2, y_start + 0.2,
                    col_widths[1] - 0.4, row_height - 0.4, "F",
                )
            self.set_xy(x_start + col_widths[0] + 1, y_start + 1)
            self.multi_cell(col_widths[1] - 2, 5.5, fila[1], align="L")

            self.set_xy(x_start, y_start + row_height)
            fill = not fill


# ============================================================================
# Función principal
# ============================================================================
def main():
    print("Generando reporte PDF del Dengue en Colombia...")
    print(f"  Directorio de gráficos: {GRAPHS_DIR}")
    print(f"  Archivo de salida: {OUTPUT_PDF}")

    # Verificar que existen los gráficos
    faltantes = []
    for g in GRAFICOS:
        if not (GRAPHS_DIR / g).exists():
            faltantes.append(g)
    if faltantes:
        print("\n  ADVERTENCIA - Gráficos no encontrados:")
        for f in faltantes:
            print(f"    - {f}")
        print("  Se continuará sin estos gráficos.\n")

    pdf = ReporteDengue()

    # --- Portada ---
    print("  [1/7] Portada...")
    pdf.portada()

    # --- Sección 1: Problema y contexto ---
    print("  [2/7] Problema y contexto...")
    pdf.add_page()
    pdf.titulo_seccion(TEXTOS["contexto_titulo"])
    pdf.parrafo(TEXTOS["contexto_texto"])

    # --- Sección 2: Pregunta de negocio ---
    print("  [3/7] Pregunta de negocio...")
    pdf.titulo_seccion(TEXTOS["pregunta_titulo"])
    pdf.parrafo(TEXTOS["pregunta_texto"])

    # --- Sección 3: Descripción de datos ---
    print("  [4/7] Descripción de datos...")
    pdf.titulo_seccion(TEXTOS["datos_titulo"])
    pdf.parrafo(TEXTOS["datos_texto"])

    # --- Sección 4: Análisis EDA ---
    print("  [5/7] Análisis Exploratorio de Datos (10 gráficos)...")
    pdf.add_page()
    pdf.titulo_seccion(TEXTOS["eda_titulo"])
    pdf.parrafo(TEXTOS["eda_intro"])

    for i, (grafico, analisis) in enumerate(zip(GRAFICOS, TEXTOS["analisis"])):
        print(f"    Gráfico {i+1}/10: {grafico}")
        pdf.agregar_grafico(
            nombre_archivo=grafico,
            subtitulo_texto=analisis["subtitulo"],
            analisis_texto=analisis["texto"],
        )

    # --- Sección 5: Conclusiones ---
    print("  [6/7] Conclusiones...")
    pdf.add_page()
    pdf.titulo_seccion(TEXTOS["conclusiones_titulo"])
    pdf.parrafo(TEXTOS["conclusiones_texto"])

    # --- Sección 6: Repositorios ---
    pdf.titulo_seccion(TEXTOS["repositorios_titulo"])
    pdf.parrafo(TEXTOS["repositorios_texto"])

    # --- Sección 7: Trabajo en equipo ---
    print("  [7/7] Trabajo en equipo...")
    pdf.add_page()
    pdf.titulo_seccion(TEXTOS["equipo_titulo"])
    pdf.parrafo(TEXTOS["equipo_intro"])
    pdf.ln(4)
    pdf.agregar_tabla(TEXTOS["equipo_tabla"])

    # --- Generar PDF ---
    pdf.output(str(OUTPUT_PDF))
    print(f"\n  Reporte generado exitosamente: {OUTPUT_PDF}")
    print(f"  Tamaño: {OUTPUT_PDF.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
