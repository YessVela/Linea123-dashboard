# ══════════════════════════════════════════════════════════════════════════════
# app.py — Dashboard Línea 123 · Streamlit
# Visualización I · Ciencia de Datos · Universidad Externado de Colombia
#
# Propósito: replicar las cuatro gráficas del dashboard de Plotly II en una
# aplicación Streamlit con controles reactivos. La diferencia clave respecto
# a Plotly (updatemenus) es que aquí los controles RECALCULAN los datos —
# no solo ocultan trazas ya existentes.
#
# Uso: streamlit run app.py
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Configuración de página ───────────────────────────────────────────────────
# Debe ser la primera instrucción de Streamlit en el script.
st.set_page_config(
    page_title="Línea 123 Bogotá · Q4 2025",
    page_icon="🚨",
    layout="wide",
    # Fondo blanco forzado independientemente del tema del sistema operativo
    # del usuario. Garantiza coherencia visual para audiencia ciudadana.
)

# Forzar tema claro vía CSS — st.set_page_config no tiene parámetro directo.
st.markdown(
    """
    <style>
    /* Fondo gris suave para el área principal */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="block-container"] {
        background-color: #F0F2F6 !important;
        color: #333333 !important;
    }
    /* Sidebar ligeramente más oscuro para contraste */
    [data-testid="stSidebar"] {
        background-color: #E4E7EF !important;
    }
    /* Texto del sidebar visible */
    [data-testid="stSidebar"] * {
        color: #333333 !important;
    }
    /* Textos generales */
    h1, h2, h3, p, span, label, div {
        color: #333333 !important;
    }
    /* Métricas KPI */
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        color: #333333 !important;
    }
    /* Caption */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #666666 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 · PALETA Y FUNCIONES AUXILIARES
# Idénticas al notebook de Plotly II para mantener coherencia visual.
# ══════════════════════════════════════════════════════════════════════════════

AZUL_BOGOTA   = '#225495'
GRIS_CONTEXTO = '#B0B0B0'

COLORES_PRIORIDAD = {
    'Crítica': '#AA1023',
    'Alta':    '#C0392B',
    'Media':   '#F7B325',
    'Baja':    '#B0B0B0',
}
ORDEN_PRIORIDAD = ['Baja', 'Media', 'Alta', 'Crítica']

COLORES_GENERO = {
    'Femenino':  '#225495',
    'Masculino': '#4E6FA3',   # oscurecido respecto a #7A98BF para soportar texto blanco
}

ORDEN_GRUPO_EDAD = [
    'Primera infancia (0-5)', 'Infancia (6-11)', 'Adolescencia (12-17)',
    'Juventud (18-28)', 'Adultez (29-59)', 'Persona mayor (60 o más)',
    'Sin información',
]

def fmt_col(valor):
    """Formato colombiano de enteros: punto como separador de miles."""
    return f'{int(valor):,}'.replace(',', '.')

def fmt_col_dec(valor, decimales=1):
    """Formato colombiano con decimales: punto=miles, coma=decimal."""
    texto = f'{valor:,.{decimales}f}'
    return texto.replace(',', '@').replace('.', ',').replace('@', '.')

# Plantilla reutilizable: tipografía, colores base y separador colombiano.
# separators=',.' → Plotly usa coma como decimal y punto como miles.
PLANTILLA_CURSO = go.layout.Template(
    layout=go.Layout(
        font=dict(family='Segoe UI, sans-serif', size=12, color='#333333'),
        title=dict(font=dict(size=14)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        separators=',.',
        xaxis=dict(
            gridcolor='#E0E0E0', gridwidth=0.5,
            linecolor='#CCCCCC', linewidth=0.5,
            tickfont=dict(color='#666666'),
        ),
        yaxis=dict(
            gridcolor='#E0E0E0', gridwidth=0.5,
            linecolor='#CCCCCC', linewidth=0.5,
            tickfont=dict(color='#666666'),
        ),
    )
)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 · CARGA Y PREPARACIÓN DE DATOS
#
# @st.cache_data indica a Streamlit que guarde el resultado en memoria.
# Sin este decorador, el DataFrame se recargaría en CADA interacción del
# usuario (cada vez que mueve un selectbox). Con él, la carga ocurre una
# sola vez por sesión.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def cargar_datos(ruta: str) -> pd.DataFrame:
    """Carga el Parquet y deriva las variables necesarias para el dashboard."""

    df = pd.read_parquet(ruta)

    # ── Variables categóricas temporales ─────────────────────────────────────
    # Nota: no se deduplica por NUMERO_INCIDENTE porque un incidente puede
    # involucrar varias personas. Cada fila representa una persona atendida.
    DIAS_EN_ES = {
        'Monday': 'lunes', 'Tuesday': 'martes', 'Wednesday': 'miércoles',
        'Thursday': 'jueves', 'Friday': 'viernes', 'Saturday': 'sábado',
        'Sunday': 'domingo',
    }
    df['DIA_SEMANA'] = (
        df['FECHA_INICIO_DESPLAZAMIENTO_MOVIL'].dt.day_name().map(DIAS_EN_ES)
    )
    ORDEN_DIAS = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
    df['DIA_SEMANA'] = pd.Categorical(df['DIA_SEMANA'], categories=ORDEN_DIAS, ordered=True)

    MESES_ES = {10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
    df['MES'] = df['FECHA_INICIO_DESPLAZAMIENTO_MOVIL'].dt.month.map(MESES_ES)
    ORDEN_MESES = ['Octubre', 'Noviembre', 'Diciembre']
    df['MES'] = pd.Categorical(df['MES'], categories=ORDEN_MESES, ordered=True)

    # FECHA y HORA ya vienen calculadas desde Preparacion_datos_Llamadas123.ipynb.
    # No se recalculan para respetar el principio "transformar una vez".
    # Se asegura que FECHA sea tipo date (por si el Parquet lo guardó como datetime).
    if df['FECHA'].dtype != 'object':
        df['FECHA'] = pd.to_datetime(df['FECHA']).dt.date

    # ── Grupo de edad ────────────────────────────────────────────────────────
    def edad_a_anios(row):
        if pd.isna(row['EDAD']) or pd.isna(row['UNIDAD']):
            return np.nan
        if row['UNIDAD'] == 'Años':
            return row['EDAD']
        elif row['UNIDAD'] == 'Meses':
            return row['EDAD'] // 12
        elif row['UNIDAD'] == 'Días':
            return row['EDAD'] // 365
        elif row['UNIDAD'] == 'Horas':
            return 0
        return np.nan

    def asignar_grupo(edad):
        if pd.isna(edad):
            return 'Sin información'
        elif edad <= 5:
            return 'Primera infancia (0-5)'
        elif edad <= 11:
            return 'Infancia (6-11)'
        elif edad <= 17:
            return 'Adolescencia (12-17)'
        elif edad <= 28:
            return 'Juventud (18-28)'
        elif edad <= 59:
            return 'Adultez (29-59)'
        else:
            return 'Persona mayor (60 o más)'

    df['EDAD_ANIOS'] = df.apply(edad_a_anios, axis=1)
    df['GRUPO_EDAD'] = df['EDAD_ANIOS'].apply(asignar_grupo)
    df['GRUPO_EDAD'] = pd.Categorical(
        df['GRUPO_EDAD'], categories=ORDEN_GRUPO_EDAD, ordered=True
    )

    return df


# ── Ruta al Parquet ───────────────────────────────────────────────────────────
# En Community Cloud el Parquet está en la raíz del repositorio.
# Para pruebas locales en Colab, reemplazar por la ruta completa de Drive.
RUTA_PARQUET = 'llamadas123_limpio.parquet'

try:
    df_total = cargar_datos(RUTA_PARQUET)
except FileNotFoundError:
    st.error(
        "No se encontró el archivo Parquet. "
        "Verificar que el archivo llamadas123_limpio.parquet esté en la misma "
        "carpeta que app.py."
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 · CONTROLES (SIDEBAR)
#
# En Streamlit, cada vez que el usuario cambia un control, TODO el script
# se re-ejecuta de arriba a abajo. Por eso los controles se definen aquí,
# antes de cualquier cálculo: los valores seleccionados se usan para filtrar
# df_total y producir df, el DataFrame que alimenta todas las gráficas.
#
# Diferencia clave con Plotly updatemenus:
#   Plotly → oculta/muestra trazas ya precalculadas (sin recalcular).
#   Streamlit → recalcula todos los datos cada vez (flexibilidad total).
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Emoji en lugar de imagen URL — no depende de servicios externos.
    st.markdown("# 🏙️")
    st.markdown("## Línea 123 · Bogotá")
    st.divider()

    # Control 1: mes
    meses_disponibles = ['Todos'] + list(df_total['MES'].cat.categories)
    mes_sel = st.selectbox("📅 Mes", options=meses_disponibles)

    # Control 2: prioridad
    prioridades_disponibles = ['Todas'] + ORDEN_PRIORIDAD[::-1]  # Crítica primero
    prioridad_sel = st.selectbox("🚨 Prioridad", options=prioridades_disponibles)

    st.divider()
    st.caption("Fuente: Datos Abiertos Bogotá · datos.gov.co · Q4 2025")

# ── Aplicar filtros ───────────────────────────────────────────────────────────
# Esta es la línea más importante del script pedagógicamente:
# df es el subconjunto filtrado que alimenta TODAS las gráficas.
# Si se elimina este filtro, todos los gráficos muestran siempre el total.
df = df_total.copy()
if mes_sel != 'Todos':
    df = df[df['MES'] == mes_sel]
if prioridad_sel != 'Todas':
    df = df[df['PRIORIDAD_FINAL'] == prioridad_sel]

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 · ENCABEZADO Y KPIs
# ══════════════════════════════════════════════════════════════════════════════

# Título y subtítulo fijos — narrativos, para audiencia ciudadana.
# No cambian con los filtros: enuncian el hallazgo central del Q4 completo.
st.markdown(
    "## 8 de cada 10 emergencias en Bogotá son de prioridad alta o crítica"
)
st.markdown("**Octubre a diciembre de 2025**")

# Subtítulo instruccional para el usuario
st.caption(
    "Los controles del panel izquierdo filtran simultáneamente las cuatro gráficas. "
    "Compare con el dashboard de Power BI: el comportamiento es equivalente al de los slicers."
)

# ── KPIs ──────────────────────────────────────────────────────────────────────
# Un incidente puede involucrar varias personas, por lo que el DataFrame
# tiene más filas que incidentes únicos. Para contar incidentes se usa
# nunique() sobre NUMERO_INCIDENTE, que ignora las filas repetidas del
# mismo incidente.
total_incidentes = df['NUMERO_INCIDENTE'].nunique()
pct_critica_alta = (
    df[df['PRIORIDAD_FINAL'].isin(['Crítica', 'Alta'])].shape[0] / len(df) * 100
    if len(df) > 0 else 0
)
localidad_top = (
    df[~df['LOCALIDAD'].isin(['Sin información', 'Fuera de Bogotá'])]
    ['LOCALIDAD'].value_counts().idxmax()
    if len(df) > 0 else '—'
)

col1, col2, col3 = st.columns(3)
col1.metric("Total incidentes", fmt_col(total_incidentes))
col2.metric("Prioridad Crítica o Alta", f"{fmt_col_dec(pct_critica_alta)}%")
col3.metric("Localidad con más incidentes", localidad_top)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 · GRÁFICAS
#
# Cada gráfica se construye como una figura Plotly independiente.
# No se usa make_subplots: la disposición espacial la gestiona Streamlit
# con st.columns(). Esto permite que cada gráfica responda de forma
# individual a los filtros sin el overhead de reconstruir todo el subplot.
# ══════════════════════════════════════════════════════════════════════════════

if len(df) == 0:
    st.warning("El filtro seleccionado no produce registros. Ajustar los controles.")
    st.stop()

# ── Datos compartidos entre gráficas ─────────────────────────────────────────
EXCLUIR_LOC = ['Sin información', 'Fuera de Bogotá']

loc_top7 = (
    df[~df['LOCALIDAD'].isin(EXCLUIR_LOC)]
    ['LOCALIDAD'].value_counts().head(7).reset_index()
)
loc_top7.columns = ['LOCALIDAD', 'INCIDENTES']
loc_top7 = loc_top7.iloc[::-1].reset_index(drop=True)  # mayor arriba

colores_barra = [
    AZUL_BOGOTA if loc in loc_top7['LOCALIDAD'].tail(3).values else GRIS_CONTEXTO
    for loc in loc_top7['LOCALIDAD']
]

n_dias = df['FECHA'].nunique()
# Conteo distintivo de incidentes por hora — no personas:
# un incidente puede involucrar varias personas; el patrón
# horario de demanda se mide en incidentes, no en atendidos.
prom_hora = (
    df.groupby('HORA')['NUMERO_INCIDENTE'].nunique()
    .reset_index(name='TOTAL')
)
prom_hora['PROMEDIO'] = (prom_hora['TOTAL'] / n_dias).round(1)
hora_pico = prom_hora.loc[prom_hora['PROMEDIO'].idxmax()]

# ── Fila superior ─────────────────────────────────────────────────────────────
col_izq, col_der = st.columns(2)

# Gráfica 1 · Barras por localidad ────────────────────────────────────────────
with col_izq:
    fig1 = go.Figure()
    fig1.add_trace(
        go.Bar(
            x=loc_top7['INCIDENTES'],
            y=loc_top7['LOCALIDAD'],
            orientation='h',
            marker_color=colores_barra,
            text=[fmt_col(v) for v in loc_top7['INCIDENTES']],
            textposition='outside',
            textfont=dict(size=10),
            hovertemplate='<b>%{y}</b><br>Personas atendidas: %{text}<extra></extra>',
        )
    )
    fig1.update_layout(
        template=PLANTILLA_CURSO,
        title=dict(
            text=(
                '<b>Kennedy, Engativá y Suba concentran el 34% de las personas atendidas</b>'
                '<br><sup>Personas involucradas en incidentes por localidad · '
                '7 localidades principales (61%)</sup>'
            ),
            font=dict(size=13),
            x=0, xanchor='left',
            pad=dict(l=10),
        ),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=[0, loc_top7['INCIDENTES'].max() * 1.35]),
        yaxis=dict(tickfont=dict(size=10), automargin=True, ticklabelstandoff=8),
        showlegend=False,
        height=380,
        margin=dict(t=90, b=40, l=10, r=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    st.plotly_chart(fig1, use_container_width=True)

# Gráfica 2 · Línea por hora ──────────────────────────────────────────────────
with col_der:
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=prom_hora['HORA'], y=prom_hora['PROMEDIO'],
            mode='lines+markers',
            line=dict(color=AZUL_BOGOTA, width=2),
            marker=dict(size=4, color=AZUL_BOGOTA),
            hovertemplate='<b>Hora %{x}:00</b><br>Promedio: %{y} inc./día<extra></extra>',
            showlegend=False,
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=[hora_pico['HORA']], y=[hora_pico['PROMEDIO']],
            mode='markers+text',
            marker=dict(size=10, color='#AA1023'),
            text=[fmt_col_dec(hora_pico['PROMEDIO'], 0)],
            textposition='top center',
            textfont=dict(size=11, color='#AA1023'),
            showlegend=False, hoverinfo='skip',
        )
    )
    fig2.update_layout(
        template=PLANTILLA_CURSO,
        title=dict(
            text=(
                '<b>En promedio, cada día registra su pico de demanda entre las 10 y las 11 a.m.</b>'
                '<br><sup>Promedio de incidentes por hora del día</sup>'
            ),
            font=dict(size=13),
            x=0, xanchor='left',
            pad=dict(l=10),
        ),
        xaxis=dict(tickfont=dict(size=10), dtick=4, range=[-0.5, 23.5],
                   title=dict(text='Hora del día', font=dict(size=11, color='#666666'))),
        yaxis=dict(tickfont=dict(size=10), automargin=True, rangemode='tozero'),
        showlegend=False,
        height=380,
        margin=dict(t=90, b=60, l=10, r=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Fila inferior ─────────────────────────────────────────────────────────────
col_izq2, col_der2 = st.columns(2)

# Gráfica 3 · Barras apiladas prioridad × tipo de incidente ───────────────────
with col_izq2:
    top_tipos = df['TIPO_INCIDENTE'].value_counts().head(6).index.tolist()
    df_tipo = df[df['TIPO_INCIDENTE'].isin(top_tipos)]

    tabla_tipo = pd.crosstab(
        df_tipo['TIPO_INCIDENTE'], df_tipo['PRIORIDAD_FINAL'], normalize='index'
    ) * 100
    for p in ['Crítica', 'Alta', 'Media', 'Baja']:
        if p not in tabla_tipo.columns:
            tabla_tipo[p] = 0.0
    tabla_tipo = tabla_tipo[['Crítica', 'Alta', 'Media', 'Baja']]

    critica_alta_abs = (
        df_tipo[df_tipo['PRIORIDAD_FINAL'].isin(['Crítica', 'Alta'])]
        .groupby('TIPO_INCIDENTE').size()
        .reindex(tabla_tipo.index, fill_value=0)
    )
    tabla_tipo = tabla_tipo.loc[critica_alta_abs.sort_values(ascending=True).index]
    orden_tipos = tabla_tipo.index.tolist()

    # ── Construcción manual de trazas go.Bar ─────────────────────────────────
    # Se abandona px.bar para fig3 porque px divide los datos por color antes
    # de asignar customdata, lo que genera NaN en las etiquetas de texto cuando
    # hay celdas con valor 0. Con go.Bar se controla cada traza explícitamente.
    #
    # Solo se muestran etiquetas de texto para Crítica y Alta (punto 6):
    # Media y Baja tienen textfont transparente para ocultar sus etiquetas
    # sin perder la barra visual.

    fig3 = go.Figure()

    # Orden de trazas: Crítica primero → queda al inicio (izquierda) de la barra apilada.
    # Alta va segundo, Media y Baja al final. Solo Crítica y Alta muestran etiquetas.
    config_prioridad = [
        ('Crítica', COLORES_PRIORIDAD['Crítica'],  True),
        ('Alta',    COLORES_PRIORIDAD['Alta'],     True),
        ('Media',   COLORES_PRIORIDAD['Media'],   False),
        ('Baja',    COLORES_PRIORIDAD['Baja'],    False),
    ]

    for prioridad, color, mostrar_etiqueta in config_prioridad:
        valores = [tabla_tipo.loc[t, prioridad] for t in orden_tipos]
        etiquetas = [
            f'{v:.0f}%' if (mostrar_etiqueta and v >= 7) else ''
            for v in valores
        ]
        hover_vals = [fmt_col_dec(v, 1) for v in valores]

        fig3.add_trace(go.Bar(
            name=prioridad,
            x=valores,
            y=orden_tipos,
            orientation='h',
            marker_color=color,
            text=etiquetas,
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(size=10, color='white'),
            customdata=hover_vals,
            hovertemplate='<b>%{y}</b><br>' + prioridad + ': %{customdata}%<extra></extra>',
        ))

    fig3.update_layout(
        template=PLANTILLA_CURSO,
        barmode='stack',
        title=dict(
            text=(
                '<b>El 80% de los tipos de atención principales se clasifican '
                'con prioridad crítica o alta</b>'
                '<br><sup>Distribución de prioridad por tipo de atención · '
                '6 tipos principales (72%)</sup>'
            ),
            font=dict(size=13),
            x=0, xanchor='left',
            pad=dict(l=10),
        ),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 105]),
        yaxis=dict(tickfont=dict(size=10), automargin=True, ticklabelstandoff=10,
                   categoryorder='array', categoryarray=orden_tipos),
        legend=dict(
            orientation='h', y=-0.15, x=0,
            title=dict(text='Prioridad final:', font=dict(size=11)),
            font=dict(size=11),
            traceorder='normal',
        ),
        height=400,
        margin=dict(t=110, b=80, l=10, r=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    st.plotly_chart(fig3, use_container_width=True)

# Gráfica 4 · Barras apiladas género × grupo de edad ─────────────────────────
with col_der2:
    df_demo = df[(df['GRUPO_EDAD'] != 'Sin información') & (df['GENERO'].notna())]
    tabla_edad = pd.crosstab(
        df_demo['GRUPO_EDAD'], df_demo['GENERO'], normalize='index'
    ) * 100
    orden_invertido = [g for g in reversed(ORDEN_GRUPO_EDAD) if g != 'Sin información']
    tabla_edad = tabla_edad.reindex(orden_invertido)

    for g in ['Femenino', 'Masculino']:
        if g not in tabla_edad.columns:
            tabla_edad[g] = 0.0

    tabla_edad_long = tabla_edad.reset_index().melt(
        id_vars='GRUPO_EDAD', var_name='GENERO', value_name='PORCENTAJE'
    )
    tabla_edad_long['PORCENTAJE_FMT'] = tabla_edad_long['PORCENTAJE'].apply(
        lambda v: fmt_col_dec(v, 1)
    )

    fig4 = px.bar(
        tabla_edad_long,
        x='PORCENTAJE', y='GRUPO_EDAD',
        color='GENERO', orientation='h',
        color_discrete_map=COLORES_GENERO,
        category_orders={
            'GENERO': ['Femenino', 'Masculino'],
            'GRUPO_EDAD': orden_invertido,
        },
        barmode='stack',
        text='PORCENTAJE',
        custom_data=['PORCENTAJE_FMT'],
    )
    fig4.update_traces(
        texttemplate='%{text:.0f}%',
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(size=10, color='white'),
        hovertemplate='<b>%{y}</b><br>%{fullData.name}: %{customdata[0]}%<extra></extra>',
    )
    fig4.update_traces(
        textfont=dict(size=10, color='white'),
        selector=dict(name='Masculino'),
    )
    fig4.update_layout(
        template=PLANTILLA_CURSO,
        title=dict(
            text=(
                '<b>Las personas mayores y los adultos concentran el 68% '
                'de las emergencias con datos <br> demográficos completos</b>'
                '<br><sup>Distribución por grupo de edad y género</sup>'
            ),
            font=dict(size=13),
            x=0, xanchor='left',
            pad=dict(l=10),
        ),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=[0, 105], title=dict(text='')),
        yaxis=dict(
            tickfont=dict(size=10),
            automargin=True,
            ticklabelstandoff=8,
            title=dict(text='Grupo de edad', font=dict(size=11, color='#666666')),
        ),
        legend=dict(
            orientation='h', y=-0.15, x=0,
            title=dict(text='Género:', font=dict(size=11)),
            font=dict(size=11),
        ),
        height=400,
        margin=dict(t=110, b=80, l=10, r=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 · PIE DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.caption(
    "Fuente: Datos Abiertos Bogotá · datos.gov.co · Línea 123 · Octubre–Diciembre 2025."
)
