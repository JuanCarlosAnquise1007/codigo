from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
from scipy.stats import poisson
import statsmodels.api as sm
import os
import io
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for generating plots without a display
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

data_cache = None  # Variable global para almacenar los datos generados
uploaded_data_cache = None  # Variable global para almacenar los datos cargados

# Función para generar datos sintéticos
def generate_synthetic_data(n_samples=100):
    np.random.seed(42)
    data = {
        'n_empleados': np.random.randint(50, 500, n_samples),
        'horas_capacitacion': np.random.uniform(10, 40, n_samples),
        'antiguedad_promedio': np.random.uniform(2, 15, n_samples),
        'presupuesto_seguridad': np.random.uniform(20, 100, n_samples),
        'n_supervisores': np.random.randint(2, 15, n_samples)
    }

    lambda_ = np.exp(
        -2 +
        0.003 * data['n_empleados'] +
        -0.05 * data['horas_capacitacion'] +
        -0.1 * data['antiguedad_promedio'] +
        -0.01 * data['presupuesto_seguridad'] +
        0.1 * data['n_supervisores']
    )

    data['n_accidentes'] = poisson.rvs(lambda_)
    return pd.DataFrame(data)

# Función para ajustar el modelo de Poisson
def fit_poisson_model(df):
    X = df[['n_empleados', 'horas_capacitacion', 'antiguedad_promedio',
            'presupuesto_seguridad', 'n_supervisores']]
    y = df['n_accidentes']
    X = sm.add_constant(X)
    modelo = sm.GLM(y, X, family=sm.families.Poisson())
    return modelo.fit()

# Ruta principal
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para generar datos sintéticos
@app.route('/generate_data', methods=['POST'])
def generate_data():
    global data_cache
    n_samples = int(request.form.get('n_samples', 100))
    data_cache = generate_synthetic_data(n_samples)
    return data_cache.to_json(orient='records')

# Ruta para descargar los datos generados
@app.route('/download_data', methods=['GET'])
def download_data():
    global data_cache
    if data_cache is None:
        return jsonify({'error': 'No hay datos generados para descargar.'}), 400

    output_format = request.args.get('format', 'excel')
    output = io.BytesIO()

    if output_format == 'csv':
        data_cache.to_csv(output, index=False)
        output.seek(0)
        mimetype = 'text/csv'
        download_name = 'datos_sinteticos.csv'
    else:
        data_cache.to_excel(output, index=False)
        output.seek(0)
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        download_name = 'datos_sinteticos.xlsx'

    return send_file(
        output,
        mimetype=mimetype,
        as_attachment=True,
        download_name=download_name
    )

# Ruta para subir archivo y ajustar modelo Poisson
@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_data_cache
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró ningún archivo.'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo.'})

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        uploaded_data_cache = df
        resultados = fit_poisson_model(df)

        coef = resultados.params
        exp_coef = np.exp(coef)
        cambio_porcentual = (exp_coef - 1) * 100

        results_df = pd.DataFrame({
            'Variable': coef.index,
            'Coeficiente': coef.values,
            'Exp(Coeficiente)': exp_coef.values,
            'Cambio %': cambio_porcentual.values,
            'P-valor': resultados.pvalues.values
        })

        return results_df.to_json(orient='records')

    except Exception as e:
        return jsonify({'error': str(e)})

# Ruta para mostrar los datos del archivo subido
@app.route('/show_uploaded_data', methods=['POST'])
def show_uploaded_data():
    global uploaded_data_cache
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró ningún archivo.'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo.'})

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        uploaded_data_cache = df
        return df.to_json(orient='records')

    except Exception as e:
        return jsonify({'error': str(e)})

# Ruta para generar el diagrama de dispersión
@app.route('/scatter_plot', methods=['POST'])
def scatter_plot():
    global uploaded_data_cache
    if uploaded_data_cache is None:
        return jsonify({'error': 'No hay datos cargados para generar el gráfico.'}), 400

    x_variable = request.form['x_variable']
    y_variable = request.form['y_variable']

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=uploaded_data_cache, x=x_variable, y=y_variable)
    plt.title('Diagrama de Dispersión')
    plt.xlabel(x_variable)
    plt.ylabel(y_variable)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

# Ruta para generar el histograma con curva de normalidad
@app.route('/histogram_plot', methods=['POST'])
def histogram_plot():
    global uploaded_data_cache
    if uploaded_data_cache is None:
        return jsonify({'error': 'No hay datos cargados para generar el gráfico.'}), 400

    variable = request.form['variable']

    plt.figure(figsize=(10, 6))
    sns.histplot(uploaded_data_cache[variable], kde=True)
    plt.title('Histograma con Curva de Normalidad')
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

# Ruta para generar el diagrama de series de tiempo
@app.route('/time_series_plot', methods=['POST'])
def time_series_plot():
    global uploaded_data_cache
    if uploaded_data_cache is None:
        return jsonify({'error': 'No hay datos cargados para generar el gráfico.'}), 400

    variable = request.form['variable']

    plt.figure(figsize=(10, 6))
    plt.plot(uploaded_data_cache.index, uploaded_data_cache[variable])
    plt.title('Diagrama de Series de Tiempo')
    plt.xlabel('Índice')
    plt.ylabel(variable)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
