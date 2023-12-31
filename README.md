# Predicción del Crédito Bancario

> **Grupo 28**: `Diana Mory, Fabián Trejo, Alejandra Cruz, Luis Silvera`

![](grupo28.jpg)

#**PROBLEMA DE NEGOCIO**


---




La importancia de reducir el riesgo crediticio ha llevado a una institución financiera alemana a buscar soluciones innovadoras. Como científicos de datos, hemos sido convocados para construir un modelo de machine learning preciso y confiable que sea capaz de evaluar con mayor precisión la probabilidad de incumplimiento crediticio de sus clientes.

### **Tus tareas principales serán:**

**1. Preprocesamiento de Datos:** Realizar limpieza de datos, manejar valores faltantes, codificación de variables categóricas y normalización/escalado de datos.

**2. Exploración de Datos:** Analizar y comprender el conjunto de datos proporcionado, identificar variables llaves y realizar visualizaciones para entender las relaciones entre las variables y seleccionar las características relevantes.

**3. Construcción de Modelos:** Experimentar con algunos algoritmos de machine learning como Regresión Logística, Árboles de Decisión, Random Forest, Naive Bayes, entre otros.

**4. Evaluación y Selección del Modelo:** Evaluar los modelos utilizando métricas como precisión, recall, área bajo la curva ROC, y F1-score. Seleccionar el modelo con el mejor rendimiento para la predicción de la solvencia crediticia.

#**1. Configuración del Ambiente**


---




Necesitamos instalar la libreria Pycaret la que usaremos mas adelante


```python
%pip install pycaret[full]
```

Importamos las librerías que estaremos utilizando


```python
import pandas as pd
pd.set_option('display.max_columns', None)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from pycaret.classification import *
global df_banco, resultados
```

#**2. Preprocesamiento de Datos**


---


Cargamos los datos que utilizaremos


```python
df_banco = pd.read_csv("german_credit.csv")
df_banco.head()
```





  <div id="df-0c9400ca-2880-46f7-90c1-38bd190e7ca4" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>account_check_status</th>
      <th>duration_in_month</th>
      <th>credit_history</th>
      <th>purpose</th>
      <th>credit_amount</th>
      <th>savings</th>
      <th>present_emp_since</th>
      <th>installment_as_income_perc</th>
      <th>personal_status_sex</th>
      <th>other_debtors</th>
      <th>present_res_since</th>
      <th>property</th>
      <th>age</th>
      <th>other_installment_plans</th>
      <th>housing</th>
      <th>credits_this_bank</th>
      <th>job</th>
      <th>people_under_maintenance</th>
      <th>telephone</th>
      <th>foreign_worker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>&lt; 0 DM</td>
      <td>6</td>
      <td>critical account/ other credits existing (not ...</td>
      <td>domestic appliances</td>
      <td>1169</td>
      <td>unknown/ no savings account</td>
      <td>.. &gt;= 7 years</td>
      <td>4</td>
      <td>male : single</td>
      <td>none</td>
      <td>4</td>
      <td>real estate</td>
      <td>67</td>
      <td>none</td>
      <td>own</td>
      <td>2</td>
      <td>skilled employee / official</td>
      <td>1</td>
      <td>yes, registered under the customers name</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0 &lt;= ... &lt; 200 DM</td>
      <td>48</td>
      <td>existing credits paid back duly till now</td>
      <td>domestic appliances</td>
      <td>5951</td>
      <td>... &lt; 100 DM</td>
      <td>1 &lt;= ... &lt; 4 years</td>
      <td>2</td>
      <td>female : divorced/separated/married</td>
      <td>none</td>
      <td>2</td>
      <td>real estate</td>
      <td>22</td>
      <td>none</td>
      <td>own</td>
      <td>1</td>
      <td>skilled employee / official</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>no checking account</td>
      <td>12</td>
      <td>critical account/ other credits existing (not ...</td>
      <td>(vacation - does not exist?)</td>
      <td>2096</td>
      <td>... &lt; 100 DM</td>
      <td>4 &lt;= ... &lt; 7 years</td>
      <td>2</td>
      <td>male : single</td>
      <td>none</td>
      <td>3</td>
      <td>real estate</td>
      <td>49</td>
      <td>none</td>
      <td>own</td>
      <td>1</td>
      <td>unskilled - resident</td>
      <td>2</td>
      <td>none</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>&lt; 0 DM</td>
      <td>42</td>
      <td>existing credits paid back duly till now</td>
      <td>radio/television</td>
      <td>7882</td>
      <td>... &lt; 100 DM</td>
      <td>4 &lt;= ... &lt; 7 years</td>
      <td>2</td>
      <td>male : single</td>
      <td>guarantor</td>
      <td>4</td>
      <td>if not A121 : building society savings agreeme...</td>
      <td>45</td>
      <td>none</td>
      <td>for free</td>
      <td>1</td>
      <td>skilled employee / official</td>
      <td>2</td>
      <td>none</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>&lt; 0 DM</td>
      <td>24</td>
      <td>delay in paying off in the past</td>
      <td>car (new)</td>
      <td>4870</td>
      <td>... &lt; 100 DM</td>
      <td>1 &lt;= ... &lt; 4 years</td>
      <td>3</td>
      <td>male : single</td>
      <td>none</td>
      <td>4</td>
      <td>unknown / no property</td>
      <td>53</td>
      <td>none</td>
      <td>for free</td>
      <td>2</td>
      <td>skilled employee / official</td>
      <td>2</td>
      <td>none</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0c9400ca-2880-46f7-90c1-38bd190e7ca4')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0c9400ca-2880-46f7-90c1-38bd190e7ca4 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0c9400ca-2880-46f7-90c1-38bd190e7ca4');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-5948e18c-2dee-4052-92a5-932198760749">
  <button class="colab-df-quickchart" onclick="quickchart('df-5948e18c-2dee-4052-92a5-932198760749')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-5948e18c-2dee-4052-92a5-932198760749 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




Obtenemos su información


```python
df_banco.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 21 columns):
     #   Column                      Non-Null Count  Dtype 
    ---  ------                      --------------  ----- 
     0   default                     1000 non-null   int64 
     1   account_check_status        1000 non-null   object
     2   duration_in_month           1000 non-null   int64 
     3   credit_history              1000 non-null   object
     4   purpose                     1000 non-null   object
     5   credit_amount               1000 non-null   int64 
     6   savings                     1000 non-null   object
     7   present_emp_since           1000 non-null   object
     8   installment_as_income_perc  1000 non-null   int64 
     9   personal_status_sex         1000 non-null   object
     10  other_debtors               1000 non-null   object
     11  present_res_since           1000 non-null   int64 
     12  property                    1000 non-null   object
     13  age                         1000 non-null   int64 
     14  other_installment_plans     1000 non-null   object
     15  housing                     1000 non-null   object
     16  credits_this_bank           1000 non-null   int64 
     17  job                         1000 non-null   object
     18  people_under_maintenance    1000 non-null   int64 
     19  telephone                   1000 non-null   object
     20  foreign_worker              1000 non-null   object
    dtypes: int64(8), object(13)
    memory usage: 164.2+ KB
    

Obtenemos su descripción


```python
df_banco.describe().T
```





  <div id="df-f003653f-3ca0-4c17-a96f-bb412f6329ad" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>default</th>
      <td>1000.0</td>
      <td>0.300</td>
      <td>0.458487</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>duration_in_month</th>
      <td>1000.0</td>
      <td>20.903</td>
      <td>12.058814</td>
      <td>4.0</td>
      <td>12.0</td>
      <td>18.0</td>
      <td>24.00</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>credit_amount</th>
      <td>1000.0</td>
      <td>3271.258</td>
      <td>2822.736876</td>
      <td>250.0</td>
      <td>1365.5</td>
      <td>2319.5</td>
      <td>3972.25</td>
      <td>18424.0</td>
    </tr>
    <tr>
      <th>installment_as_income_perc</th>
      <td>1000.0</td>
      <td>2.973</td>
      <td>1.118715</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>present_res_since</th>
      <td>1000.0</td>
      <td>2.845</td>
      <td>1.103718</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>age</th>
      <td>1000.0</td>
      <td>35.546</td>
      <td>11.375469</td>
      <td>19.0</td>
      <td>27.0</td>
      <td>33.0</td>
      <td>42.00</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>credits_this_bank</th>
      <td>1000.0</td>
      <td>1.407</td>
      <td>0.577654</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>people_under_maintenance</th>
      <td>1000.0</td>
      <td>1.155</td>
      <td>0.362086</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f003653f-3ca0-4c17-a96f-bb412f6329ad')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f003653f-3ca0-4c17-a96f-bb412f6329ad button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f003653f-3ca0-4c17-a96f-bb412f6329ad');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-ebe2d257-46d7-4c20-a458-526b96f258e3">
  <button class="colab-df-quickchart" onclick="quickchart('df-ebe2d257-46d7-4c20-a458-526b96f258e3')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-ebe2d257-46d7-4c20-a458-526b96f258e3 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




Veamos la cantidad de valores únicos de cada columna


```python
df_banco.nunique()
```




    default                         2
    account_check_status            4
    duration_in_month              33
    credit_history                  5
    purpose                        10
    credit_amount                 921
    savings                         5
    present_emp_since               5
    installment_as_income_perc      4
    personal_status_sex             4
    other_debtors                   3
    present_res_since               4
    property                        4
    age                            53
    other_installment_plans         3
    housing                         3
    credits_this_bank               4
    job                             4
    people_under_maintenance        2
    telephone                       2
    foreign_worker                  2
    dtype: int64



Procesamos los datos mediante mediante un `map` para convertir los valores de texto a numéricos


```python
def procesar_datos():
    global df_banco  # Define las variables globales

    # Eliminar duplicados
    df_banco = df_banco.drop_duplicates()

    # Tratamiento de nulos
    df_banco = df_banco.dropna()  # Eliminar registros con valores nulos
    # O
    # df_banco = df_banco.fillna(valor)  # Sustituir los valores nulos por un valor específico

    # Reemplazar textos por números en cada columna que contenga datos cualitativos
    diccionario_account = {'< 0 DM': 1, '0 <= ... < 200 DM': 2, '>= 200 DM / salary assignments for at least 1 year': 3, 'no checking account': 4}
    diccionario_credith = {'no credits taken/ all credits paid back duly': 1, 'all credits at this bank paid back duly': 2, 'existing credits paid back duly till now': 3, 'delay in paying off in the past': 4, 'critical account/ other credits existing (not at this bank)': 5}
    diccionario_purpose = {'car (new)': 1, 'car (used)': 2, 'furniture/equipment': 3, 'radio/television': 4, 'domestic appliances': 5, 'repairs': 6, 'education': 7, '(vacation - does not exist?)': 8, 'retraining': 9, 'business': 10, 'others': 10}
    diccionario_savings = {'unknown/ no savings account': 1, '.. >= 1000 DM ': 2, '500 <= ... < 1000 DM ': 3, '100 <= ... < 500 DM': 4, '... < 100 DM': 5}
    diccionario_present_es = {'.. >= 7 years': 1, '4 <= ... < 7 years': 2, '1 <= ... < 4 years': 3, '... < 1 year ': 4, 'unemployed': 5}
    diccionario_personal_ss = {'male : divorced/separated': 1, 'female : divorced/separated/married': 2, 'male : single': 3, 'male : married/widowed': 4, 'female : single': 5}
    diccionario_other_debtors = {'none': 1, 'co-applicant': 2, 'guarantor': 3}
    diccionario_property = {'real estate': 1, 'if not A121 : building society savings agreement/ life insurance': 2, 'if not A121/A122 : car or other, not in attribute 6': 3, 'unknown / no property': 4}
    diccionario_other_ip = {'bank': 1, 'stores': 2, 'none': 3}
    diccionario_housing = {'rent': 1, 'own': 2, 'for free': 3}
    diccionario_job = {'unemployed/ unskilled - non-resident': 1, 'unskilled - resident': 2, 'skilled employee / official': 3, 'management/ self-employed/ highly qualified employee/ officer': 4}
    diccionario_telephone = {'none': 0, 'yes, registered under the customers name ': 1}
    diccionario_foreign_worker = {'no': 0, 'yes': 1}


    # Crea un diccionario de reemplazo
    df_banco['account_check_status'] = df_banco['account_check_status'].map(diccionario_account)
    df_banco['credit_history'] = df_banco['credit_history'].map(diccionario_credith)
    df_banco['purpose'] = df_banco['purpose'].map(diccionario_purpose)
    df_banco['savings'] = df_banco['savings'].map(diccionario_savings)
    df_banco['present_emp_since'] = df_banco['present_emp_since'].map(diccionario_present_es)
    df_banco['personal_status_sex'] = df_banco['personal_status_sex'].map(diccionario_personal_ss)
    df_banco['other_debtors'] = df_banco['other_debtors'].map(diccionario_other_debtors)
    df_banco['property'] = df_banco['property'].map(diccionario_property)
    df_banco['other_installment_plans'] = df_banco['other_installment_plans'].map(diccionario_other_ip)
    df_banco['housing'] = df_banco['housing'].map(diccionario_housing)
    df_banco['job'] = df_banco['job'].map(diccionario_job)
    df_banco['telephone'] = df_banco['telephone'].map(diccionario_telephone)
    df_banco['foreign_worker'] = df_banco['foreign_worker'].map(diccionario_foreign_worker)
```


```python
procesar_datos()
df_banco.head()
```





  <div id="df-186b7039-959e-4957-8524-95965aa7c28e" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>account_check_status</th>
      <th>duration_in_month</th>
      <th>credit_history</th>
      <th>purpose</th>
      <th>credit_amount</th>
      <th>savings</th>
      <th>present_emp_since</th>
      <th>installment_as_income_perc</th>
      <th>personal_status_sex</th>
      <th>other_debtors</th>
      <th>present_res_since</th>
      <th>property</th>
      <th>age</th>
      <th>other_installment_plans</th>
      <th>housing</th>
      <th>credits_this_bank</th>
      <th>job</th>
      <th>people_under_maintenance</th>
      <th>telephone</th>
      <th>foreign_worker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>1169</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>67</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>48</td>
      <td>3</td>
      <td>5</td>
      <td>5951</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>22</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>4</td>
      <td>12</td>
      <td>5</td>
      <td>8</td>
      <td>2096</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>49</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>42</td>
      <td>3</td>
      <td>4</td>
      <td>7882</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>45</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>24</td>
      <td>4</td>
      <td>1</td>
      <td>4870</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>53</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-186b7039-959e-4957-8524-95965aa7c28e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-186b7039-959e-4957-8524-95965aa7c28e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-186b7039-959e-4957-8524-95965aa7c28e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-8636884f-9bd5-439a-b661-64b90e20bc8e">
  <button class="colab-df-quickchart" onclick="quickchart('df-8636884f-9bd5-439a-b661-64b90e20bc8e')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-8636884f-9bd5-439a-b661-64b90e20bc8e button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




Volvemos a obtener la información luego de mapear los datos


```python
df_banco.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1000 entries, 0 to 999
    Data columns (total 21 columns):
     #   Column                      Non-Null Count  Dtype
    ---  ------                      --------------  -----
     0   default                     1000 non-null   int64
     1   account_check_status        1000 non-null   int64
     2   duration_in_month           1000 non-null   int64
     3   credit_history              1000 non-null   int64
     4   purpose                     1000 non-null   int64
     5   credit_amount               1000 non-null   int64
     6   savings                     1000 non-null   int64
     7   present_emp_since           1000 non-null   int64
     8   installment_as_income_perc  1000 non-null   int64
     9   personal_status_sex         1000 non-null   int64
     10  other_debtors               1000 non-null   int64
     11  present_res_since           1000 non-null   int64
     12  property                    1000 non-null   int64
     13  age                         1000 non-null   int64
     14  other_installment_plans     1000 non-null   int64
     15  housing                     1000 non-null   int64
     16  credits_this_bank           1000 non-null   int64
     17  job                         1000 non-null   int64
     18  people_under_maintenance    1000 non-null   int64
     19  telephone                   1000 non-null   int64
     20  foreign_worker              1000 non-null   int64
    dtypes: int64(21)
    memory usage: 171.9 KB
    

#**3. Exploración de Datos**


---


Separamos las columnas discretas para su posterior transformación


```python
variables_discretas = ['personal_status_sex', 'age', 'duration_in_month', 'credit_amount', 'default']
df_banco[variables_discretas].head()
```





  <div id="df-2ede3c36-d60c-4c32-991b-8beb56122683" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>personal_status_sex</th>
      <th>age</th>
      <th>duration_in_month</th>
      <th>credit_amount</th>
      <th>default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>67</td>
      <td>6</td>
      <td>1169</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>22</td>
      <td>48</td>
      <td>5951</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>49</td>
      <td>12</td>
      <td>2096</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>45</td>
      <td>42</td>
      <td>7882</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>53</td>
      <td>24</td>
      <td>4870</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2ede3c36-d60c-4c32-991b-8beb56122683')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2ede3c36-d60c-4c32-991b-8beb56122683 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2ede3c36-d60c-4c32-991b-8beb56122683');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e795b8ab-a865-4720-a2dd-ef1182000f6e">
  <button class="colab-df-quickchart" onclick="quickchart('df-e795b8ab-a865-4720-a2dd-ef1182000f6e')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e795b8ab-a865-4720-a2dd-ef1182000f6e button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




Realizamos la transformación de estas columnas mediante el método `cut`


```python
def feature_engineering():
  global df_banco  # Define las variables globales
  #Usaremos el diccionario de personal_status_sex para crear 2 nuevas columnas
  diccionario_personal_ss = {1 : 0, 2 : 1, 3 : 0, 4 : 0, 5 : 1}
  df_banco['sexo'] = df_banco['personal_status_sex'].map(diccionario_personal_ss)
  diccionario_personal_ss = {1 : 0, 2 : 0, 3 : 1, 4 : 0, 5 : 1}
  df_banco['estado_civil'] = df_banco['personal_status_sex'].map(diccionario_personal_ss)
  #Aquí usaremos la función cut para colocar las diferentes divisiones de las columnas a modificar
  #age, duration_in_month, credit_amount
  df_banco['rango_edad'] = pd.cut(x = df_banco['age'], bins=[18, 30, 40, 50, 60, 70, 80], labels = [1, 2, 3, 4, 5, 6])
  df_banco['rango_plazos_credito'] = pd.cut(x = df_banco['duration_in_month'], bins=[1, 12, 24, 36, 48, 60, 72], labels = [1, 2, 3, 4, 5, 6])
  df_banco['rango_valor_credito'] = pd.cut(x = df_banco['credit_amount'], bins=[1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000], labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
  #Finalmente eliminaremos las columnas base con las que trabajamos
  df_banco.drop(['personal_status_sex','age','duration_in_month', 'credit_amount'], axis=1, inplace=True)
```


```python
feature_engineering()
df_banco.head()
```





  <div id="df-7923f2a7-3f1c-46fc-88dd-2e444360cae2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>account_check_status</th>
      <th>credit_history</th>
      <th>purpose</th>
      <th>savings</th>
      <th>present_emp_since</th>
      <th>installment_as_income_perc</th>
      <th>other_debtors</th>
      <th>present_res_since</th>
      <th>property</th>
      <th>other_installment_plans</th>
      <th>housing</th>
      <th>credits_this_bank</th>
      <th>job</th>
      <th>people_under_maintenance</th>
      <th>telephone</th>
      <th>foreign_worker</th>
      <th>sexo</th>
      <th>estado_civil</th>
      <th>rango_edad</th>
      <th>rango_plazos_credito</th>
      <th>rango_valor_credito</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>8</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7923f2a7-3f1c-46fc-88dd-2e444360cae2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7923f2a7-3f1c-46fc-88dd-2e444360cae2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7923f2a7-3f1c-46fc-88dd-2e444360cae2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-9447d833-66db-47e8-b2ad-2ada3cc5e6d4">
  <button class="colab-df-quickchart" onclick="quickchart('df-9447d833-66db-47e8-b2ad-2ada3cc5e6d4')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-9447d833-66db-47e8-b2ad-2ada3cc5e6d4 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




Graficamos las columnas categóricas de nuestro set de datos


```python
atributos = ['sexo','estado_civil','rango_plazos_credito','rango_edad','default']

def analisis_exploratorio():

    # Calcula el número de filas necesarias
    nrows = len(atributos) // 2
    if len(atributos) % 2 != 0:
        nrows += 1

    fig, axs = plt.subplots(nrows, 2, figsize=(10, nrows*5))

    # Se asegúra de que axs sea una lista de listas en caso de que solo haya una fila
    if nrows == 1:
        axs = [axs]

    colors = ['blue', 'orange', 'green', 'red', 'purple']

    # Recorre cada variable discreta y crea un gráfico de barras
    for idx, var in enumerate(atributos):
        row = idx // 2
        col = idx % 2
        axs[row][col].bar(df_banco[var].value_counts().index, df_banco[var].value_counts().values, color=colors)
        axs[row][col].set_title(var)

    # Si el número de variables discretas es impar, elimina el último gráfico vacío
    if len(atributos) % 2 != 0:
        fig.delaxes(axs[row][1])

    plt.tight_layout()
    plt.show()
```


```python
analisis_exploratorio()
```


    
![png](Plantilla_Credit_Scoring_Prediction_files/Plantilla_Credit_Scoring_Prediction_32_0.png)
    


Aplicamos la técnica de sobremuestreo en nuestros datos para balancear la columna `default`


```python
# 'default' esta desbalanceado, usaremos SMOTE
sm = SMOTE(random_state=123)

# Dividimos la columna 'default' del resto del dataframe
X = df_banco.drop('default', axis=1)
y = df_banco['default']

# Aplico SMOTE
X, y = sm.fit_resample(X, y)

# Unimos de nuevo
X['default'] = y
df_banco = X

# Vemos como se distribuye ahora
df_banco['default'].value_counts()
```




    0    700
    1    700
    Name: default, dtype: int64



Visualizamos el mapa de calor de los datos


```python
plt.figure(figsize=(30, 10))
heatmap = sns.heatmap(df_banco.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1) ### AGREGADO vmin=-1
heatmap.tick_params(axis='both', which='major', labelsize=14)
plt.title('Mapa de Calor de Correlaciones', fontsize=18)
plt.show()
```


    
![png](Plantilla_Credit_Scoring_Prediction_files/Plantilla_Credit_Scoring_Prediction_36_0.png)
    


Visualizamos una tabla cruzada para ver como se comportan juntas las columnas `estado_civil` y `sexo`


```python
pd.crosstab(df_banco['estado_civil'], df_banco['sexo'])
```





  <div id="df-75ced2dc-2aa7-4509-b250-ef6b2fd365eb" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>sexo</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>estado_civil</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>339</td>
      <td>401</td>
    </tr>
    <tr>
      <th>1</th>
      <td>660</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-75ced2dc-2aa7-4509-b250-ef6b2fd365eb')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-75ced2dc-2aa7-4509-b250-ef6b2fd365eb button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-75ced2dc-2aa7-4509-b250-ef6b2fd365eb');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-9eb34562-4ee5-4def-b4c2-cc26c8cf4804">
  <button class="colab-df-quickchart" onclick="quickchart('df-9eb34562-4ee5-4def-b4c2-cc26c8cf4804')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-9eb34562-4ee5-4def-b4c2-cc26c8cf4804 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




#**4. Construcción de Modelos**


---


Dividimos los datos para entrenamiento y para prueba


```python
X_train, X_test, y_train, y_test = train_test_split(df_banco.drop('default', axis=1), df_banco['default'], test_size=0.2, random_state=123)
```

**Pycaret** es una biblioteca de Python de código abierto y de bajo código que se utiliza para el desarrollo de modelos de Machine Learning. Su objetivo es reducir el tiempo del ciclo de hipótesis a conocimiento en un experimento de ML. Permite a los científicos de datos realizar experimentos de extremo a extremo de manera rápida y eficiente.

Con **PyCaret**, puedes hacer muchas cosas:

* Aplicar imputación de valores perdidos, escalado, ingeniería de características o selección de características de una forma muy sencilla.

* Entrenar más de 100 modelos de machine learning, de todo tipo (clasificación, regresión, pronóstico) con una sola línea de código.

* Registrar los modelos entrenados en MLFlow de una forma muy sencilla.

* Crear una API o un Docker para poner el modelo en producción.

* Subir tu modelo a la nube para poder agilizar el despliegue en producción.

Además, **PyCaret** es compatible con cualquier tipo de notebook de Python y permite realizar comparaciones de varios modelos automáticamente. Por lo tanto, **PyCaret** es una herramienta muy útil que todo científico de datos debe conocer.


```python
## Corriendo esta celda se pueden ver las funciones extras que pueden en setup (inglés)
# print(setup.__doc__)
```

Construímos nuestro modelo Pycaret


```python
# Pycaret nos obliga a unir los datos de entrenamiento
X_train['default'] = y_train

# Setup es el primer y único paso obligatorio en cualquier experimento de aprendizaje automático que utilice PyCaret
s = setup(X_train, target = 'default')
```


<style type="text/css">
#T_45957_row9_col1 {
  background-color: lightgreen;
}
</style>
<table id="T_45957" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_45957_level0_col0" class="col_heading level0 col0" >Description</th>
      <th id="T_45957_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_45957_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_45957_row0_col0" class="data row0 col0" >Session id</td>
      <td id="T_45957_row0_col1" class="data row0 col1" >7299</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_45957_row1_col0" class="data row1 col0" >Target</td>
      <td id="T_45957_row1_col1" class="data row1 col1" >default</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_45957_row2_col0" class="data row2 col0" >Target type</td>
      <td id="T_45957_row2_col1" class="data row2 col1" >Binary</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_45957_row3_col0" class="data row3 col0" >Original data shape</td>
      <td id="T_45957_row3_col1" class="data row3 col1" >(1120, 22)</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_45957_row4_col0" class="data row4 col0" >Transformed data shape</td>
      <td id="T_45957_row4_col1" class="data row4 col1" >(1120, 48)</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_45957_row5_col0" class="data row5 col0" >Transformed train set shape</td>
      <td id="T_45957_row5_col1" class="data row5 col1" >(784, 48)</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_45957_row6_col0" class="data row6 col0" >Transformed test set shape</td>
      <td id="T_45957_row6_col1" class="data row6 col1" >(336, 48)</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_45957_row7_col0" class="data row7 col0" >Numeric features</td>
      <td id="T_45957_row7_col1" class="data row7 col1" >18</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_45957_row8_col0" class="data row8 col0" >Categorical features</td>
      <td id="T_45957_row8_col1" class="data row8 col1" >3</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_45957_row9_col0" class="data row9 col0" >Preprocess</td>
      <td id="T_45957_row9_col1" class="data row9 col1" >True</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_45957_row10_col0" class="data row10 col0" >Imputation type</td>
      <td id="T_45957_row10_col1" class="data row10 col1" >simple</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_45957_row11_col0" class="data row11 col0" >Numeric imputation</td>
      <td id="T_45957_row11_col1" class="data row11 col1" >mean</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_45957_row12_col0" class="data row12 col0" >Categorical imputation</td>
      <td id="T_45957_row12_col1" class="data row12 col1" >mode</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_45957_row13_col0" class="data row13 col0" >Maximum one-hot encoding</td>
      <td id="T_45957_row13_col1" class="data row13 col1" >25</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_45957_row14_col0" class="data row14 col0" >Encoding method</td>
      <td id="T_45957_row14_col1" class="data row14 col1" >None</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_45957_row15_col0" class="data row15 col0" >Fold Generator</td>
      <td id="T_45957_row15_col1" class="data row15 col1" >StratifiedKFold</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_45957_row16_col0" class="data row16 col0" >Fold Number</td>
      <td id="T_45957_row16_col1" class="data row16 col1" >10</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_45957_row17_col0" class="data row17 col0" >CPU Jobs</td>
      <td id="T_45957_row17_col1" class="data row17 col1" >-1</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_45957_row18_col0" class="data row18 col0" >Use GPU</td>
      <td id="T_45957_row18_col1" class="data row18 col1" >False</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_45957_row19_col0" class="data row19 col0" >Log Experiment</td>
      <td id="T_45957_row19_col1" class="data row19 col1" >False</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_45957_row20_col0" class="data row20 col0" >Experiment Name</td>
      <td id="T_45957_row20_col1" class="data row20 col1" >clf-default-name</td>
    </tr>
    <tr>
      <th id="T_45957_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_45957_row21_col0" class="data row21 col0" >USI</td>
      <td id="T_45957_row21_col1" class="data row21 col1" >872d</td>
    </tr>
  </tbody>
</table>



> AYUDA SOBRE METRICAS EN EL CREDITO BANCARIO

1. Área bajo la curva ROC (``AUC``): Esta métrica es crucial ya que mide la capacidad del modelo para distinguir entre los clientes que cumplirán con sus obligaciones de crédito y los que no. Un AUC-ROC más alto indica un mejor rendimiento del modelo.

2. Precisión (`Accuracy`): Esta métrica mide la proporción de predicciones correctas hechas por el modelo. En el contexto del scoring bancario, esto podría ser la proporción de clientes que el modelo predijo correctamente que pagarían o incumplirían sus obligaciones de crédito.

3. Sensibilidad (``Recall``): Esta métrica es importante en el scoring bancario porque mide la proporción de incumplimientos reales que el modelo es capaz de capturar.

4. Valor predictivo positivo (``Precision``): Esta métrica mide la proporción de incumplimientos predichos que son realmente incumplimientos.


```python
## Para ver opciones a usar dentro de compare_models correr esta celda
# print(compare_models.__doc__)
```

Como mencionamos en la explicación anterior, la metrica mas utilizada en el scoring bancario es `auc`


```python
# Comparar todos los modelos que incluye pycaret ordenado por su valor 'AUC'
best = compare_models(sort='AUC')
```






<style type="text/css">
#T_00acc th {
  text-align: left;
}
#T_00acc_row0_col0, #T_00acc_row0_col1, #T_00acc_row0_col3, #T_00acc_row0_col4, #T_00acc_row0_col5, #T_00acc_row0_col6, #T_00acc_row0_col7, #T_00acc_row1_col0, #T_00acc_row1_col1, #T_00acc_row1_col2, #T_00acc_row1_col3, #T_00acc_row1_col4, #T_00acc_row1_col5, #T_00acc_row1_col6, #T_00acc_row1_col7, #T_00acc_row2_col0, #T_00acc_row2_col2, #T_00acc_row2_col3, #T_00acc_row3_col0, #T_00acc_row3_col1, #T_00acc_row3_col2, #T_00acc_row3_col3, #T_00acc_row3_col4, #T_00acc_row3_col5, #T_00acc_row3_col6, #T_00acc_row3_col7, #T_00acc_row4_col0, #T_00acc_row4_col1, #T_00acc_row4_col2, #T_00acc_row4_col3, #T_00acc_row4_col4, #T_00acc_row4_col5, #T_00acc_row4_col6, #T_00acc_row4_col7, #T_00acc_row5_col0, #T_00acc_row5_col1, #T_00acc_row5_col2, #T_00acc_row5_col4, #T_00acc_row5_col5, #T_00acc_row5_col6, #T_00acc_row5_col7, #T_00acc_row6_col0, #T_00acc_row6_col1, #T_00acc_row6_col2, #T_00acc_row6_col3, #T_00acc_row6_col4, #T_00acc_row6_col5, #T_00acc_row6_col6, #T_00acc_row6_col7, #T_00acc_row7_col0, #T_00acc_row7_col1, #T_00acc_row7_col2, #T_00acc_row7_col3, #T_00acc_row7_col4, #T_00acc_row7_col5, #T_00acc_row7_col6, #T_00acc_row7_col7, #T_00acc_row8_col0, #T_00acc_row8_col1, #T_00acc_row8_col2, #T_00acc_row8_col3, #T_00acc_row8_col4, #T_00acc_row8_col5, #T_00acc_row8_col6, #T_00acc_row8_col7, #T_00acc_row9_col0, #T_00acc_row9_col1, #T_00acc_row9_col2, #T_00acc_row9_col3, #T_00acc_row9_col4, #T_00acc_row9_col5, #T_00acc_row9_col6, #T_00acc_row9_col7, #T_00acc_row10_col0, #T_00acc_row10_col1, #T_00acc_row10_col2, #T_00acc_row10_col3, #T_00acc_row10_col4, #T_00acc_row10_col5, #T_00acc_row10_col6, #T_00acc_row10_col7, #T_00acc_row11_col0, #T_00acc_row11_col1, #T_00acc_row11_col2, #T_00acc_row11_col3, #T_00acc_row11_col4, #T_00acc_row11_col5, #T_00acc_row11_col6, #T_00acc_row11_col7, #T_00acc_row12_col0, #T_00acc_row12_col1, #T_00acc_row12_col2, #T_00acc_row12_col3, #T_00acc_row12_col4, #T_00acc_row12_col5, #T_00acc_row12_col6, #T_00acc_row12_col7, #T_00acc_row13_col0, #T_00acc_row13_col1, #T_00acc_row13_col2, #T_00acc_row13_col3, #T_00acc_row13_col4, #T_00acc_row13_col5, #T_00acc_row13_col6, #T_00acc_row13_col7, #T_00acc_row14_col0, #T_00acc_row14_col1, #T_00acc_row14_col2, #T_00acc_row14_col3, #T_00acc_row14_col4, #T_00acc_row14_col5, #T_00acc_row14_col6, #T_00acc_row14_col7, #T_00acc_row15_col0, #T_00acc_row15_col1, #T_00acc_row15_col2, #T_00acc_row15_col3, #T_00acc_row15_col4, #T_00acc_row15_col5, #T_00acc_row15_col6, #T_00acc_row15_col7 {
  text-align: left;
}
#T_00acc_row0_col2, #T_00acc_row2_col1, #T_00acc_row2_col4, #T_00acc_row2_col5, #T_00acc_row2_col6, #T_00acc_row2_col7, #T_00acc_row5_col3 {
  text-align: left;
  background-color: yellow;
}
#T_00acc_row0_col8, #T_00acc_row1_col8, #T_00acc_row2_col8, #T_00acc_row3_col8, #T_00acc_row4_col8, #T_00acc_row5_col8, #T_00acc_row6_col8, #T_00acc_row7_col8, #T_00acc_row8_col8, #T_00acc_row9_col8, #T_00acc_row10_col8, #T_00acc_row11_col8, #T_00acc_row12_col8, #T_00acc_row14_col8, #T_00acc_row15_col8 {
  text-align: left;
  background-color: lightgrey;
}
#T_00acc_row13_col8 {
  text-align: left;
  background-color: yellow;
  background-color: lightgrey;
}
</style>
<table id="T_00acc" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_00acc_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_00acc_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_00acc_level0_col2" class="col_heading level0 col2" >AUC</th>
      <th id="T_00acc_level0_col3" class="col_heading level0 col3" >Recall</th>
      <th id="T_00acc_level0_col4" class="col_heading level0 col4" >Prec.</th>
      <th id="T_00acc_level0_col5" class="col_heading level0 col5" >F1</th>
      <th id="T_00acc_level0_col6" class="col_heading level0 col6" >Kappa</th>
      <th id="T_00acc_level0_col7" class="col_heading level0 col7" >MCC</th>
      <th id="T_00acc_level0_col8" class="col_heading level0 col8" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_00acc_level0_row0" class="row_heading level0 row0" >rf</th>
      <td id="T_00acc_row0_col0" class="data row0 col0" >Random Forest Classifier</td>
      <td id="T_00acc_row0_col1" class="data row0 col1" >0.8049</td>
      <td id="T_00acc_row0_col2" class="data row0 col2" >0.8904</td>
      <td id="T_00acc_row0_col3" class="data row0 col3" >0.8282</td>
      <td id="T_00acc_row0_col4" class="data row0 col4" >0.7906</td>
      <td id="T_00acc_row0_col5" class="data row0 col5" >0.8080</td>
      <td id="T_00acc_row0_col6" class="data row0 col6" >0.6099</td>
      <td id="T_00acc_row0_col7" class="data row0 col7" >0.6123</td>
      <td id="T_00acc_row0_col8" class="data row0 col8" >0.5500</td>
    </tr>
    <tr>
      <th id="T_00acc_level0_row1" class="row_heading level0 row1" >et</th>
      <td id="T_00acc_row1_col0" class="data row1 col0" >Extra Trees Classifier</td>
      <td id="T_00acc_row1_col1" class="data row1 col1" >0.7869</td>
      <td id="T_00acc_row1_col2" class="data row1 col2" >0.8873</td>
      <td id="T_00acc_row1_col3" class="data row1 col3" >0.8000</td>
      <td id="T_00acc_row1_col4" class="data row1 col4" >0.7799</td>
      <td id="T_00acc_row1_col5" class="data row1 col5" >0.7880</td>
      <td id="T_00acc_row1_col6" class="data row1 col6" >0.5739</td>
      <td id="T_00acc_row1_col7" class="data row1 col7" >0.5772</td>
      <td id="T_00acc_row1_col8" class="data row1 col8" >0.4320</td>
    </tr>
    <tr>
      <th id="T_00acc_level0_row2" class="row_heading level0 row2" >catboost</th>
      <td id="T_00acc_row2_col0" class="data row2 col0" >CatBoost Classifier</td>
      <td id="T_00acc_row2_col1" class="data row2 col1" >0.8060</td>
      <td id="T_00acc_row2_col2" class="data row2 col2" >0.8816</td>
      <td id="T_00acc_row2_col3" class="data row2 col3" >0.8282</td>
      <td id="T_00acc_row2_col4" class="data row2 col4" >0.7919</td>
      <td id="T_00acc_row2_col5" class="data row2 col5" >0.8085</td>
      <td id="T_00acc_row2_col6" class="data row2 col6" >0.6122</td>
      <td id="T_00acc_row2_col7" class="data row2 col7" >0.6151</td>
      <td id="T_00acc_row2_col8" class="data row2 col8" >1.5920</td>
    </tr>
    <tr>
      <th id="T_00acc_level0_row3" class="row_heading level0 row3" >xgboost</th>
      <td id="T_00acc_row3_col0" class="data row3 col0" >Extreme Gradient Boosting</td>
      <td id="T_00acc_row3_col1" class="data row3 col1" >0.7946</td>
      <td id="T_00acc_row3_col2" class="data row3 col2" >0.8719</td>
      <td id="T_00acc_row3_col3" class="data row3 col3" >0.8128</td>
      <td id="T_00acc_row3_col4" class="data row3 col4" >0.7847</td>
      <td id="T_00acc_row3_col5" class="data row3 col5" >0.7978</td>
      <td id="T_00acc_row3_col6" class="data row3 col6" >0.5893</td>
      <td id="T_00acc_row3_col7" class="data row3 col7" >0.5909</td>
      <td id="T_00acc_row3_col8" class="data row3 col8" >0.2110</td>
    </tr>
    <tr>
      <th id="T_00acc_level0_row4" class="row_heading level0 row4" >lightgbm</th>
      <td id="T_00acc_row4_col0" class="data row4 col0" >Light Gradient Boosting Machine</td>
      <td id="T_00acc_row4_col1" class="data row4 col1" >0.7933</td>
      <td id="T_00acc_row4_col2" class="data row4 col2" >0.8706</td>
      <td id="T_00acc_row4_col3" class="data row4 col3" >0.8205</td>
      <td id="T_00acc_row4_col4" class="data row4 col4" >0.7765</td>
      <td id="T_00acc_row4_col5" class="data row4 col5" >0.7974</td>
      <td id="T_00acc_row4_col6" class="data row4 col6" >0.5867</td>
      <td id="T_00acc_row4_col7" class="data row4 col7" >0.5885</td>
      <td id="T_00acc_row4_col8" class="data row4 col8" >0.2100</td>
    </tr>
    <tr>
      <th id="T_00acc_level0_row5" class="row_heading level0 row5" >gbc</th>
      <td id="T_00acc_row5_col0" class="data row5 col0" >Gradient Boosting Classifier</td>
      <td id="T_00acc_row5_col1" class="data row5 col1" >0.7907</td>
      <td id="T_00acc_row5_col2" class="data row5 col2" >0.8667</td>
      <td id="T_00acc_row5_col3" class="data row5 col3" >0.8359</td>
      <td id="T_00acc_row5_col4" class="data row5 col4" >0.7660</td>
      <td id="T_00acc_row5_col5" class="data row5 col5" >0.7980</td>
      <td id="T_00acc_row5_col6" class="data row5 col6" >0.5817</td>
      <td id="T_00acc_row5_col7" class="data row5 col7" >0.5868</td>
      <td id="T_00acc_row5_col8" class="data row5 col8" >0.2430</td>
    </tr>
    <tr>
      <th id="T_00acc_level0_row6" class="row_heading level0 row6" >ada</th>
      <td id="T_00acc_row6_col0" class="data row6 col0" >Ada Boost Classifier</td>
      <td id="T_00acc_row6_col1" class="data row6 col1" >0.7705</td>
      <td id="T_00acc_row6_col2" class="data row6 col2" >0.8381</td>
      <td id="T_00acc_row6_col3" class="data row6 col3" >0.7744</td>
      <td id="T_00acc_row6_col4" class="data row6 col4" >0.7680</td>
      <td id="T_00acc_row6_col5" class="data row6 col5" >0.7696</td>
      <td id="T_00acc_row6_col6" class="data row6 col6" >0.5411</td>
      <td id="T_00acc_row6_col7" class="data row6 col7" >0.5433</td>
      <td id="T_00acc_row6_col8" class="data row6 col8" >0.3110</td>
    </tr>
    <tr>
      <th id="T_00acc_level0_row7" class="row_heading level0 row7" >knn</th>
      <td id="T_00acc_row7_col0" class="data row7 col0" >K Neighbors Classifier</td>
      <td id="T_00acc_row7_col1" class="data row7 col1" >0.7615</td>
      <td id="T_00acc_row7_col2" class="data row7 col2" >0.8347</td>
      <td id="T_00acc_row7_col3" class="data row7 col3" >0.8282</td>
      <td id="T_00acc_row7_col4" class="data row7 col4" >0.7315</td>
      <td id="T_00acc_row7_col5" class="data row7 col5" >0.7755</td>
      <td id="T_00acc_row7_col6" class="data row7 col6" >0.5234</td>
      <td id="T_00acc_row7_col7" class="data row7 col7" >0.5303</td>
      <td id="T_00acc_row7_col8" class="data row7 col8" >0.2420</td>
    </tr>
    <tr>
      <th id="T_00acc_level0_row8" class="row_heading level0 row8" >lr</th>
      <td id="T_00acc_row8_col0" class="data row8 col0" >Logistic Regression</td>
      <td id="T_00acc_row8_col1" class="data row8 col1" >0.7654</td>
      <td id="T_00acc_row8_col2" class="data row8 col2" >0.8319</td>
      <td id="T_00acc_row8_col3" class="data row8 col3" >0.7769</td>
      <td id="T_00acc_row8_col4" class="data row8 col4" >0.7590</td>
      <td id="T_00acc_row8_col5" class="data row8 col5" >0.7664</td>
      <td id="T_00acc_row8_col6" class="data row8 col6" >0.5310</td>
      <td id="T_00acc_row8_col7" class="data row8 col7" >0.5332</td>
      <td id="T_00acc_row8_col8" class="data row8 col8" >1.4900</td>
    </tr>
    <tr>
      <th id="T_00acc_level0_row9" class="row_heading level0 row9" >lda</th>
      <td id="T_00acc_row9_col0" class="data row9 col0" >Linear Discriminant Analysis</td>
      <td id="T_00acc_row9_col1" class="data row9 col1" >0.7667</td>
      <td id="T_00acc_row9_col2" class="data row9 col2" >0.8298</td>
      <td id="T_00acc_row9_col3" class="data row9 col3" >0.7846</td>
      <td id="T_00acc_row9_col4" class="data row9 col4" >0.7573</td>
      <td id="T_00acc_row9_col5" class="data row9 col5" >0.7695</td>
      <td id="T_00acc_row9_col6" class="data row9 col6" >0.5336</td>
      <td id="T_00acc_row9_col7" class="data row9 col7" >0.5358</td>
      <td id="T_00acc_row9_col8" class="data row9 col8" >0.1110</td>
    </tr>
    <tr>
      <th id="T_00acc_level0_row10" class="row_heading level0 row10" >nb</th>
      <td id="T_00acc_row10_col0" class="data row10 col0" >Naive Bayes</td>
      <td id="T_00acc_row10_col1" class="data row10 col1" >0.5305</td>
      <td id="T_00acc_row10_col2" class="data row10 col2" >0.7392</td>
      <td id="T_00acc_row10_col3" class="data row10 col3" >0.1000</td>
      <td id="T_00acc_row10_col4" class="data row10 col4" >0.7033</td>
      <td id="T_00acc_row10_col5" class="data row10 col5" >0.1733</td>
      <td id="T_00acc_row10_col6" class="data row10 col6" >0.0571</td>
      <td id="T_00acc_row10_col7" class="data row10 col7" >0.1105</td>
      <td id="T_00acc_row10_col8" class="data row10 col8" >0.1900</td>
    </tr>
    <tr>
      <th id="T_00acc_level0_row11" class="row_heading level0 row11" >dt</th>
      <td id="T_00acc_row11_col0" class="data row11 col0" >Decision Tree Classifier</td>
      <td id="T_00acc_row11_col1" class="data row11 col1" >0.7283</td>
      <td id="T_00acc_row11_col2" class="data row11 col2" >0.7287</td>
      <td id="T_00acc_row11_col3" class="data row11 col3" >0.7718</td>
      <td id="T_00acc_row11_col4" class="data row11 col4" >0.7111</td>
      <td id="T_00acc_row11_col5" class="data row11 col5" >0.7383</td>
      <td id="T_00acc_row11_col6" class="data row11 col6" >0.4570</td>
      <td id="T_00acc_row11_col7" class="data row11 col7" >0.4613</td>
      <td id="T_00acc_row11_col8" class="data row11 col8" >0.4160</td>
    </tr>
    <tr>
      <th id="T_00acc_level0_row12" class="row_heading level0 row12" >qda</th>
      <td id="T_00acc_row12_col0" class="data row12 col0" >Quadratic Discriminant Analysis</td>
      <td id="T_00acc_row12_col1" class="data row12 col1" >0.5141</td>
      <td id="T_00acc_row12_col2" class="data row12 col2" >0.7138</td>
      <td id="T_00acc_row12_col3" class="data row12 col3" >0.1128</td>
      <td id="T_00acc_row12_col4" class="data row12 col4" >0.6535</td>
      <td id="T_00acc_row12_col5" class="data row12 col5" >0.1602</td>
      <td id="T_00acc_row12_col6" class="data row12 col6" >0.0244</td>
      <td id="T_00acc_row12_col7" class="data row12 col7" >0.0652</td>
      <td id="T_00acc_row12_col8" class="data row12 col8" >0.3740</td>
    </tr>
    <tr>
      <th id="T_00acc_level0_row13" class="row_heading level0 row13" >dummy</th>
      <td id="T_00acc_row13_col0" class="data row13 col0" >Dummy Classifier</td>
      <td id="T_00acc_row13_col1" class="data row13 col1" >0.5025</td>
      <td id="T_00acc_row13_col2" class="data row13 col2" >0.5000</td>
      <td id="T_00acc_row13_col3" class="data row13 col3" >0.0000</td>
      <td id="T_00acc_row13_col4" class="data row13 col4" >0.0000</td>
      <td id="T_00acc_row13_col5" class="data row13 col5" >0.0000</td>
      <td id="T_00acc_row13_col6" class="data row13 col6" >0.0000</td>
      <td id="T_00acc_row13_col7" class="data row13 col7" >0.0000</td>
      <td id="T_00acc_row13_col8" class="data row13 col8" >0.1030</td>
    </tr>
    <tr>
      <th id="T_00acc_level0_row14" class="row_heading level0 row14" >svm</th>
      <td id="T_00acc_row14_col0" class="data row14 col0" >SVM - Linear Kernel</td>
      <td id="T_00acc_row14_col1" class="data row14 col1" >0.6964</td>
      <td id="T_00acc_row14_col2" class="data row14 col2" >0.0000</td>
      <td id="T_00acc_row14_col3" class="data row14 col3" >0.7872</td>
      <td id="T_00acc_row14_col4" class="data row14 col4" >0.7074</td>
      <td id="T_00acc_row14_col5" class="data row14 col5" >0.7153</td>
      <td id="T_00acc_row14_col6" class="data row14 col6" >0.3940</td>
      <td id="T_00acc_row14_col7" class="data row14 col7" >0.4377</td>
      <td id="T_00acc_row14_col8" class="data row14 col8" >0.2300</td>
    </tr>
    <tr>
      <th id="T_00acc_level0_row15" class="row_heading level0 row15" >ridge</th>
      <td id="T_00acc_row15_col0" class="data row15 col0" >Ridge Classifier</td>
      <td id="T_00acc_row15_col1" class="data row15 col1" >0.7718</td>
      <td id="T_00acc_row15_col2" class="data row15 col2" >0.0000</td>
      <td id="T_00acc_row15_col3" class="data row15 col3" >0.7923</td>
      <td id="T_00acc_row15_col4" class="data row15 col4" >0.7613</td>
      <td id="T_00acc_row15_col5" class="data row15 col5" >0.7752</td>
      <td id="T_00acc_row15_col6" class="data row15 col6" >0.5438</td>
      <td id="T_00acc_row15_col7" class="data row15 col7" >0.5462</td>
      <td id="T_00acc_row15_col8" class="data row15 col8" >0.1740</td>
    </tr>
  </tbody>
</table>




    Processing:   0%|          | 0/69 [00:00<?, ?it/s]







```python
## Para ver opciones a usar dentro de plot_models correr esta celda
# print(plot_model.__doc__)
```

Visualizamos los resultados del entrenamiento con el mejor de los modelos entrenados


```python
# Visualiza las transformaciones que aplicó pycaret de forma automática
plot_model(best, plot = 'pipeline')
```


    
![png](Plantilla_Credit_Scoring_Prediction_files/Plantilla_Credit_Scoring_Prediction_52_0.png)
    



```python
# Generamos una matriz de confusión
plot_model(best, plot = 'confusion_matrix')
```






    
![png](Plantilla_Credit_Scoring_Prediction_files/Plantilla_Credit_Scoring_Prediction_53_1.png)
    



```python
# Visualizar el área bajo la curva
plot_model(best, plot = 'auc')
```






    
![png](Plantilla_Credit_Scoring_Prediction_files/Plantilla_Credit_Scoring_Prediction_54_1.png)
    



```python
# Visualizar los errores en la prediccion
plot_model(best, plot = 'error')
```






    
![png](Plantilla_Credit_Scoring_Prediction_files/Plantilla_Credit_Scoring_Prediction_55_1.png)
    



```python
# Visualizar la importancia de las variables
plot_model(best, plot = 'feature_all')
```






    
![png](Plantilla_Credit_Scoring_Prediction_files/Plantilla_Credit_Scoring_Prediction_56_1.png)
    



```python
# Visualizar la curva de validación cruzada del modelo
plot_model(best, plot = 'vc')
```






    
![png](Plantilla_Credit_Scoring_Prediction_files/Plantilla_Credit_Scoring_Prediction_57_1.png)
    


Finalizamos el entrenamiento, guardamos el mejor modelo que encontro pycaret y realizamos las predicciones para nuestros datos de prueba


```python
# finalizar el modelo
final_best = finalize_model(best)

# guarda el modelo como archivo pickle
save_model(final_best, 'credit_score')

# Generamos las predicciones con los datos de test
predictions = predict_model(final_best, data=X_test)
predictions.head()
```

    Transformation Pipeline and Model Successfully Saved
    









  <div id="df-b2613c11-74c1-46c0-9a17-7cc43b1d1a6b" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>account_check_status</th>
      <th>credit_history</th>
      <th>purpose</th>
      <th>savings</th>
      <th>present_emp_since</th>
      <th>installment_as_income_perc</th>
      <th>other_debtors</th>
      <th>present_res_since</th>
      <th>property</th>
      <th>other_installment_plans</th>
      <th>...</th>
      <th>people_under_maintenance</th>
      <th>telephone</th>
      <th>foreign_worker</th>
      <th>sexo</th>
      <th>estado_civil</th>
      <th>rango_edad</th>
      <th>rango_plazos_credito</th>
      <th>rango_valor_credito</th>
      <th>prediction_label</th>
      <th>prediction_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>952</th>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>908</th>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>221</th>
      <td>1</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0.61</td>
    </tr>
    <tr>
      <th>310</th>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>218</th>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>0.91</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b2613c11-74c1-46c0-9a17-7cc43b1d1a6b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b2613c11-74c1-46c0-9a17-7cc43b1d1a6b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b2613c11-74c1-46c0-9a17-7cc43b1d1a6b');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-ef456a22-3e4c-406e-9a02-54d1fd3f8fa5">
  <button class="colab-df-quickchart" onclick="quickchart('df-ef456a22-3e4c-406e-9a02-54d1fd3f8fa5')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-ef456a22-3e4c-406e-9a02-54d1fd3f8fa5 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




#**5. Evaluación y Selección del Modelo**


---


Evaluamos los resultados del modelo seleccionado en 5 metricas diferentes


```python
resultados = {
  "AUC score": round(roc_auc_score(y_test, predictions['prediction_label']), 3),
  "Accuracy score": round(accuracy_score(y_test, predictions['prediction_label']), 3),
  "Recall score": round(recall_score(y_test, predictions['prediction_label']), 3),
  "Precision score": round(precision_score(y_test, predictions['prediction_label']), 3),
  "F1 score": round(f1_score(y_test, predictions['prediction_label']), 3)
}

print(f"El mejor modelo que seleccionó pycaret es: {str(final_best.base_estimator_).split('(')[0]}")
print("\nLos resultados para nuestros datos de prueba fueron los siguiente:\n")
pd.DataFrame(resultados.items(), columns=['Metrica', 'Valor'])
```

    El mejor modelo que seleccionó pycaret es: DecisionTreeClassifier
    
    Los resultados para nuestros datos de prueba fueron los siguiente:
    
    





  <div id="df-268109dc-07b3-427d-859b-c8bae4539011" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metrica</th>
      <th>Valor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AUC score</td>
      <td>0.835</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Accuracy score</td>
      <td>0.836</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Recall score</td>
      <td>0.853</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Precision score</td>
      <td>0.830</td>
    </tr>
    <tr>
      <th>4</th>
      <td>F1 score</td>
      <td>0.841</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-268109dc-07b3-427d-859b-c8bae4539011')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-268109dc-07b3-427d-859b-c8bae4539011 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-268109dc-07b3-427d-859b-c8bae4539011');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-2ac6cb75-6546-457d-a123-be4680cdf568">
  <button class="colab-df-quickchart" onclick="quickchart('df-2ac6cb75-6546-457d-a123-be4680cdf568')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-2ac6cb75-6546-457d-a123-be4680cdf568 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>



