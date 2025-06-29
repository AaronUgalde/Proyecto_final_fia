import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import os

# Determinar ruta base (misma carpeta que este script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'rf_model.pkl')

# Cargar el modelo desde el archivo .pkl
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    print(f"Advertencia: No se encontró el modelo en {MODEL_PATH}")

# Opciones para dropdowns (traducidas al español)
order_sleep = ['Menos de 5 horas', '5-6 horas', 'Otras', '7-8 horas', 'Más de 8 horas']
order_dietary = ['No saludable', 'Moderada', 'Saludable']

class ClasificadorSaludMentalApp:
    def __init__(self, root):
        self.root = root
        self.configurar_ventana()
        self.configurar_variables()
        self.crear_widgets()
        
    def configurar_ventana(self):
        """Configurar la ventana principal"""
        self.root.title("🧠 Clasificador de Depresion")
        self.root.geometry("500x700")
        self.root.resizable(True, True)
        
        # Configurar estilo
        style = ttk.Style()
        style.theme_use('clam')
        
    def configurar_variables(self):
        """Inicializar variables de entrada"""
        self.genero_var = tk.StringVar(value='Hombre')
        self.edad_var = tk.StringVar()
        self.presion_academica_var = tk.IntVar(value=3)
        self.promedio_var = tk.StringVar()
        self.satisfaccion_laboral_var = tk.IntVar(value=2)
        self.sueno_var = tk.StringVar(value=order_sleep[3])
        self.dieta_var = tk.StringVar(value=order_dietary[1])
        self.nivel_educativo_var = tk.StringVar(value='licenciatura')
        self.suicida_var = tk.StringVar(value='NO')
        self.horas_var = tk.StringVar()
        self.estres_financiero_var = tk.IntVar(value=3)
        self.historial_familiar_var = tk.StringVar(value='NO')
        
    def crear_widgets(self):
        """Crear todos los widgets de la interfaz"""
        # Frame principal con scroll
        frame_principal = ttk.Frame(self.root)
        frame_principal.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Canvas y scrollbar
        canvas = tk.Canvas(frame_principal)
        scrollbar = ttk.Scrollbar(frame_principal, orient="vertical", command=canvas.yview)
        frame_desplazable = ttk.Frame(canvas)
        
        frame_desplazable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=frame_desplazable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind scroll con mouse
        def _en_rueda_mouse(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _en_rueda_mouse)
        
        # Título
        etiqueta_titulo = tk.Label(frame_desplazable, 
                              text="🧠 Clasificador de Depresion",
                              font=('Arial', 16, 'bold'),
                              fg='#2c3e50')
        etiqueta_titulo.pack(pady=(0, 5))
        
        etiqueta_subtitulo = tk.Label(frame_desplazable,
                                 text="Complete la información para obtener una evaluación",
                                 font=('Arial', 10),
                                 fg='#34495e')
        etiqueta_subtitulo.pack(pady=(0, 20))
        
        if not model_loaded:
            etiqueta_advertencia = tk.Label(frame_desplazable,
                                   text="⚠️ Modelo no encontrado - Modo demostración",
                                   font=('Arial', 10),
                                   fg='#e74c3c')
            etiqueta_advertencia.pack(pady=(0, 10))
        
        # Sección 1: Información Personal
        frame_personal = ttk.LabelFrame(frame_desplazable, text="👤 Información Personal", padding=10)
        frame_personal.pack(fill='x', pady=(0, 10))
        
        # Género
        tk.Label(frame_personal, text="Género:", font=('Arial', 10)).grid(row=0, column=0, sticky='w', pady=5)
        frame_genero = tk.Frame(frame_personal)
        frame_genero.grid(row=0, column=1, sticky='w', padx=10)
        tk.Radiobutton(frame_genero, text='Hombre', variable=self.genero_var, value='Hombre').pack(side='left')
        tk.Radiobutton(frame_genero, text='Mujer', variable=self.genero_var, value='Mujer').pack(side='left', padx=(10, 0))
        
        # Edad
        tk.Label(frame_personal, text="Edad:", font=('Arial', 10)).grid(row=1, column=0, sticky='w', pady=5)
        entrada_edad = tk.Entry(frame_personal, textvariable=self.edad_var, width=15)
        entrada_edad.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        
        # Sección 2: Información Académica/Laboral
        frame_academico = ttk.LabelFrame(frame_desplazable, text="🎓 Información Académica/Laboral", padding=10)
        frame_academico.pack(fill='x', pady=(0, 10))
        
        # Nivel educativo
        tk.Label(frame_academico, text="Nivel educativo:", font=('Arial', 10)).grid(row=0, column=0, sticky='w', pady=5)
        combo_nivel = ttk.Combobox(frame_academico, textvariable=self.nivel_educativo_var,
                                   values=['licenciatura', 'maestria', 'doctorado', 'secundaria'], 
                                   state='readonly', width=15)
        combo_nivel.grid(row=0, column=1, sticky='w', padx=10, pady=5)
        
        # Promedio escolar
        tk.Label(frame_academico, text="Promedio escolar (0-10):", font=('Arial', 10)).grid(row=1, column=0, sticky='w', pady=5)
        entrada_promedio = tk.Entry(frame_academico, textvariable=self.promedio_var, width=15)
        entrada_promedio.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        
        # Presión académica
        tk.Label(frame_academico, text="Presión académica (0-5):", font=('Arial', 10)).grid(row=2, column=0, sticky='w', pady=5)
        frame_pa = tk.Frame(frame_academico)
        frame_pa.grid(row=2, column=1, sticky='w', padx=10, pady=5)
        tk.Scale(frame_pa, from_=0, to=5, variable=self.presion_academica_var,
                orient='horizontal', length=150).pack(side='left')
        tk.Label(frame_pa, textvariable=self.presion_academica_var, 
                font=('Arial', 10, 'bold'), width=3).pack(side='left', padx=(5, 0))
        
        # Satisfacción laboral
        tk.Label(frame_academico, text="Satisfacción laboral (0-4):", font=('Arial', 10)).grid(row=3, column=0, sticky='w', pady=5)
        frame_sl = tk.Frame(frame_academico)
        frame_sl.grid(row=3, column=1, sticky='w', padx=10, pady=5)
        tk.Scale(frame_sl, from_=0, to=4, variable=self.satisfaccion_laboral_var,
                orient='horizontal', length=150).pack(side='left')
        tk.Label(frame_sl, textvariable=self.satisfaccion_laboral_var, 
                font=('Arial', 10, 'bold'), width=3).pack(side='left', padx=(5, 0))
        
        # Horas de trabajo/estudio
        tk.Label(frame_academico, text="Horas trabajo/estudio/día:", font=('Arial', 10)).grid(row=4, column=0, sticky='w', pady=5)
        entrada_horas = tk.Entry(frame_academico, textvariable=self.horas_var, width=15)
        entrada_horas.grid(row=4, column=1, sticky='w', padx=10, pady=5)
        
        # Sección 3: Estilo de Vida
        frame_estilo_vida = ttk.LabelFrame(frame_desplazable, text="🌱 Estilo de Vida", padding=10)
        frame_estilo_vida.pack(fill='x', pady=(0, 10))
        
        # Duración del sueño
        tk.Label(frame_estilo_vida, text="Duración del sueño:", font=('Arial', 10)).grid(row=0, column=0, sticky='w', pady=5)
        combo_sueno = ttk.Combobox(frame_estilo_vida, textvariable=self.sueno_var,
                                  values=order_sleep, state='readonly', width=20)
        combo_sueno.grid(row=0, column=1, sticky='w', padx=10, pady=5)
        
        # Hábitos alimentarios
        tk.Label(frame_estilo_vida, text="Hábitos alimentarios:", font=('Arial', 10)).grid(row=1, column=0, sticky='w', pady=5)
        combo_dieta = ttk.Combobox(frame_estilo_vida, textvariable=self.dieta_var,
                                 values=order_dietary, state='readonly', width=20)
        combo_dieta.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        
        # Estrés financiero
        tk.Label(frame_estilo_vida, text="Estrés financiero (1-5):", font=('Arial', 10)).grid(row=2, column=0, sticky='w', pady=5)
        frame_ef = tk.Frame(frame_estilo_vida)
        frame_ef.grid(row=2, column=1, sticky='w', padx=10, pady=5)
        tk.Scale(frame_ef, from_=1, to=5, variable=self.estres_financiero_var,
                orient='horizontal', length=150).pack(side='left')
        tk.Label(frame_ef, textvariable=self.estres_financiero_var, 
                font=('Arial', 10, 'bold'), width=3).pack(side='left', padx=(5, 0))
        
        # Sección 4: Salud Mental
        frame_salud = ttk.LabelFrame(frame_desplazable, text="🧠 Información de Salud Mental", padding=10)
        frame_salud.pack(fill='x', pady=(0, 10))
        
        # Pensamientos suicidas
        tk.Label(frame_salud, text="¿Pensamientos suicidas?:", font=('Arial', 10)).grid(row=0, column=0, sticky='w', pady=5)
        frame_su = tk.Frame(frame_salud)
        frame_su.grid(row=0, column=1, sticky='w', padx=10, pady=5)
        tk.Radiobutton(frame_su, text='Sí', variable=self.suicida_var, value='SÍ').pack(side='left')
        tk.Radiobutton(frame_su, text='No', variable=self.suicida_var, value='NO').pack(side='left', padx=(10, 0))
        
        # Historial familiar
        tk.Label(frame_salud, text="¿Historial familiar de enfermedades mentales?:", font=('Arial', 10)).grid(row=1, column=0, sticky='w', pady=5)
        frame_hf = tk.Frame(frame_salud)
        frame_hf.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        tk.Radiobutton(frame_hf, text='Sí', variable=self.historial_familiar_var, value='SÍ').pack(side='left')
        tk.Radiobutton(frame_hf, text='No', variable=self.historial_familiar_var, value='NO').pack(side='left', padx=(10, 0))
        
        # Botones
        frame_botones = tk.Frame(frame_desplazable)
        frame_botones.pack(fill='x', pady=20)
        
        boton_limpiar = tk.Button(frame_botones, text='🔄 Limpiar', command=self.limpiar_formulario,
                             bg="#0997A1", fg='black', font=('Arial', 11, 'bold'))
        boton_limpiar.pack(side='left', padx=(0, 10))
        
        boton_clasificar = tk.Button(frame_botones, text='🔍 Clasificar', command=self.clasificar,
                               bg='#2ecc71', fg='black', font=('Arial', 11, 'bold'))
        boton_clasificar.pack(side='left')
        
        # Resultado
        self.frame_resultado = ttk.LabelFrame(frame_desplazable, text="📊 Resultado", padding=10)
        self.frame_resultado.pack(fill='x', pady=(0, 10))
        
        self.etiqueta_resultado = tk.Label(self.frame_resultado, 
                                   text="Complete el formulario y presione 'Clasificar'",
                                   font=('Arial', 11),
                                   fg='#7f8c8d')
        self.etiqueta_resultado.pack()
        
    def validar_entradas(self):
        """Validar todas las entradas"""
        errores = []
        
        # Validar edad
        if not self.edad_var.get().strip():
            errores.append("Por favor ingrese la edad")
        else:
            try:
                edad = int(self.edad_var.get())
                if edad < 0 or edad > 120:
                    errores.append("La edad debe estar entre 0 y 120 años")
            except ValueError:
                errores.append("Por favor ingrese una edad válida")
                
        # Validar promedio
        if not self.promedio_var.get().strip():
            errores.append("Por favor ingrese el promedio escolar")
        else:
            try:
                promedio = float(self.promedio_var.get())
                if promedio < 0 or promedio > 10:
                    errores.append("El promedio debe estar entre 0 y 10")
            except ValueError:
                errores.append("Por favor ingrese un promedio válido")
                
        # Validar horas
        if not self.horas_var.get().strip():
            errores.append("Por favor ingrese las horas de trabajo/estudio")
        else:
            try:
                horas = float(self.horas_var.get())
                if horas < 0 or horas > 24:
                    errores.append("Las horas deben estar entre 0 y 24")
            except ValueError:
                errores.append("Por favor ingrese un número válido de horas")
                
        return errores
        
    def clasificar(self):
        """Realizar clasificación"""
        # Validar entradas
        errores = self.validar_entradas()
        if errores:
            messagebox.showerror("Errores de validación", "\n".join(errores))
            return
            
        if not model_loaded:
            # Simular predicción para demostración
            resultado = "Modelo no disponible"
            self.mostrar_resultado(resultado, es_demo=True)
            return
            
        try:
            # Mapeo de variables categóricas a numérico
            codigo_genero = 1 if self.genero_var.get() == 'Mujer' else 0
            codigo_sueno = order_sleep.index(self.sueno_var.get())
            codigo_dieta = order_dietary.index(self.dieta_var.get())
            mapa_nivel = {'licenciatura': 0, 'maestria': 1, 'doctorado': 2, 'secundaria': 3}
            codigo_suicida = 1 if self.suicida_var.get() == 'SÍ' else 0
            codigo_historial_familiar = 1 if self.historial_familiar_var.get() == 'SÍ' else 0

            # Crear arreglo de características
            caracteristicas = [
                codigo_genero,
                int(self.edad_var.get()),
                self.presion_academica_var.get(),
                float(self.promedio_var.get()),
                self.satisfaccion_laboral_var.get(),
                codigo_sueno,
                codigo_dieta,
                mapa_nivel[self.nivel_educativo_var.get()],
                codigo_suicida,
                float(self.horas_var.get()),
                self.estres_financiero_var.get(),
                codigo_historial_familiar
            ]
            
            arr = np.array(caracteristicas).reshape(1, -1)
            pred = model.predict(arr)[0]
            if(pred == 1):
                pred = "Posible depresión"
            else:
                pred = "Sin depresión"
            self.mostrar_resultado(pred)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en la clasificación: {str(e)}")
            
    def mostrar_resultado(self, prediccion, es_demo=False):
        """Mostrar resultado de la clasificación"""
        # Limpiar resultado anterior
        for widget in self.frame_resultado.winfo_children():
            widget.destroy()
            
        texto_demo = " (DEMOSTRACIÓN)" if es_demo else ""
        
        # Resultado
        texto_resultado = tk.Label(self.frame_resultado,
                              text=f"Predicción: {prediccion}{texto_demo}",
                              font=('Arial', 14, 'bold'),
                              fg='#2c3e50')
        texto_resultado.pack(pady=(0, 5))
        
        # Nota
        if es_demo:
            nota_demo = tk.Label(self.frame_resultado,
                               text="⚠️ Resultado simulado - Modelo no disponible",
                               font=('Arial', 10),
                               fg='#e74c3c')
            nota_demo.pack()
        else:
            descargo = tk.Label(self.frame_resultado,
                                text="⚠️ Este resultado es solo orientativo.\nConsulte a un profesional de la salud mental.",
                                font=('Arial', 9),
                                fg='#e67e22',
                                justify='center')
            descargo.pack()
            
    def limpiar_formulario(self):
        """Limpiar formulario"""
        self.genero_var.set('Hombre')
        self.edad_var.set('')
        self.presion_academica_var.set(3)
        self.promedio_var.set('')
        self.satisfaccion_laboral_var.set(2)
        self.sueno_var.set(order_sleep[3])
        self.dieta_var.set(order_dietary[1])
        self.nivel_educativo_var.set('licenciatura')
        self.suicida_var.set('NO')
        self.horas_var.set('')
        self.estres_financiero_var.set(3)
        self.historial_familiar_var.set('NO')
        
        # Limpiar resultado
        for widget in self.frame_resultado.winfo_children():
            widget.destroy()
            
        etiqueta_resultado = tk.Label(self.frame_resultado, 
                              text="Complete el formulario y presione 'Clasificar'",
                              font=('Arial', 11),
                              fg='#7f8c8d')
        etiqueta_resultado.pack()

def crear_app():
    """Función principal para crear la aplicación"""
    root = tk.Tk()
    app = ClasificadorSaludMentalApp(root)
    return root

if __name__ == '__main__':
    app = crear_app()
    app.mainloop()