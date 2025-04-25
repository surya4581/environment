import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.over_sampling import SMOTE
import numpy as np

# Load and preprocess dataset
try:
    data = pd.read_csv("dataset.csv")
    
    # Check initial class distribution
    print("Initial Class Distribution:")
    print(data['disease'].value_counts())

    # Filter out rare diseases
    MIN_SAMPLES = 2
    valid_classes = data['disease'].value_counts()[data['disease'].value_counts() >= MIN_SAMPLES].index
    data = data[data['disease'].isin(valid_classes)]

    if data.empty:
        messagebox.showerror("Error", "Insufficient data after filtering!\nNeed at least 2 cases per disease.")
        exit()

    print("\nFiltered Class Distribution:")
    print(data['disease'].value_counts())

    # Handle symptom synonyms
    symptom_synonyms = {
        "manic": "mood swings",
        "loose stools": "diarrhea",
        "throw up": "vomiting",
        "running nose": "runny nose",
        "stomach ache": "abdominal pain",
        "pinkeye": "conjunctivitis",
        "stuffy nose": "congestion"
    }

    data['symptoms'] = data['symptoms'].apply(
        lambda x: [symptom_synonyms.get(s.strip().lower(), s.strip().lower()) for s in x.split(',')]
    )

    # Prepare features
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(data['symptoms'])
    y = data['disease']

    # Conditional SMOTE Application
    if len(y.unique()) > 1:
        min_samples = max(y.value_counts().min(), 1)  # Ensure minimum 1 sample
        smote = SMOTE(random_state=42, k_neighbors=min(min_samples-1, 1))
        X_balanced, y_balanced = smote.fit_resample(X, y)
    else:
        X_balanced, y_balanced = X, y

    if len(X_balanced) < 2:
        messagebox.showerror("Error", "Not enough data for training!\nPlease check your dataset.")
        exit()

    # Calculate class weights and train model
    class_distribution = pd.Series(y_balanced).value_counts(normalize=True)
    class_weights = {cls: 1.0/weight for cls, weight in class_distribution.items()}

    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    }

    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    
    model = GridSearchCV(
        SVC(probability=True, class_weight=class_weights),
        param_grid,
        cv=min(3, len(X_balanced)//2),  # Dynamic CV folds
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    print("\nBest parameters found:", model.best_params_)

except Exception as e:
    messagebox.showerror("Critical Error", f"Failed to initialize model: {str(e)}")
    exit()

# Enhanced GUI with Comprehensive Symptom Selection
class DiseasePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Medical Diagnosis Predictor")
        self.root.geometry("950x750")

        # Initialize variables
        self.symptom_vars = {}        # Add this
        self.common_symptom_vars = {} # Add this
        
        self.configure_styles()
        self.create_widgets()
        self.load_common_symptoms()
        
    def configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('TFrame', background='#f5f5f5')
        style.configure('Header.TLabel', font=('Arial', 10, 'bold'), foreground='#333333', background='#f5f5f5')
        style.configure('TLabelframe', font=('Arial', 10, 'bold'), foreground='#333333', background='#f5f5f5')
        style.configure('TLabelframe.Label', font=('Arial', 10, 'bold'), foreground='#333333')
        style.configure('TButton', font=('Arial', 10), padding=6)
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'), foreground='white', background='#4a90e2')
        style.configure('Result.TLabel', font=('Arial', 11, 'bold'), foreground='#0066cc')
        style.configure('Confidence.TLabel', font=('Arial', 10), foreground='#333333')
        style.configure('TCheckbutton', font=('Arial', 9), foreground='#333333')
        style.configure('TCombobox', padding=5)
        
        style.map('Accent.TButton', background=[('active', '#3a7bc8'), ('pressed', '#2a6bb0')])

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding=15)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Body systems mapping
        self.body_parts = {
            "Head/Neck": ["headache", "sore throat", "runny nose", "confusion", "blurred vision"],
            "Chest": ["chest pain", "shortness of breath", "cough", "wheezing"],
            "Abdomen": ["abdominal pain", "nausea", "vomiting", "diarrhea"],
            "Extremities": ["muscle aches", "joint pain", "swelling", "numbness"],
            "Skin": ["rash", "itching", "redness", "blisters"],
            "General": ["fever", "fatigue", "unexplained weight loss", "sweating"]
        }
        
        self.special_categories = {
            "Urinary/Reproductive": ["painful urination", "frequent urination", "blood in urine"],
            "Neurological": ["seizures", "tremors", "stiffness"],
            "Respiratory": ["sputum production", "congestion", "sneezing"],
            "Systemic": ["high blood pressure", "high cholesterol", "heat intolerance"]
        }

        # Body part selection
        ttk.Label(self.main_frame, text="1. Select Body System:", style='Header.TLabel').grid(
            row=0, column=0, sticky="w", pady=(0, 10))
        
        self.body_part_var = tk.StringVar()
        self.body_part_dropdown = ttk.Combobox(
            self.main_frame, 
            textvariable=self.body_part_var, 
            values=sorted(list(self.body_parts.keys()) + list(self.special_categories.keys())),
            state="readonly",
            style='TCombobox'
        )
        self.body_part_dropdown.grid(row=0, column=1, sticky="ew", padx=10, pady=(0, 10))
        self.body_part_dropdown.bind("<<ComboboxSelected>>", self.load_symptoms)

        # Symptom selection frame
        self.symptoms_frame = ttk.LabelFrame(
            self.main_frame, 
            text="2. Select Specific Symptoms:", 
            style='TLabelframe'
        )
        self.symptoms_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=5)
        
        self.canvas = tk.Canvas(self.symptoms_frame, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.symptoms_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Common symptoms
        ttk.Label(self.main_frame, text="3. Common Systemic Symptoms:", style='Header.TLabel').grid(row=2, column=0, sticky="w", pady=(10, 5))
        self.common_frame = ttk.Frame(self.main_frame)
        self.common_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        # Selected symptoms display
        ttk.Label(self.main_frame, text="Currently Selected Symptoms:", style='Header.TLabel').grid(row=4, column=0, sticky="w", pady=(10, 5))
        self.selected_symptoms_text = tk.Text(
            self.main_frame, 
            height=5, 
            width=85, 
            wrap=tk.WORD,
            font=('Arial', 10),
            highlightbackground="#e0e0e0",
            highlightthickness=1
        )
        self.selected_symptoms_text.grid(row=5, column=0, columnspan=2, pady=(0, 10))

        # Control buttons
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=15)
        ttk.Button(button_frame, text="Predict Disease", command=self.predict, style='TButton').pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Clear All", command=self.clear_selections, style='TButton').pack(side=tk.LEFT, padx=10)

        # Results display
        results_frame = ttk.LabelFrame(
            self.main_frame,
            text="Diagnosis Results",
            style='TLabelframe'  
        )
        results_frame.grid(row=7, column=0, columnspan=2, sticky="nsew", pady=5)
        
        self.result_var = tk.StringVar(value="No prediction yet")
        ttk.Label(results_frame, textvariable=self.result_var, wraplength=850, style='Result.TLabel').pack(anchor="w", padx=10, pady=10)
        self.confidence_var = tk.StringVar()
        ttk.Label(results_frame, textvariable=self.confidence_var, style='Confidence.TLabel').pack(anchor="w", padx=10, pady=(0, 10))

        # Configure grid weights
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.rowconfigure(7, weight=1)
    
    def configure_styles(self):
        style = ttk.Style()
    
        # Main frame
        style.configure('TFrame', background='#f5f5f5')
    
        # Labels
        style.configure('Header.TLabel', 
                   font=('Arial', 10, 'bold'), 
                   foreground='#333333',
                   background='#f5f5f5')
    
        # LabelFrames (corrected style name)
        style.configure('TLabelframe', 
                    font=('Arial', 10, 'bold'), 
                    foreground='#333333',
                    background='#f5f5f5')
        style.configure('TLabelframe.Label', 
                   font=('Arial', 10, 'bold'), 
                   foreground='#333333')
    
        # Buttons
        style.configure('TButton', 
                   font=('Arial', 10), 
                   padding=6,
                   relief='flat')
        style.configure('Accent.TButton', 
                   font=('Arial', 10, 'bold'), 
                   foreground='white',
                   background='#4a90e2',
                   padding=6,
                   relief='flat')
    
        # Results
        style.configure('Result.TLabel', 
                   font=('Arial', 11, 'bold'), 
                   foreground='#0066cc',
                   background='#f5f5f5')
        style.configure('Confidence.TLabel', 
                   font=('Arial', 10), 
                   foreground='#333333',
                   background='#f5f5f5')
    
        # Checkbuttons
        style.configure('TCheckbutton', 
                   font=('Arial', 9), 
                   foreground='#333333',
                   background='#f5f5f5')
    
        # Combobox
        style.configure('TCombobox', padding=5)
    
        # Button states
        style.map('Accent.TButton',
             background=[('active', '#3a7bc8'), 
                         ('pressed', '#2a6bb0')])
    
    def load_symptoms(self, event):
        # Clear previous checkboxes
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Get appropriate symptom list
        body_part = self.body_part_var.get()
        symptoms = self.body_parts.get(body_part, self.special_categories.get(body_part, []))
        
        # Create checkboxes in 3-column grid
        self.symptom_vars = {}
        for i, symptom in enumerate(sorted(symptoms)):
            self.symptom_vars[symptom] = tk.BooleanVar()
            cb = ttk.Checkbutton(
                self.scrollable_frame, 
                text=symptom.title(), 
                variable=self.symptom_vars[symptom],
                command=self.update_selected_symptoms,
                style='TCheckbutton'
            )
            cb.grid(row=i//3, column=i%3, sticky="w", padx=8, pady=3)
    
    def load_common_symptoms(self):
        common_symptoms = sorted([
            "fever", "fatigue", "headache", "nausea", "vomiting", "diarrhea",
            "dizziness", "sweating", "chills", "unexplained weight loss"
        ])
        
        for i, symptom in enumerate(common_symptoms):
            self.common_symptom_vars[symptom] = tk.BooleanVar()
            cb = ttk.Checkbutton(
                self.common_frame, 
                text=symptom.title(), 
                variable=self.common_symptom_vars[symptom],
                command=self.update_selected_symptoms,
                style='TCheckbutton'
            )
            cb.grid(row=0, column=i, padx=8, pady=3)
    
    def update_selected_symptoms(self):
        self.selected_symptoms_text.delete(1.0, tk.END)
        selected = []
        
        # Get body-specific symptoms
        for symptom, var in self.symptom_vars.items():
            if var.get():
                selected.append(symptom)
        
        # Get common symptoms
        for symptom, var in self.common_symptom_vars.items():
            if var.get():
                selected.append(symptom)
        
        # Display in alphabetical order
        self.selected_symptoms_text.insert(
            tk.END, 
            ", ".join(sorted(selected)).title() if selected else "No symptoms selected"
        )
    
    def clear_selections(self):
        # Clear all checkboxes
        for var in self.symptom_vars.values():
            var.set(False)
        for var in self.common_symptom_vars.values():
            var.set(False)
        self.selected_symptoms_text.delete(1.0, tk.END)
        self.result_var.set("No prediction yet")
        self.confidence_var.set("")
        self.body_part_var.set("")
    
    def predict(self):
        # Get selected symptoms
        selected_symptoms = []
        for symptom, var in {**self.symptom_vars, **self.common_symptom_vars}.items():
            if var.get():
                selected_symptoms.append(symptom.lower())
        
        if not selected_symptoms:
            messagebox.showwarning("Warning", "Please select at least one symptom!")
            return
        
        # Validate and map symptoms
        valid_symptoms = []
        for symptom in selected_symptoms:
            # Check both original and synonym-mapped versions
            mapped_symptom = symptom_synonyms.get(symptom, symptom)
            if mapped_symptom in mlb.classes_:
                valid_symptoms.append(mapped_symptom)
            else:
                print(f"Symptom not in training data: {symptom}")
        
        if not valid_symptoms:
            messagebox.showerror("Error", "None of the selected symptoms are recognized!")
            return
        
        # Convert to model input format
        input_vector = np.zeros(len(mlb.classes_))
        for symptom in valid_symptoms:
            idx = list(mlb.classes_).index(symptom)
            input_vector[idx] = 1
        
        # Make prediction
        try:
            proba = model.predict_proba([input_vector])[0]
            top_3 = np.argsort(proba)[-3:][::-1]  # Get indices of top 3 predictions
            
            # Build result string
            result_text = "Top Predictions:\n"
            for i, idx in enumerate(top_3):
                disease = model.classes_[idx]
                confidence = proba[idx]
                result_text += f"{i+1}. {disease} ({confidence*100:.1f}%)\n"
            
            self.result_var.set(result_text)
            self.confidence_var.set("")
            
            # Show detailed info for top prediction
            top_disease = model.classes_[top_3[0]]
            disease_info = data[data['disease'] == top_disease].iloc[0]
            
            messagebox.showinfo(
                "Detailed Diagnosis",
                f"Most Likely Condition: {top_disease}\n\n"
                f"Typical Symptoms:\n{', '.join(disease_info['symptoms']).title()}\n\n"
                f"Recommended Treatments:\n{disease_info['cures']}\n\n"
                f"Consult: {disease_info['doctor']}\n"
                f"Risk Level: {disease_info['risk level']}"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DiseasePredictorApp(root)
    root.mainloop()
