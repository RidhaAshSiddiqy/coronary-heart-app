# coronary_heart_app.py
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

class CoronaryHeartPredictor:
    def __init__(self):
        self.rf_model = None
        self.lr_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.df = None
        self.df_cleaned = None
        self.models_trained = False
        self.smote_applied = False
        self.class_distribution_before = None
        self.class_distribution_after = None
    
    def load_and_preprocess_data(self):
        """Load dan preprocess data dari path yang sudah ditentukan"""
        try:
            # Load data dengan path hardcoded
            self.df = pd.read_csv('WorldCoronaryHeart.csv')
            
            # Preprocessing target variable
            self.df['Target'] = self.df['Diagnostic_Results'].apply(
                lambda x: 1 if 'Increasing to Coronary Heart' in str(x) else 0
            )
            
            # Auto-cleaning: Bersihkan data secara otomatis
            self.df_cleaned = self._clean_dataset(self.df)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
            return False
    
    def _clean_dataset(self, df):
        """Bersihkan dataset dari missing values dan masalah data - SEMUA MISSING VALUES DIUBAH MENJADI 0"""
        df_clean = df.copy()
        
        # 1. Handle BMI format (convert from string to float)
        if 'BMI' in df_clean.columns:
            df_clean['BMI'] = df_clean['BMI'].astype(str).str.replace(',', '.').astype(float)
        
        # 2. Handle missing values untuk SEMUA KOLOM - DIUBAH MENJADI 0
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                # Untuk numerical columns, isi dengan 0
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col].fillna(0, inplace=True)
                # Untuk categorical columns, isi dengan 'Unknown' atau '0'
                else:
                    df_clean[col].fillna('Unknown', inplace=True)
        
        return df_clean
    
    def get_data_for_visualization(self):
        """Return dataframe untuk visualisasi - SUDAH DIBERSIHKAN"""
        if self.df_cleaned is None:
            success = self.load_and_preprocess_data()
            if not success:
                return None
        return self.df_cleaned
    
    def get_original_data_with_missing_info(self):
        """Return data original dengan info missing values untuk analysis"""
        if self.df is None:
            success = self.load_and_preprocess_data()
            if not success:
                return None, None, None, None
        
        # Hitung missing values sebelum cleaning
        missing_before = self.df.isnull().sum()
        missing_before_pct = (missing_before / len(self.df)) * 100
        
        # Hitung missing values setelah cleaning
        missing_after = self.df_cleaned.isnull().sum()
        
        return self.df, missing_before, missing_before_pct, missing_after
    
    def _preprocess_features(self, X):
        """Preprocess features untuk training"""
        # Encode categorical variables
        categorical_columns = ['Gender', 'Physical_Activity', 'Diet_Habits', 
                             'Smoking_History', 'Diabetes_History', 
                             'Hypertension_History', 'Family_History']
        
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        return X
    
    def auto_train_models(self):
        """Automatically train models with SMOTE for handling class imbalance"""
        if self.models_trained:
            return True
            
        if self.df_cleaned is None:
            success = self.load_and_preprocess_data()
            if not success:
                return False
        
        # Select relevant features
        feature_columns = [
            'Age', 'Gender', 'BloodGlucose', 'TotCholesterol', 'LDL', 'HDL', 
            'Triglyceride', 'Sistole', 'Diastole', 'Height_cm', 'Weight_kg',
            'BMI', 'Physical_Activity', 'Diet_Habits', 'Smoking_History',
            'Diabetes_History', 'Hypertension_History', 'Family_History'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in self.df_cleaned.columns]
        self.feature_names = available_features
        
        X = self.df_cleaned[available_features].copy()
        y = self.df_cleaned['Target']
        
        # Store class distribution before SMOTE
        self.class_distribution_before = y.value_counts()
        
        # Preprocessing
        X = self._preprocess_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE to training data only
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        self.smote_applied = True
        self.class_distribution_after = y_train_resampled.value_counts()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_resampled)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Additional balancing
        )
        self.rf_model.fit(X_train_scaled, y_train_resampled)
        
        # Train Logistic Regression
        self.lr_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Additional balancing
        )
        self.lr_model.fit(X_train_scaled, y_train_resampled)
        
        self.models_trained = True
        return True
    
    def save_models(self, filepath):
        """Save models dan preprocessing objects"""
        joblib.dump({
            'rf_model': self.rf_model,
            'lr_model': self.lr_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, filepath)
    
    def load_models(self, filepath):
        """Load models dan preprocessing objects"""
        try:
            loaded = joblib.load(filepath)
            self.rf_model = loaded['rf_model']
            self.lr_model = loaded['lr_model']
            self.scaler = loaded['scaler']
            self.label_encoders = loaded['label_encoders']
            self.feature_names = loaded['feature_names']
            self.models_trained = True
            return True
        except:
            return False
    
    def predict(self, input_data, model_type='rf'):
        """Make prediction dengan model pilihan"""
        if not self.models_trained:
            # Auto train models jika belum di-train
            success = self.auto_train_models()
            if not success:
                return None, 0
        
        # Preprocess input
        processed_input = self._preprocess_user_input(input_data)
        
        if processed_input is None:
            return None, 0
            
        # Scale features
        scaled_input = self.scaler.transform(processed_input)
        
        # Make prediction
        if model_type == 'rf':
            prediction = self.rf_model.predict(scaled_input)[0]
            probability = self.rf_model.predict_proba(scaled_input)[0]
        else:
            prediction = self.lr_model.predict(scaled_input)[0]
            probability = self.lr_model.predict_proba(scaled_input)[0]
        
        return prediction, max(probability)
    
    def _preprocess_user_input(self, input_data):
        """Preprocess user input untuk prediction"""
        try:
            # Create feature array in correct order
            feature_array = []
            for feature in self.feature_names:
                if feature in input_data:
                    value = input_data[feature]
                    
                    # Handle categorical encoding
                    if feature in self.label_encoders:
                        le = self.label_encoders[feature]
                        # Convert user input to encoded value
                        if value in le.classes_:
                            encoded_value = le.transform([value])[0]
                        else:
                            # Default to first class if not found
                            encoded_value = 0
                        feature_array.append(encoded_value)
                    else:
                        # Numerical features
                        feature_array.append(float(value))
                else:
                    # If feature missing, use default value
                    feature_array.append(0)
            
            return np.array(feature_array).reshape(1, -1)
            
        except Exception as e:
            st.error(f"Error processing input: {e}")
            return None

def main():
    """Main Streamlit Application"""
    
    # Set page configuration
    st.set_page_config(
        page_title="Coronary Heart Disease Predictor",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #e63946;
            text-align: center;
            margin-bottom: 2rem;
        }
        .prediction-card {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
            border-left: 5px solid #e63946;
            margin: 1rem 0;
        }
        .high-risk {
            background-color: #ffe6e6;
            border-left: 5px solid #e63946;
        }
        .low-risk {
            background-color: #e6f7e6;
            border-left: 5px solid #28a745;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #4e73df;
            margin: 0.5rem 0;
        }
        .footer {
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            color: #6c757d;
        }
        .cleaning-report {
            background-color: #e8f4fd;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #17a2b8;
            margin: 1rem 0;
        }
        .zero-missing {
            background-color: #d4edda;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #28a745;
            margin: 1rem 0;
        }
        .smote-info {
            background-color: #fff3cd;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #ffc107;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">❤️ Coronary Heart Disease Predictor</h1>', 
               unsafe_allow_html=True)

    # Initialize predictor
    predictor = CoronaryHeartPredictor()

    # Sidebar - HANYA NAVIGATION
    st.sidebar.title("Navigation")
    
    # Main content navigation
    app_mode = st.sidebar.selectbox(
        "Choose Section",
        ["Data Analysis", "Prediction", "Model Accuracy", "Data Visualization"]
    )

    if app_mode == "Data Analysis":
        show_data_analysis(predictor)
    elif app_mode == "Prediction":
        show_prediction_interface(predictor)
    elif app_mode == "Model Accuracy":
        show_model_accuracy(predictor)
    else:
        show_data_visualization(predictor)

    # Footer dengan copyright yang benar
    st.markdown("---")
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("### Copyright © 2025 - Coronary Heart Disease Prediction System")
    st.markdown("**Medical AI Assistant for Early Detection of Heart Conditions**")
    st.markdown('</div>', unsafe_allow_html=True)

def show_data_analysis(predictor):
    """Show data analysis interface - DENGAN INFO CLEANING"""
    st.header("📊 Data Analysis")
    
    # Dapatkan data original dan info missing values
    original_df, missing_before, missing_before_pct, missing_after = predictor.get_original_data_with_missing_info()
    cleaned_df = predictor.get_data_for_visualization()
    
    if original_df is None or cleaned_df is None:
        st.error("❌ Cannot load dataset. Please ensure the dataset file exists in the correct location.")
        return
    
    # Basic statistics - PAKAI DATA BERSIH
    st.subheader("📈 Dataset Overview (After Cleaning)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(cleaned_df))
    with col2:
        st.metric("Number of Features", len(cleaned_df.columns))
    with col3:
        target_counts = cleaned_df['Target'].value_counts()
        st.metric("High Risk Cases", target_counts[1])
    with col4:
        st.metric("Low Risk Cases", target_counts[0])
    
    # Class Distribution Analysis
    st.subheader("🎯 Class Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Class Distribution:**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        colors = ['#28a745', '#e63946']
        target_counts.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title('Original Class Distribution')
        ax1.set_xlabel('Risk Category')
        ax1.set_ylabel('Count')
        ax1.set_xticklabels(['Low Risk', 'High Risk'], rotation=0)
        
        # Add value labels on bars
        for i, v in enumerate(target_counts):
            ax1.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig1)
    
    with col2:
        st.write("**Class Distribution Percentage:**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        percentages = (target_counts / len(cleaned_df)) * 100
        wedges, texts, autotexts = ax2.pie(percentages, labels=['Low Risk', 'High Risk'], 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Class Distribution (%)')
        
        # Improve label visibility
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        st.pyplot(fig2)
    
    # Missing Values Analysis - BEFORE vs AFTER CLEANING
    st.subheader("🔧 Data Cleaning Report")
    
    # Hitung total missing values sebelum cleaning
    total_missing_before = missing_before.sum()
    total_missing_after = missing_after.sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Missing Values Before Cleaning", total_missing_before)
    with col2:
        delta_value = total_missing_before - total_missing_after
        delta_label = f"-{delta_value}" if delta_value > 0 else None
        st.metric("Missing Values After Cleaning", total_missing_after, 
                 delta=delta_label)
    
    # Tampilkan detail missing values per kolom (sebelum cleaning)
    st.subheader("📋 Missing Values Analysis (Before Cleaning)")
    
    if total_missing_before > 0:
        missing_df = pd.DataFrame({
            'Column': missing_before.index,
            'Missing Values': missing_before.values,
            'Percentage': missing_before_pct.values
        })
        # Hanya tampilkan kolom dengan missing values
        missing_df_display = missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False)
        
        if not missing_df_display.empty:
            st.dataframe(missing_df_display)
            
            # Info cleaning yang dilakukan - SEMUA DIUBAH MENJADI 0
            st.markdown('<div class="zero-missing">', unsafe_allow_html=True)
            st.markdown("### ✅ **Auto-Cleaning Applied**")
            st.markdown("**SEMUA MISSING VALUES TELAH DIUBAH MENJADI 0:**")
            st.markdown("- **Numerical features:** Diisi dengan 0")
            st.markdown("- **Categorical features:** Diisi dengan 'Unknown'") 
            st.markdown("- **BMI format:** Converted from string to float")
            st.markdown("### 🎯 **Semua missing values sekarang bernilai 0**")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.success("🎉 **Dataset Status:** No missing values found in the displayed columns!")
    else:
        st.success("🎉 **Dataset Status:** No missing values found in the entire dataset!")
    
    # Dataset preview - PAKAI DATA BERSIH
    st.subheader("👀 Data Preview (After Cleaning - All Missing Values = 0)")
    st.dataframe(cleaned_df.head(10))
    
    # Statistical summary - PAKAI DATA BERSIH
    st.subheader("📊 Statistical Summary (After Cleaning - All Missing Values = 0)")
    st.dataframe(cleaned_df.describe())
    
    # Verifikasi bahwa tidak ada missing values setelah cleaning
    st.subheader("✅ Data Quality Verification")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_missing_after = cleaned_df.isnull().sum().sum()
        st.metric("Total Missing Values After Cleaning", total_missing_after)
    
    with col2:
        total_zeros = (cleaned_df == 0).sum().sum()
        st.metric("Total Zero Values in Dataset", total_zeros)
    
    with col3:
        dataset_size = cleaned_df.size
        zero_percentage = (total_zeros / dataset_size) * 100
        st.metric("Percentage of Zero Values", f"{zero_percentage:.2f}%")

def show_prediction_interface(predictor):
    """Show prediction interface"""
    st.header("🔍 Risk Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Personal Information")
        age = st.number_input("Age", min_value=20, max_value=100, value=45, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=140, max_value=200, value=170, step=1)
        weight = st.number_input("Weight (kg)", min_value=40, max_value=150, value=70, step=1)
        
        # Calculate BMI
        if height > 0:
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.2f}")
        else:
            bmi = 22.0
    
    with col2:
        st.subheader("💉 Blood Measurements")
        blood_glucose = st.number_input("Blood Glucose (mg/dL)", min_value=70, max_value=200, value=100, step=1)
        tot_cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=300, value=200, step=1)
        ldl = st.number_input("LDL (mg/dL)", min_value=50, max_value=250, value=120, step=1)
        hdl = st.number_input("HDL (mg/dL)", min_value=20, max_value=100, value=50, step=1)
        triglyceride = st.number_input("Triglyceride (mg/dL)", min_value=100, max_value=600, value=150, step=1)
    
    with col3:
        st.subheader("🏥 Medical History")
        sistole = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=200, value=120, step=1)
        diastole = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=120, value=80, step=1)
        
        physical_activity = st.selectbox(
            "Physical Activity", 
            ["Activity", "No Activity"]
        )
        diet_habits = st.selectbox(
            "Diet Habits", 
            ["Healthy Diet", "Unhealthy Diet"]
        )
        smoking_history = st.selectbox(
            "Smoking History", 
            ["No Smoking", "Smoking"]
        )
        diabetes_history = st.selectbox(
            "Diabetes History", 
            ["Normal", "Diabetes"]
        )
        hypertension_history = st.selectbox(
            "Hypertension History", 
            ["Normal", "Hypertension"]
        )
        family_history = st.selectbox(
            "Family History", 
            ["Normal", "Coronary"]
        )
    
    # Model selection for prediction
    st.subheader("🤖 Model Selection")
    model_choice = st.radio(
        "Choose prediction model:",
        ["Random Forest", "Logistic Regression"],
        horizontal=True
    )
    
    # SMOTE Information
    st.markdown('<div class="smote-info">', unsafe_allow_html=True)
    st.markdown("### 🔄 **SMOTE Applied for Class Imbalance**")
    st.markdown("**To handle class imbalance, we use SMOTE (Synthetic Minority Over-sampling Technique):**")
    st.markdown("- Generates synthetic samples for minority class")
    st.markdown("- Improves model performance on imbalanced data")
    st.markdown("- Applied only to training data (not test data)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prepare input data
    input_data = {
        'Age': age,
        'Gender': gender,
        'BloodGlucose': blood_glucose,
        'TotCholesterol': tot_cholesterol,
        'LDL': ldl,
        'HDL': hdl,
        'Triglyceride': triglyceride,
        'Sistole': sistole,
        'Diastole': diastole,
        'Height_cm': height,
        'Weight_kg': weight,
        'BMI': bmi,
        'Physical_Activity': physical_activity,
        'Diet_Habits': diet_habits,
        'Smoking_History': smoking_history,
        'Diabetes_History': diabetes_history,
        'Hypertension_History': hypertension_history,
        'Family_History': family_history
    }
    
    # Prediction button
    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        with st.spinner("Processing prediction..."):
            model_type = 'rf' if model_choice == "Random Forest" else 'lr'
            prediction, probability = predictor.predict(input_data, model_type)
            
            if prediction is None:
                st.error("❌ Error making prediction. Please check if dataset is available.")
                return
            
            # Display results
            if prediction == 1:
                risk_level = "HIGH RISK"
                risk_message = "Increasing to Coronary Heart"
                card_class = "high-risk"
                emoji = "⚠️"
            else:
                risk_level = "LOW RISK"
                risk_message = "Normal"
                card_class = "low-risk"
                emoji = "✅"
            
            st.markdown(f"""
            <div class="prediction-card {card_class}">
                <h2>{emoji} Prediction Result: {risk_level}</h2>
                <h3>{risk_message}</h3>
                <p><strong>Model Used:</strong> {model_choice}</p>
                <p><strong>Confidence:</strong> {probability:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk factors analysis
            st.subheader("📊 Risk Factors Analysis")
            display_risk_analysis(input_data)

def show_model_accuracy(predictor):
    """Show model accuracy comparison with SMOTE information"""
    st.header("📈 Model Accuracy Comparison")
    
    # Auto train models untuk mendapatkan accuracy
    with st.spinner("Training models for accuracy comparison..."):
        success = predictor.auto_train_models()
    
    if not success:
        st.error("❌ Cannot load dataset. Please ensure the dataset file exists in the correct location.")
        return
    
    # Display SMOTE information
    if predictor.smote_applied and predictor.class_distribution_before is not None:
        st.markdown('<div class="smote-info">', unsafe_allow_html=True)
        st.markdown("### 🔄 **SMOTE Class Balancing Applied**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Before SMOTE:**")
            st.write(f"Low Risk (0): {predictor.class_distribution_before[0]}")
            st.write(f"High Risk (1): {predictor.class_distribution_before[1]}")
            imbalance_ratio = predictor.class_distribution_before[1] / predictor.class_distribution_before[0]
            st.write(f"Imbalance Ratio: {imbalance_ratio:.2f}")
        
        with col2:
            st.write("**After SMOTE (Training Data):**")
            st.write(f"Low Risk (0): {predictor.class_distribution_after[0]}")
            st.write(f"High Risk (1): {predictor.class_distribution_after[1]}")
            st.write("Balanced: ✅")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Select relevant features
    feature_columns = [
        'Age', 'Gender', 'BloodGlucose', 'TotCholesterol', 'LDL', 'HDL', 
        'Triglyceride', 'Sistole', 'Diastole', 'Height_cm', 'Weight_kg',
        'BMI', 'Physical_Activity', 'Diet_Habits', 'Smoking_History',
        'Diabetes_History', 'Hypertension_History', 'Family_History'
    ]
    available_features = [col for col in feature_columns if col in predictor.df_cleaned.columns]
    X = predictor.df_cleaned[available_features].copy()
    y = predictor.df_cleaned['Target']
    
    # Preprocessing
    X = predictor._preprocess_features(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features (using the same scaler that was fit on SMOTE data)
    X_test_scaled = predictor.scaler.transform(X_test)
    
    # Calculate predictions
    rf_pred = predictor.rf_model.predict(X_test_scaled)
    lr_pred = predictor.lr_model.predict(X_test_scaled)
    
    # Calculate accuracies
    rf_accuracy = accuracy_score(y_test, rf_pred)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Random Forest Accuracy", f"{rf_accuracy:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # RF Classification Report
        st.subheader("🌲 Random Forest - Classification Report")
        st.text(classification_report(y_test, rf_pred))
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Logistic Regression Accuracy", f"{lr_accuracy:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # LR Classification Report
        st.subheader("📈 Logistic Regression - Classification Report")
        st.text(classification_report(y_test, lr_pred))
    
    # Confusion Matrices
    st.subheader("🔄 Confusion Matrices")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🌲 Random Forest**")
        fig, ax = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(y_test, rf_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Random Forest Confusion Matrix')
        st.pyplot(fig)
    
    with col2:
        st.write("**📈 Logistic Regression**")
        fig, ax = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(y_test, lr_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Logistic Regression Confusion Matrix')
        st.pyplot(fig)
    
    # Feature Importance (Random Forest only)
    st.subheader("🔍 Random Forest Feature Importance")
    if predictor.rf_model is not None:
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': predictor.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax)
        ax.set_title('Top 10 Most Important Features (Random Forest)')
        ax.set_xlabel('Feature Importance')
        st.pyplot(fig)

def show_data_visualization(predictor):
    """Show data visualization - PAKAI DATA BERSIH"""
    st.header("📊 Data Visualization")
    
    df = predictor.get_data_for_visualization()
    if df is None:
        st.error("❌ Cannot load dataset. Please ensure the dataset file exists in the correct location.")
        return
    
    # Visualization options
    viz_option = st.selectbox(
        "Choose Visualization:",
        ["Target Distribution", "Age Distribution", "Blood Metrics Distribution", 
         "BMI vs Blood Pressure", "Correlation Heatmap", "Gender Distribution",
         "Blood Pressure Categories", "Risk by Age Group", "Feature Distributions"]
    )
    
    if viz_option == "Target Distribution":
        st.subheader("🎯 Target Variable Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        target_counts = df['Target'].value_counts()
        colors = ['#28a745', '#e63946']
        ax.pie(target_counts.values, labels=['Low Risk', 'High Risk'], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Distribution of Coronary Heart Disease Risk')
        st.pyplot(fig)
    
    elif viz_option == "Age Distribution":
        st.subheader("📅 Age Distribution by Risk Category")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='Age', hue='Target', 
                    bins=20, kde=True, ax=ax)
        ax.set_title('Age Distribution by Risk Category')
        ax.legend(['Low Risk', 'High Risk'])
        st.pyplot(fig)
    
    elif viz_option == "Blood Metrics Distribution":
        st.subheader("💊 Blood Metrics Distribution")
        metrics = ['BloodGlucose', 'TotCholesterol', 'LDL', 'HDL', 'Triglyceride']
        selected_metric = st.selectbox("Select Metric:", metrics)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='Target', y=selected_metric, ax=ax)
        ax.set_title(f'{selected_metric} Distribution by Risk Category')
        ax.set_xticklabels(['Low Risk', 'High Risk'])
        
        # Tambahkan label nilai pada boxplot
        for i, box in enumerate(ax.artists):
            # Dapatkan statistik untuk setiap box
            stats = df.groupby('Target')[selected_metric].describe()
            median_val = stats.loc[i, '50%']
            q1_val = stats.loc[i, '25%']
            q3_val = stats.loc[i, '75%']
            
            # Tambahkan teks untuk median
            ax.text(i, median_val, f'Median: {median_val:.1f}', 
                   ha='center', va='bottom', fontweight='bold', color='red')
            
            # Tambahkan teks untuk Q1 dan Q3
            ax.text(i, q1_val, f'Q1: {q1_val:.1f}', 
                   ha='center', va='top', fontsize=8, color='blue')
            ax.text(i, q3_val, f'Q3: {q3_val:.1f}', 
                   ha='center', va='bottom', fontsize=8, color='blue')
        
        st.pyplot(fig)
    
    elif viz_option == "BMI vs Blood Pressure":
        st.subheader("⚖️ BMI vs Blood Pressure")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df['BMI'], df['Sistole'], 
                           c=df['Target'], cmap='coolwarm', alpha=0.6)
        ax.set_xlabel('BMI')
        ax.set_ylabel('Systolic Blood Pressure')
        ax.set_title('BMI vs Blood Pressure (Color: Risk Level)')
        plt.colorbar(scatter, label='Risk Level (0=Low, 1=High)')
        st.pyplot(fig)
    
    elif viz_option == "Correlation Heatmap":
        st.subheader("🔗 Feature Correlation Heatmap")
        # Select numerical columns for correlation
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        # Remove Target dari correlation
        numerical_cols = [col for col in numerical_cols if col != 'Target']
        corr_matrix = df[numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
        ax.set_title('Feature Correlation Heatmap')
        st.pyplot(fig)
    
    elif viz_option == "Gender Distribution":
        st.subheader("🚻 Gender Distribution by Risk Category")
        fig, ax = plt.subplots(figsize=(10, 6))
        gender_risk = pd.crosstab(df['Gender'], df['Target'])
        gender_risk_pct = gender_risk.div(gender_risk.sum(axis=1), axis=0) * 100
        
        # Plot bar chart dengan persentase
        bars = gender_risk_pct.plot(kind='bar', ax=ax, color=['#28a745', '#e63946'])
        ax.set_title('Gender Distribution by Risk Category (%)')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Percentage')
        ax.legend(['Low Risk', 'High Risk'])
        ax.tick_params(axis='x', rotation=0)
        
        # Tambahkan label persentase di atas setiap bar
        for i, (idx, row) in enumerate(gender_risk_pct.iterrows()):
            for j, value in enumerate(row):
                ax.text(i, value + 1, f'{value:.1f}%', 
                       ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
        
        # Tampilkan juga tabel frekuensi
        st.subheader("📋 Gender Distribution - Frequency Table")
        st.dataframe(gender_risk)
    
    elif viz_option == "Blood Pressure Categories":
        st.subheader("🩺 Blood Pressure Categories Distribution")
        # Create BP categories
        def bp_category(sistole, diastole):
            if sistole < 120 and diastole < 80:
                return 'Normal'
            elif sistole < 130 and diastole < 80:
                return 'Elevated'
            elif sistole < 140 or diastole < 90:
                return 'High Stage 1'
            else:
                return 'High Stage 2'
        
        df_bp = df.copy()
        df_bp['BP_Category'] = df_bp.apply(lambda row: bp_category(row['Sistole'], row['Diastole']), axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bp_counts = df_bp['BP_Category'].value_counts()
        colors = ['#28a745', '#ffc107', '#fd7e14', '#e63946']
        wedges, texts, autotexts = ax.pie(bp_counts.values, labels=bp_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax.set_title('Blood Pressure Categories Distribution')
        
        # Improve label visibility
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        st.pyplot(fig)
        
        # Tampilkan juga distribusi berdasarkan risk category
        st.subheader("🩺 Blood Pressure Categories by Risk")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        bp_risk = pd.crosstab(df_bp['BP_Category'], df_bp['Target'])
        bp_risk_pct = bp_risk.div(bp_risk.sum(axis=1), axis=0) * 100
        
        bars = bp_risk_pct.plot(kind='bar', ax=ax2, color=['#28a745', '#e63946'])
        ax2.set_title('Blood Pressure Categories by Risk Level (%)')
        ax2.set_xlabel('Blood Pressure Category')
        ax2.set_ylabel('Percentage')
        ax2.legend(['Low Risk', 'High Risk'])
        ax2.tick_params(axis='x', rotation=45)
        
        # Tambahkan label persentase di atas setiap bar
        for i, (idx, row) in enumerate(bp_risk_pct.iterrows()):
            total_height = 0
            for j, value in enumerate(row):
                if value > 5:  # Hanya tambahkan label jika persentase cukup besar
                    ax2.text(i, total_height + value/2, f'{value:.1f}%', 
                           ha='center', va='center', fontweight='bold', color='white')
                total_height += value
        
        st.pyplot(fig2)
    
    elif viz_option == "Risk by Age Group":
        st.subheader("📊 Risk Distribution by Age Group")
        # Create age groups
        df_age = df.copy()
        df_age['Age_Group'] = pd.cut(df_age['Age'], 
                                   bins=[20, 30, 40, 50, 60, 70, 100], 
                                   labels=['20-29', '30-39', '40-49', '50-59', '60-69', '70+'])
        
        age_risk = pd.crosstab(df_age['Age_Group'], df_age['Target'])
        age_risk_pct = age_risk.div(age_risk.sum(axis=1), axis=0) * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = age_risk_pct.plot(kind='bar', ax=ax, color=['#28a745', '#e63946'])
        ax.set_title('Risk Distribution by Age Group (%)')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Percentage')
        ax.legend(['Low Risk', 'High Risk'])
        ax.tick_params(axis='x', rotation=45)
        
        # Tambahkan label persentase di atas setiap bar
        for i, (idx, row) in enumerate(age_risk_pct.iterrows()):
            total_height = 0
            for j, value in enumerate(row):
                if value > 5:  # Hanya tambahkan label jika persentase cukup besar
                    ax.text(i, total_height + value/2, f'{value:.1f}%', 
                           ha='center', va='center', fontweight='bold', color='white',
                           fontsize=8)
                total_height += value
        
        st.pyplot(fig)
        
        # Tampilkan juga tabel frekuensi
        st.subheader("📋 Age Group Distribution - Frequency Table")
        st.dataframe(age_risk)
    
    elif viz_option == "Feature Distributions":
        st.subheader("📊 Feature Distributions by Risk Category")
        
        # Select feature to visualize
        numerical_features = ['Age', 'BloodGlucose', 'TotCholesterol', 'LDL', 'HDL', 
                             'Triglyceride', 'Sistole', 'Diastole', 'BMI']
        selected_feature = st.selectbox("Select Feature:", numerical_features)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        sns.histplot(data=df, x=selected_feature, hue='Target', 
                    bins=20, kde=True, ax=ax1)
        ax1.set_title(f'{selected_feature} Distribution by Risk Category')
        ax1.legend(['Low Risk', 'High Risk'])
        
        # Box plot
        sns.boxplot(data=df, x='Target', y=selected_feature, ax=ax2)
        ax2.set_title(f'{selected_feature} by Risk Category')
        ax2.set_xticklabels(['Low Risk', 'High Risk'])
        
        st.pyplot(fig)

def display_risk_analysis(input_data):
    """Display risk factors analysis"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🔴 Key Risk Indicators:**")
        risk_factors = []
        
        if input_data['Age'] > 50:
            risk_factors.append("Age > 50 years")
        if input_data['TotCholesterol'] > 200:
            risk_factors.append("High Cholesterol (>200 mg/dL)")
        if input_data['LDL'] > 130:
            risk_factors.append("High LDL (>130 mg/dL)")
        if input_data['HDL'] < 40:
            risk_factors.append("Low HDL (<40 mg/dL)")
        if input_data['BloodGlucose'] > 126:
            risk_factors.append("High Blood Glucose (>126 mg/dL)")
        if input_data['Sistole'] > 140 or input_data['Diastole'] > 90:
            risk_factors.append("High Blood Pressure")
        if input_data['BMI'] > 25:
            risk_factors.append(f"Overweight (BMI: {input_data['BMI']:.1f})")
        if input_data['Smoking_History'] == 'Smoking':
            risk_factors.append("Smoking History")
        if input_data['Diabetes_History'] == 'Diabetes':
            risk_factors.append("Diabetes History")
        if input_data['Hypertension_History'] == 'Hypertension':
            risk_factors.append("Hypertension History")
        if input_data['Family_History'] == 'Coronary':
            risk_factors.append("Family History of Coronary Disease")
        if input_data['Physical_Activity'] == 'No Activity':
            risk_factors.append("Physical Inactivity")
        if input_data['Diet_Habits'] == 'Unhealthy Diet':
            risk_factors.append("Unhealthy Diet")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.write("• No significant risk factors detected")
    
    with col2:
        st.write("**💚 Health Recommendations:**")
        recommendations = [
            "🏃 Regular exercise (30-45 mins daily)",
            "🥗 Balanced diet with fruits and vegetables",
            "⚖️ Maintain healthy weight",
            "🩺 Regular health check-ups",
            "😊 Manage stress levels",
            "🚭 Avoid smoking",
            "🍷 Limit alcohol consumption",
            "🧂 Reduce salt intake",
            "💤 Ensure adequate sleep (7-8 hours)",
            "📊 Monitor blood pressure regularly"
        ]
        
        for rec in recommendations:
            st.write(f"• {rec}")

if __name__ == "__main__":
    main()



