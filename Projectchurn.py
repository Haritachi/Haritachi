# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Modern GUI library
import customtkinter as ctk
from tkinter import messagebox

# Set appearance and theme for customtkinter
ctk.set_appearance_mode("Dark")  # Light or Dark
ctk.set_default_color_theme("blue")  # blue, dark-blue, green

# Step 1: Load and process the dataset
df = pd.read_csv("Churn2.csv")
df = df.drop(columns=["customerID"])
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
df["TotalCharges"] = df["TotalCharges"].astype(float)
df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

# Encode categorical columns
object_columns = df.select_dtypes(include="object").columns
encoders = {}
for column in object_columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder

# Save encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Split data
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_smote, y_train_smote)

# Save model
model_data = {"model": rfc, "features_names": X.columns.tolist()}
with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

# Load model and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# -----------------------------------------------
# GUI Part
# -----------------------------------------------

# Function to predict churn
def predict_churn():
    try:
        input_data = {
            "gender": gender_option.get(),
            "SeniorCitizen": int(senior_citizen_entry.get()),
            "Partner": partner_option.get(),
            "Dependents": dependents_option.get(),
            "tenure": float(tenure_entry.get()),
            "PhoneService": phone_service_option.get(),
            "MultipleLines": multiple_lines_option.get(),
            "InternetService": internet_service_option.get(),
            "OnlineSecurity": online_security_option.get(),
            "OnlineBackup": online_backup_option.get(),
            "DeviceProtection": device_protection_option.get(),
            "TechSupport": tech_support_option.get(),
            "StreamingTV": streaming_tv_option.get(),
            "StreamingMovies": streaming_movies_option.get(),
            "Contract": contract_option.get(),
            "PaperlessBilling": paperless_billing_option.get(),
            "PaymentMethod": payment_method_option.get(),
            "MonthlyCharges": float(monthly_charges_entry.get()),
            "TotalCharges": float(total_charges_entry.get())
        }

        input_df = pd.DataFrame([input_data])

        # Encode categorical columns
        for column, encoder in encoders.items():
            input_df[column] = encoder.transform(input_df[column])

        # Make prediction
        prediction = loaded_model.predict(input_df)
        pred_prob = loaded_model.predict_proba(input_df)

        # Show prediction
        result = f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}\nProbability: {pred_prob[0][prediction[0]]:.2f}"
        messagebox.showinfo("Prediction Result", result)

    except Exception as e:
        messagebox.showerror("Error", f"Invalid Input: {str(e)}")

# GUI Setup
app = ctk.CTk()
app.geometry("1000x700")
app.title("Customer Churn Prediction")

# Frame for two columns
main_frame = ctk.CTkFrame(master=app, corner_radius=15)
main_frame.pack(pady=20, padx=20, fill="both", expand=True)

# Two subframes for left and right columns
left_frame = ctk.CTkFrame(master=main_frame)
left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="n")

right_frame = ctk.CTkFrame(master=main_frame)
right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="n")

# Function to create label-entry pair inside a frame
def create_entry(master, label_text):
    label = ctk.CTkLabel(master=master, text=label_text)
    label.pack(pady=(10, 5))
    entry = ctk.CTkEntry(master=master, width=250)
    entry.pack(pady=(0, 10))
    return entry

# Function to create label-optionmenu (dropdown) pair inside a frame
def create_dropdown(master, label_text, options):
    label = ctk.CTkLabel(master=master, text=label_text)
    label.pack(pady=(10, 5))
    option_menu = ctk.CTkOptionMenu(master=master, values=options, width=250)
    option_menu.pack(pady=(0, 10))
    return option_menu

# Create fields (alternating between left and right)
gender_option = create_dropdown(left_frame, "Gender", ["Female", "Male"])
senior_citizen_entry = create_entry(right_frame, "Senior Citizen (0/1)")
partner_option = create_dropdown(left_frame, "Partner", ["Yes", "No"])
dependents_option = create_dropdown(right_frame, "Dependents", ["Yes", "No"])
tenure_entry = create_entry(left_frame, "Tenure (months)")
phone_service_option = create_dropdown(right_frame, "Phone Service", ["Yes", "No"])
multiple_lines_option = create_dropdown(left_frame, "Multiple Lines", ["Yes", "No", "No phone service"])
internet_service_option = create_dropdown(right_frame, "Internet Service", ["DSL", "Fiber optic", "No"])
online_security_option = create_dropdown(left_frame, "Online Security", ["Yes", "No", "No internet service"])
online_backup_option = create_dropdown(right_frame, "Online Backup", ["Yes", "No", "No internet service"])
device_protection_option = create_dropdown(left_frame, "Device Protection", ["Yes", "No", "No internet service"])
tech_support_option = create_dropdown(right_frame, "Tech Support", ["Yes", "No", "No internet service"])
streaming_tv_option = create_dropdown(left_frame, "Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies_option = create_dropdown(right_frame, "Streaming Movies", ["Yes", "No", "No internet service"])
contract_option = create_dropdown(left_frame, "Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing_option = create_dropdown(right_frame, "Paperless Billing", ["Yes", "No"])
payment_method_option = create_dropdown(left_frame, "Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges_entry = create_entry(right_frame, "Monthly Charges")
total_charges_entry = create_entry(left_frame, "Total Charges")

# Predict button
predict_btn = ctk.CTkButton(master=app, text="Predict Churn", command=predict_churn, width=300, height=50, font=("Arial", 20))
predict_btn.pack(pady=30)

# Start the app
app.mainloop()
