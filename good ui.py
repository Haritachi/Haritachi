import pandas as pd
import customtkinter as ctk
import tkinter.messagebox
import pickle

# Load the model
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["features_names"]

# Load the encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Set appearance and color
ctk.set_appearance_mode("System")  # System, Light, or Dark
# No custom theme file needed

# Create the main app window
app = ctk.CTk()
app.title("Customer Churn Prediction")
app.geometry("900x700")  # Bigger window to fit all elements
app.configure(fg_color="#fff5e1")  # Warm background

# Create a scrollable frame
scrollable_frame = ctk.CTkScrollableFrame(app, fg_color="#fef4e8", corner_radius=20)
scrollable_frame.pack(padx=20, pady=20, fill="both", expand=True)

# Title label
title_label = ctk.CTkLabel(scrollable_frame, text="Customer Churn Prediction", font=("Helvetica", 24, "bold"),
                           text_color="#ff7f50")
title_label.pack(pady=20)

# Frame for input fields
form_frame = ctk.CTkFrame(scrollable_frame, fg_color="#fffaf0", corner_radius=20)
form_frame.pack(padx=20, pady=20, fill="both", expand=True)

# Define categorical options for dropdowns
categorical_options = {
    'gender': ["Female", "Male"],
    'Partner': ["Yes", "No"],
    'Dependents': ["Yes", "No"],
    'PhoneService': ["Yes", "No"],
    'MultipleLines': ["No", "Yes", "No phone service"],
    'InternetService': ["DSL", "Fiber optic", "No"],
    'OnlineSecurity': ["Yes", "No", "No internet service"],
    'OnlineBackup': ["Yes", "No", "No internet service"],
    'DeviceProtection': ["Yes", "No", "No internet service"],
    'TechSupport': ["Yes", "No", "No internet service"],
    'StreamingTV': ["Yes", "No", "No internet service"],
    'StreamingMovies': ["Yes", "No", "No internet service"],
    'Contract': ["Month-to-month", "One year", "Two year"],
    'PaperlessBilling': ["Yes", "No"],
    'PaymentMethod': ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
}

# Create input fields dynamically
input_widgets = {}


def create_label_entry(master, field_name, row, col):
    label = ctk.CTkLabel(master, text=field_name, anchor="w", text_color = 'black')
    label.grid(row=row, column=col, padx=10, pady=10, sticky="w")

    if field_name in categorical_options:
        entry = ctk.CTkComboBox(master, values=categorical_options[field_name])
        entry.set(categorical_options[field_name][0])
    else:
        entry = ctk.CTkEntry(master)

    entry.grid(row=row, column=col + 1, padx=10, pady=10, sticky="ew")
    input_widgets[field_name] = entry


# Set two-column layout
fields = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

for index, field in enumerate(fields):
    create_label_entry(form_frame, field, row=index // 2, col=(index % 2) * 2)


# Prediction function
def predict():
    try:
        input_data = {}

        for feature, widget in input_widgets.items():
            value = widget.get()
            if feature in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
                if value.strip() == "":
                    tkinter.messagebox.showerror("Input Error", f"{feature} cannot be empty.")
                    return
                input_data[feature] = float(value)
            else:
                input_data[feature] = value

        input_df = pd.DataFrame([input_data])

        # Apply encoding
        for col, encoder in encoders.items():
            input_df[col] = encoder.transform(input_df[col])

        # Predict
        prediction = model.predict(input_df)
        pred_prob = model.predict_proba(input_df)

        result = "Churn" if prediction[0] == 1 else "No Churn"
        probability = f"{pred_prob[0][prediction[0]] * 100:.2f}%"

        tkinter.messagebox.showinfo("Prediction Result", f"Prediction: {result}\nProbability: {probability}")

    except Exception as e:
        tkinter.messagebox.showerror("Error", str(e))


# Predict button
predict_button = ctk.CTkButton(scrollable_frame, text="Predict", command=predict, fg_color="#ff7f50",
                               hover_color="#ff5722", text_color="white", corner_radius=20, font=("Helvetica", 18))
predict_button.pack(pady=20)

# Run the app
app.mainloop()
