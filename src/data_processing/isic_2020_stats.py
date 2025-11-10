import pandas as pd

df = pd.read_csv("labels/ISIC_2020_Training_GroundTruth.csv")
num_images = len(df)
num_patients = df['patient_id'].nunique()
num_classes = df['benign_malignant'].nunique()
class_counts = df['benign_malignant'].value_counts()

print(f"Total images: {num_images}")
print(f"Total patients: {num_patients}")
print(f"Classes: {num_classes}")
print("Class distribution:\n", class_counts)
print("\nAge statistics:")
print(df['age_approx'].describe())
print("\nSex distribution:\n", df['sex'].value_counts())
print("\nAnatomical site distribution:\n", df['anatom_site_general_challenge'].value_counts())
