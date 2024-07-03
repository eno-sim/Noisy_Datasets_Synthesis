import os
import pandas as pd
from pymfe.mfe import MFE
import re

datasets_folder = "Noisy_Datasets"
datasets_files = os.listdir(datasets_folder)

# Initialize an empty list to store meta-features of all datasets
meta_features_list = []
dataset_info_list = []

for file in datasets_files:
    # Assuming CSV files. Adjust the read function based on your dataset format.
    dataset_path = os.path.join(datasets_folder, file)
    
    try:
        dataset = pd.read_csv(dataset_path)
        
        if dataset.empty:
            print(f"Warning: The file {file} is empty.")
            continue
        
        # Print dataset shape for debugging
        print(f"Processing file: {file}, Shape: {dataset.shape}")
        
        # Exclude the last column
        features = dataset.iloc[:, :-1].values
        
        if features.size == 0:
            print(f"Warning: After excluding the last column, the features array for {file} is empty.")
            continue
        
        mfe = MFE(groups="all")
        mfe.fit(features)
        ft = mfe.extract()
        
        # Append the extracted features to the list
        meta_features_list.append(ft[1])
        
        # Extract information from filename
        g_present = 1 if '_g_' in file else 0
        h_present = 1 if '_h_' in file else 0
        c_present = 1 if '_c_' in file else 0
        
        # Extract contamination level
        cont_match = re.search(r'cont_([\d.]+)', file)
        cont_level = float(cont_match.group(1)) if cont_match else None
        
        dataset_info_list.append([g_present, h_present, c_present, cont_level])
        
    except Exception as e:
        print(f"Error processing file {file}: {str(e)}")

if not meta_features_list:
    print("No valid datasets were processed. Check your data files.")
else:
    # Convert the list of meta-features to a DataFrame
    meta_features_df = pd.DataFrame(meta_features_list, columns=ft[0])
    
    # Add the new columns
    dataset_info_df = pd.DataFrame(dataset_info_list, columns=['g', 'h', 'c', 'cont'])
    
    # Combine meta-features with dataset info
    final_df = pd.concat([meta_features_df, dataset_info_df], axis=1)

    # Optionally, save the meta-features DataFrame to a CSV file
    final_df.to_csv("./Metadatasets/meta_features.csv", index=False)
    print("Meta-features saved successfully.")