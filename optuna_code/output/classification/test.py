import os
from dotenv import load_dotenv
import os
from dotenv import load_dotenv

load_dotenv()

# Load variables
root_folder = os.getenv("ROOT_FOLDER")
source_data = os.getenv("SOURCE_DATA")
final_df = os.getenv("FINAL_DF")

# Debug prints
print("root_folder:", root_folder)
print("source_data:", source_data)
print("final_df:", final_df)

# Ensure all required variables are set
if None in [root_folder, source_data, final_df]:
    raise ValueError("One or more environment variables are missing!")


# load_dotenv()
# target = "target"
# run_id = "001"
# # Set up paths from environment variables
# root_folder = os.getenv("ROOT_FOLDER")
# source_data = os.getenv("SOURCE_DATA")
# final_df = os.getenv("FINAL_DF")
# comparison_set = os.getenv("PATIENTS_1000")
# output_folder = os.getenv("OUTPUT_FOLDER")
# best_classification_params = os.getenv("CLASSIFICATION_BEST_PARAMS")
# best_regression_params = os.getenv("REGRESSION_BEST_PARAMS")
# print("root_folder:", root_folder)
# print("source_data:", source_data)
# print("final_df:", final_df)

# Full path to data sources
# final_df_path = os.path.join(root_folder, source_data, final_df)
# comparison_ids_path = os.path.join(root_folder, source_data, comparison_set)
# output_path = os.path.join(root_folder, source_data, output_folder)
# output_file = os.path.join(output_path, "classification/", f"lr_c_model_{target}_{run_id}.pkl")
# print(output_file)