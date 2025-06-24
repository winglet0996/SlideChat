"""
Evaluation samples generated from /home/winglet/pathology/vqa/dataset_pp/PathoVerse_train_stage1_caption.json
Generated with seed=42, sample_num=2
"""

# Evaluation images (WSI feature paths)
evaluation_images = [
    "/mnt/f/TCGA_KEEP_features_768_0527/TCGA-OR-A5JJ-01Z-00-DX4.456E010E-DC79-4041-951F-8302EB9BE2A6.h5",  # Sample 1
    "/mnt/f/TCGA_KEEP_features_768_0527/TCGA-OR-A5JT-01Z-00-DX4.31D3541B-A916-4A24-A267-781E08F314E5.h5",  # Sample 2
]

# Evaluation inputs (questions/prompts)
evaluation_inputs = [
    "The following image is a WSI of human adrenal. Enumerate the key pathological changes observed at both tissue and cellular levels. State your final diagnosis, commencing with `Final diagnosis:`.",  # Question 1
    "Assess this human adrenal WSI. Describe its key pathological attributes (tissue and cellular). Your final diagnosis should commence with `Final diagnosis:`.",  # Question 2
]

# Evaluation targets (ground truth answers)
evaluation_targets = [
    "The tumour is poorly differentiated. It shows all 9 of 9 Weiss criteria of malignancy. The mitotic count is 10 mitoses per 10 high power field. There is widespread vascular and sinusoidal invasion. There is extensive coagulative necrosis.\nFinal diagnosis: Adrenocortical carcinoma- Usual Type",  # Target 1
    "Adrenocortical carcinoma with extensive necrosis. Mitotic Rate: 5 mitoses per 50 high power field. No capsular or vascular invasion. Incidental adrenocortical adenoma.\nFinal diagnosis: Adrenocortical carcinoma- Usual Type",  # Target 2
]
