"""
Evaluation samples generated from /home/winglet/pathology/vqa/dataset_pp/PathoVerse_train_stage1_caption.json
Generated with seed=42, sample_num=5
"""

# Evaluation images (WSI feature paths)
evaluation_images = [
    "/mnt/f/TCGA_KEEP_features_768_0527/TCGA-OR-A5JJ-01Z-00-DX4.456E010E-DC79-4041-951F-8302EB9BE2A6.h5",  # Sample 1
    "/mnt/f/TCGA_KEEP_features_768_0527/TCGA-OR-A5JT-01Z-00-DX4.31D3541B-A916-4A24-A267-781E08F314E5.h5",  # Sample 2
    "/mnt/f/TCGA_KEEP_features_768_0527/TCGA-OU-A5PI-01Z-00-DX4.3FF0C236-5500-47B1-BBB5-83747F5A0ED0.h5",  # Sample 3
    "/mnt/f/TCGA_KEEP_features_768_0527/TCGA-DD-AAD8-01Z-00-DX1.7D3A7FA8-A5FE-49C4-9C45-4F4775278937.h5",  # Sample 4
    "/mnt/f/TCGA_KEEP_features_768_0527/TCGA-OR-A5JM-01Z-00-DX5.4B3D3E4D-E577-41AF-8239-68EB0A6B38F8.h5",  # Sample 5
]

# Evaluation inputs (questions/prompts)
evaluation_inputs = [
    "The following image is a WSI of human adrenal. Enumerate the key pathological changes observed at both tissue and cellular levels. State your final diagnosis, commencing with `Final diagnosis:`.",  # Question 1
    "Assess this human adrenal WSI. Describe its key pathological attributes (tissue and cellular). Your final diagnosis should commence with `Final diagnosis:`.",  # Question 2
    "Analyze the provided WSI from a human adrenal specimen. Detail the significant pathological features at tissue and cellular scales. Conclude your assessment with a final diagnosis, beginning with `Final diagnosis:`.",  # Question 3
    "Describe the essential pathological features of this WSI from a human liver specimen, including tissue and cellular observations. State your final diagnosis, ensuring it begins with `Final diagnosis:`.",  # Question 4
    "This is a whole slide image (WSI) of a human adrenal tissue sample. Describe the key pathological findings visible in this WSI at both the tissue and cellular levels, and conclude with your final diagnosis starting with `Final diagnosis:`.",  # Question 5
]

# Evaluation targets (ground truth answers)
evaluation_targets = [
    "The tumour is poorly differentiated. It shows all 9 of 9 Weiss criteria of malignancy. The mitotic count is 10 mitoses per 10 high power field. There is widespread vascular and sinusoidal invasion. There is extensive coagulative necrosis.\nFinal diagnosis: Adrenocortical carcinoma- Usual Type",  # Target 1
    "Adrenocortical carcinoma with extensive necrosis. Mitotic Rate: 5 mitoses per 50 high power field. No capsular or vascular invasion. Incidental adrenocortical adenoma.\nFinal diagnosis: Adrenocortical carcinoma- Usual Type",  # Target 2
    "Adrenal cortical tumor with 8 mitoses/10 HPF with abnormal mitosis. Tumor focally infiltrates the periadrenal adipose tissue and multiple discreet periadrenal tumor nodules. Prominent tumor necrosis is present. Capsular invasion is present. Cellular pleomorphism is moderate. Vascular invasion is indeterminate.\nFinal diagnosis: Adrenocortical carcinoma- Usual Type",  # Target 3
    "Hepatocellular carcinoma is present with Edmondson-Steiner grade. The worst differentiation is II and the major differentiation is II. The histologic type is trabecular and the cell type is hepatic. Fatty change is present (10%). Fibrous capsule formation is absent. Septum formation is absent. Serosal invasion, portal vein invasion, microvessel invasion, intrahepatic metastasis, and multicentric occurrence are absent.\nFinal diagnosis: Hepatocellular Carcinoma",  # Target 4
    "Adrenal gland with lobulated tumour composed of nests and fascicies of cells with intervening sinusoidal vascular pattern. Moderate to severe nuclear pleomorphism with broad areas of necrosis are present. Nuclear hyperchromasia and granular chromatin with nucleoli are observed in many cells. Cytoplasm is pale to eosinophilic and in some areas somewhat vacuolated. Up to 18 mitoses per 10/hpf are identified. Infiltration into periadrenal adipose tissue and invasion into adrenal hilar vein are noted.\nFinal diagnosis: Adrenocortical carcinoma- Usual Type",  # Target 5
]
