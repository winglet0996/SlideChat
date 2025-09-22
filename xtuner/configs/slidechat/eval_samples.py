"""
Evaluation samples generated from /mnt/petrelfs/zhouxiao/project/TCGA/dataset_pp/PathoVerse_train_stage1_caption_test.json
Generated with seed=42, sample_num=50
"""

# Evaluation images (WSI feature paths)
evaluation_images = [
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-SARC/TCGA-3B-A9HR-01Z-00-DX1.8EE6D8FD-6781-439C-941F-6EE58439795A.h5",  # Sample 1
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-BRCA/TCGA-OL-A5DA-01Z-00-DX1.1B1E9CC4-7B42-43BB-A1D7-A26F1D1F8557.h5",  # Sample 2
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-BLCA/TCGA-GU-A764-01Z-00-DX1.213E1BDD-33DE-454E-BE79-940E5D793D42.h5",  # Sample 3
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-THCA/TCGA-MK-A4N7-01Z-00-DX1.830E125F-78B8-4D18-8523-5A82175030A0.h5",  # Sample 4
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-HNSC/TCGA-F7-7848-01Z-00-DX1.ddeb9cb5-c474-4fdb-a9a4-7946aca5eecf.h5",  # Sample 5
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-GBM/TCGA-06-0208-01Z-00-DX3.99d68d3a-5884-4eb8-bdee-8ed1bc08bc61.h5",  # Sample 6
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-GBM/TCGA-06-0143-01Z-00-DX3.e9011249-11f6-454b-98f1-7f2bcfea228c.h5",  # Sample 7
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-BRCA/TCGA-GM-A2DO-01Z-00-DX1.60817A51-93B7-483D-ABC9-8ED6341C6660.h5",  # Sample 8
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-THCA/TCGA-BJ-A4O9-01Z-00-DX1.E815BF13-A03C-4687-BE1A-C874567A1450.h5",  # Sample 9
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-BRCA/TCGA-JL-A3YX-01Z-00-DX1.0FE8F389-F79A-4CF8-A83C-420380A02552.h5",  # Sample 10
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-STAD/TCGA-VQ-A922-01Z-00-DX1.445608A5-D4F1-44AB-8642-3C944D753D87.h5",  # Sample 11
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-THCA/TCGA-BJ-A45D-01Z-00-DX1.671AA845-0931-4830-B837-5E121339A7AB.h5",  # Sample 12
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-PAAD/TCGA-Z5-AAPL-01Z-00-DX1.30371C08-9075-44A9-8ED7-560256D65A7C.h5",  # Sample 13
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-BRCA/TCGA-AC-A3YJ-01Z-00-DX1.8E665F69-FD8C-419A-871F-3AEE2E5A3A60.h5",  # Sample 14
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-PRAD/TCGA-KK-A6E1-01Z-00-DX1.83E2D047-9152-4CD5-A5C2-EC6606F2D4EF.h5",  # Sample 15
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-LGG/TCGA-HT-7694-01Z-00-DX4.E855393D-C8C1-4449-9F64-226DACA4BD89.h5",  # Sample 16
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-BLCA/TCGA-FD-A3B5-01Z-00-DX1.69DEA650-3D53-46AC-BB60-BF68C4413608.h5",  # Sample 17
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-BLCA/TCGA-XF-AAMG-01Z-00-DX1.E0234C99-915C-4190-805A-147D4E3B10D9.h5",  # Sample 18
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-BRCA/TCGA-E2-A1BD-01Z-00-DX1.A2AFF7AD-ED47-43E4-87FE-62882BAEB8DA.h5",  # Sample 19
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-GBM/TCGA-06-0646-01Z-00-DX1.459f1e60-f299-4fea-bc00-9aa068e67f76.h5",  # Sample 20
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-GBM/TCGA-06-1805-01Z-00-DX2.f917dda2-ae7d-403f-be46-b753059f8aaa.h5",  # Sample 21
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-LUSC/TCGA-21-1070-01Z-00-DX1.06363f8a-ef29-4d73-95da-a3172d7873c0.h5",  # Sample 22
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-READ/TCGA-EI-6882-01Z-00-DX1.b4b4638d-be05-453a-9028-de170db5b51e.h5",  # Sample 23
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-BLCA/TCGA-G2-A2EC-01Z-00-DX2.42C4093A-A313-41E6-B2FD-B7928F9B32F9.h5",  # Sample 24
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-PCPG/TCGA-W2-A7HD-01Z-00-DX1.10816393-0FEE-476C-AA7D-624DE2B32171.h5",  # Sample 25
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-COAD/TCGA-D5-6532-01Z-00-DX1.a28f2969-31ae-408f-99f5-5428e183e123.h5",  # Sample 26
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-THCA/TCGA-DJ-A3V2-01Z-00-DX1.659E7A5E-E173-4023-BCC8-16FC2978DD2F.h5",  # Sample 27
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-SARC/TCGA-DX-AB2L-01Z-00-DX9.B00068FB-ABE2-417F-86E6-C777438CEBDB.h5",  # Sample 28
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-TGCT/TCGA-SB-A76C-01Z-00-DXA.DE1A7777-25B0-4F7C-BC65-A2EB9BD4A1AC.h5",  # Sample 29
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-LGG/TCGA-WY-A85C-01Z-00-DX1.E0A6429A-91B3-4FFE-9FF9-28D956864D24.h5",  # Sample 30
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-GBM/TCGA-19-4068-01Z-00-DX1.923018f2-10fd-4191-a9f8-e098495f2377.h5",  # Sample 31
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-LIHC/TCGA-DD-AADK-01Z-00-DX1.85C2C328-8124-443A-A869-FEBA46EC6A20.h5",  # Sample 32
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-PRAD/TCGA-EJ-5521-01Z-00-DX1.f23207f7-5e45-499f-a4b0-0203b60b3569.h5",  # Sample 33
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-HNSC/TCGA-QK-A6IH-01Z-00-DX1.64FBAC70-4F12-4CB4-AA19-DC00AAC18F44.h5",  # Sample 34
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-UVM/TCGA-VD-A8KM-01Z-00-DX1.1D881A73-FE65-4CC3-AA03-4F4BC168205A.h5",  # Sample 35
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-ACC/TCGA-OR-A5LL-01Z-00-DX1.08588029-C532-4CDD-B945-251315EFF5C0.h5",  # Sample 36
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-UCEC/TCGA-EO-A22R-01Z-00-DX1.4741D94B-5C0F-467A-A228-7C100A8D97B4.h5",  # Sample 37
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-UVM/TCGA-V4-A9EU-01Z-00-DX1.4BFDB2BE-471D-4151-8854-03462D335F94.h5",  # Sample 38
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-CESC/TCGA-VS-A9UP-01Z-00-DX1.D5A56C98-7E32-47B5-B918-98A5D730C276.h5",  # Sample 39
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-TGCT/TCGA-VF-A8A8-01Z-00-DX1.4516F2C7-9563-469D-A34E-8C9F3E01CD1C.h5",  # Sample 40
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-LIHC/TCGA-2Y-A9H9-01Z-00-DX1.4C1CCB4D-6011-4275-B562-28BFFA5F0C7F.h5",  # Sample 41
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-KIRP/TCGA-2Z-A9J6-01Z-00-DX1.EBF72BB6-C4B0-4BA8-820F-8BAE10DF20A2.h5",  # Sample 42
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-CESC/TCGA-VS-A950-01Z-00-DX1.F46A6D25-A35B-459E-A28C-246F4204ED3B.h5",  # Sample 43
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-GBM/TCGA-06-0749-01Z-00-DX4.d44dbb9e-345f-46b0-a023-a1258d384b6f.h5",  # Sample 44
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-UCEC/TCGA-AX-A05T-01Z-00-DX2.F0515861-5D5D-4EE2-B885-FD80353A3AF0.h5",  # Sample 45
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-KIRP/TCGA-A4-A6HP-01Z-00-DX1.E323CECA-C081-4A67-98E3-B0BD28800D7A.h5",  # Sample 46
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-BRCA/TCGA-E2-A1BC-01Z-00-DX1.FD19F2F8-497F-4D7D-97C6-271DC6B75173.h5",  # Sample 47
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-LGG/TCGA-HT-7475-01Z-00-DX3.154DA9D3-55E8-41B4-B442-77839528BB05.h5",  # Sample 48
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-BRCA/TCGA-D8-A145-01Z-00-DX2.B834BF47-1CD6-45EA-BB88-D8ECE1FDDC6A.h5",  # Sample 49
    "/mnt/petrelfs/zhouxiao/KEEP_features/TCGA_768/TCGA-KIRP/TCGA-B3-A6W5-01Z-00-DX2.B7C13C84-9D45-4285-8D7F-8FE148E9F42F.h5",  # Sample 50
]

# Evaluation inputs (questions/prompts)
evaluation_inputs = [
    "Describe the essential pathological features of this WSI from a human lower extremity - thigh/knee specimen, including tissue and cellular observations. State your final diagnosis, ensuring it begins with `Final diagnosis:`.",  # Question 1
    "For the WSI of human breast material, provide an account of the key pathological findings (tissue and cellular). The concluding statement should be the final diagnosis, starting `Final diagnosis:`.",  # Question 2
    "Describe the pathological state of the human bladder tissue in this WSI, noting key tissue and cellular findings. Conclude with your final diagnosis, beginning with `Final diagnosis:`.",  # Question 3
    "Consider this WSI of human thyroid material. Outline the key pathological features evident at tissue and cellular magnifications. Your final statement should be the diagnosis, using the prefix `Final diagnosis:`.",  # Question 4
    "Provide a pathological description for this human head and neck WSI, covering tissue and cellular findings. The final diagnosis should start with `Final diagnosis:`.",  # Question 5
    "Examine the WSI showing human brain tissue. Describe its important pathological characteristics at the tissue and cellular levels. Summarize with a final diagnosis that begins `Final diagnosis:`.",  # Question 6
    "Describe the pathological state of the human brain tissue in this WSI, noting key tissue and cellular findings. Conclude with your final diagnosis, beginning with `Final diagnosis:`.",  # Question 7
    "Analyze the human breast WSI. Describe key pathological findings (tissue and cellular levels). Conclude with: `Final diagnosis: [diagnosis]`.",  # Question 8
    "Focusing on this human thyroid WSI, detail the key pathological findings at tissue and cellular levels. Finish with the final diagnosis, which must be prefixed by `Final diagnosis:`.",  # Question 9
    "From this WSI of a human breast preparation, report on the key pathological findings. Cover both tissue organization and cellular atypia. Your final diagnosis should be prefaced with `Final diagnosis:`.",  # Question 10
    "Describe the pathological landscape of this human stomach WSI, focusing on key findings at tissue and cellular levels. Conclude with the final diagnosis, which must start with `Final diagnosis:`.",  # Question 11
    "Human thyroid WSI evaluation: Identify and describe key pathological findings at both tissue and cellular levels. Conclude with the final diagnosis, starting with the phrase `Final diagnosis:`.",  # Question 12
    "WSI of human pancreas: Detail key pathological features at tissue and cellular levels. Conclude with the final diagnosis, prefixed by `Final diagnosis:`.",  # Question 13
    "WSI of human breast: Detail key pathological features at tissue and cellular levels. Conclude with the final diagnosis, prefixed by `Final diagnosis:`.",  # Question 14
    "WSI of human prostate: Detail key pathological features at tissue and cellular levels. Conclude with the final diagnosis, prefixed by `Final diagnosis:`.",  # Question 15
    "Regarding this WSI of a human central nervous system sample, describe its principal pathological features (tissue and cellular). Then, provide the final diagnosis, ensuring it starts with `Final diagnosis:`.",  # Question 16
    "In this WSI of a human bladder lesion, describe the main pathological findings at tissue and cellular detail. The report must end with a final diagnosis, starting with `Final diagnosis:`.",  # Question 17
    "Analyze the human bladder WSI. Describe key pathological findings (tissue and cellular levels). Conclude with: `Final diagnosis: [diagnosis]`.",  # Question 18
    "This is a digital scan of a human breast slide. Detail its primary pathological characteristics, addressing both tissue and cellular aspects. Finish with the final diagnosis, which begins with `Final diagnosis:`.",  # Question 19
    "For the human brain WSI provided, describe its key pathological findings at tissue and cellular levels. End your description with the final diagnosis, starting `Final diagnosis:`.",  # Question 20
    "Assess the pathological features in this WSI of human brain tissue. Describe findings at the tissue level and cellular level. The conclusion must be your final diagnosis, starting with `Final diagnosis:`.",  # Question 21
    "For the human lung WSI provided, describe its key pathological findings at tissue and cellular levels. End your description with the final diagnosis, starting `Final diagnosis:`.",  # Question 22
    "Analyze the human rectum WSI. Describe key pathological findings (tissue and cellular levels). Conclude with: `Final diagnosis: [diagnosis]`.",  # Question 23
    "Regarding this WSI of a human bladder sample, describe its principal pathological features (tissue and cellular). Then, provide the final diagnosis, ensuring it starts with `Final diagnosis:`.",  # Question 24
    "Provide a pathological description for this human adrenal gland WSI, covering tissue and cellular findings. The final diagnosis should start with `Final diagnosis:`.",  # Question 25
    "Analyze the human colon WSI. Describe key pathological findings (tissue and cellular levels). Conclude with: `Final diagnosis: [diagnosis]`.",  # Question 26
    "Describe the essential pathological features of this WSI from a human thyroid specimen, including tissue and cellular observations. State your final diagnosis, ensuring it begins with `Final diagnosis:`.",  # Question 27
    "This histopathological image is a WSI from a human lower extremity - thigh/knee. Detail the notable pathological findings at both cellular and tissue levels. The final part of your response must be the diagnosis, starting with `Final diagnosis:`.",  # Question 28
    "Given the WSI of a human testes sample, detail the important pathological observations (tissue architecture and cellular morphology). Conclude with the final diagnosis, starting with `Final diagnosis:`.",  # Question 29
    "Human central nervous system WSI analysis: Describe key pathological changes (tissue and cellular). Then, provide the final diagnosis, starting with `Final diagnosis:`.",  # Question 30
    "From this WSI of a human brain preparation, report on the key pathological findings. Cover both tissue organization and cellular atypia. Your final diagnosis should be prefaced with `Final diagnosis:`.",  # Question 31
    "Input: WSI of human liver tissue. Output: Description of key pathological findings (tissue/cellular) and final diagnosis (starting `Final diagnosis:`).",  # Question 32
    "Analyze the provided WSI from a human prostate specimen. Detail the significant pathological features at tissue and cellular scales. Conclude your assessment with a final diagnosis, beginning with `Final diagnosis:`.",  # Question 33
    "This WSI presents a human head and neck tissue section. What are the key pathological findings at the tissue and cellular levels? Conclude with a final diagnosis, introduced by `Final diagnosis:`.",  # Question 34
    "Given the WSI of a human choroid sample, detail the important pathological observations (tissue architecture and cellular morphology). Conclude with the final diagnosis, starting with `Final diagnosis:`.",  # Question 35
    "Considering this WSI of human adrenal tissue, what are its most important pathological findings at the tissue and cellular levels? Provide the final diagnosis, commencing with `Final diagnosis:`.",  # Question 36
    "This WSI is from a human endometrial. Identify and describe the significant pathological changes (tissue & cellular). Your final diagnosis must start with `Final diagnosis:`.",  # Question 37
    "For this WSI of a human choroid|ciliary body sample, provide a description of its key pathological findings, addressing both tissue-level and cellular-level details. Conclude with the final diagnosis, starting with `Final diagnosis:`.",  # Question 38
    "Considering this WSI of human cervical tissue, what are its most important pathological findings at the tissue and cellular levels? Provide the final diagnosis, commencing with `Final diagnosis:`.",  # Question 39
    "The WSI displays human testes tissue. Provide a summary of the key pathological findings (tissue and cellular). Conclude with a final diagnosis, prefixed by `Final diagnosis:`.",  # Question 40
    "The image shows a human liver WSI. Report on the key pathological aspects, covering both tissue patterns and cellular features. Conclude with the final diagnosis, starting with `Final diagnosis:`.",  # Question 41
    "This is a WSI of a kidney from a human patient. Describe its key pathological elements at the tissue and cellular level. Conclude with a final diagnosis starting `Final diagnosis:`.",  # Question 42
    "Examine the WSI showing human cervical tissue. Describe its important pathological characteristics at the tissue and cellular levels. Summarize with a final diagnosis that begins `Final diagnosis:`.",  # Question 43
    "Describe the essential pathological features of this WSI from a human brain specimen, including tissue and cellular observations. State your final diagnosis, ensuring it begins with `Final diagnosis:`.",  # Question 44
    "The WSI displays human endometrial tissue. Provide a summary of the key pathological findings (tissue and cellular). Conclude with a final diagnosis, prefixed by `Final diagnosis:`.",  # Question 45
    "Consider this WSI of human kidney material. Outline the key pathological features evident at tissue and cellular magnifications. Your final statement should be the diagnosis, using the prefix `Final diagnosis:`.",  # Question 46
    "Describe the essential pathological features of this WSI from a human breast specimen, including tissue and cellular observations. State your final diagnosis, ensuring it begins with `Final diagnosis:`.",  # Question 47
    "You are viewing a digital slide of human central nervous system tissue. Report the primary pathological observations, covering both tissue architecture and cellular morphology. Provide the final diagnosis, ensuring it starts with `Final diagnosis:`.",  # Question 48
    "From this WSI of a human breast preparation, report on the key pathological findings. Cover both tissue organization and cellular atypia. Your final diagnosis should be prefaced with `Final diagnosis:`.",  # Question 49
    "Analyze the provided WSI from a human kidney specimen. Detail the significant pathological features at tissue and cellular scales. Conclude your assessment with a final diagnosis, beginning with `Final diagnosis:`.",  # Question 50
]

# Evaluation targets (ground truth answers)
evaluation_targets = [
    "The tumor exhibits spindle and pleomorphic cellular morphology, with 5-6 mitoses per 10 high-power field and 15% tumor necrosis.\nFinal diagnosis: Leiomyosarcoma (LMS)",  # Target 1
    "Invasive lobular carcinoma, SBR grade II, with focal necrosis\nFinal diagnosis: Infiltrating Lobular Carcinoma",  # Target 2
    "Invasive high grade urothelial carcinoma, invasive into muscularis propria. Tumor invades deep muscularis propria (detrusor muscle). Lymph-vascular invasion is absent.\nFinal diagnosis: Muscle invasive urothelial carcinoma (pT2 or above)",  # Target 3
    "Encapsulated tumor composed of papillary structures with central fibrovascular cores lined by crowded cuboidal to cylindrical cells with high N/C ratios, nuclear grooves and optically clear nuclei. The tumor is focally invading the capsule without extracapsular extension. Lymphovascular invasion is identified. Perineural invasion is not seen.\nFinal diagnosis: Thyroid Papillary Carcinoma - Classical/usual",  # Target 4
    "Keratinizing squamous cell carcinoma, moderately differentiated, involving deep muscles of the tongue\nFinal diagnosis: Head & Neck Squamous Cell Carcinoma",  # Target 5
    "Hypercellularity, marked pleomorphism, frequent mitotic figures, pronounced vasculo-endothelial hyperplasia and large areas of tumor necrosis. In less undifferentiated areas, histological pattern of oligodendroglioma is identified.\nFinal diagnosis: Untreated primary (de novo) GBM",  # Target 6
    "Portions of cerebrum infiltrated by malignant glioma, areas with necrotic tumor undergoing phagocytosis, neoplasm with degenerative changes, and neoplasm with mitoses and microvascular proliferation. Associated desmoplasia, and zones of white matter geographic necrosis with prominent and atypical glial reaction.\nFinal diagnosis: Untreated primary (de novo) GBM",  # Target 7
    "Invasive lobular carcinoma of the breast, modified Black's nuclear grade 1, well differentiated. Atypical lobular hyperplasia. Fibrocystic changes.\nFinal diagnosis: Infiltrating Lobular Carcinoma",  # Target 8
    "Papillary thyroid carcinoma, encapsulated follicular variant; no tumor capsule invasion. No angiolymphatic invasion or extrathyroidal extension. Nodular thyroid hyperplasia.\nFinal diagnosis: Thyroid Papillary Carcinoma - Follicular (>= 99% follicular patterned)",  # Target 9
    "Breast tissue shows an infiltrative neoplastic lesion composed of cells showing moderate pleomorphism with nuclear hyperchromasia and eosinophilic cytoplasm. Few mitotic figures are seen. Almost 100% of the tumor is viable. No areas of necrosis are seen.\nFinal diagnosis: Infiltrating Lobular Carcinoma",  # Target 10
    "The tumor exhibits a tubular pattern and is moderately differentiated, classified as intestinal type (Lauren classification). Invasion into the muscularis propria, duodenum, and peripancreatic fat tissue is observed with an expansive tumor invasion type. A moderate inflammatory response is present. Lymphatic invasion is identified, while blood vessel invasion remains doubtful.\nFinal diagnosis: Stomach, Intestinal Adenocarcinoma, Tubular Type",  # Target 11
    "Papillary thyroid carcinoma, encapsulated follicular variant with extensive post aspirate degenerative changes. A fragment of degenerated hemorrhagic tumor is present in a vessel.\nFinal diagnosis: Thyroid Papillary Carcinoma - Follicular (>= 99% follicular patterned)",  # Target 12
    "Histology consistent with poorly differentiated ductal adenocarcinoma (G1), infiltrating pancreas, adjacent adipose tissue with multiple nests of intra- and extra-pancreatic invasion.\nFinal diagnosis: Pancreas-Adenocarcinoma Ductal Type",  # Target 13
    "Multiple areas of invasive papillary carcinoma. Architectural score: 1 of 3. Nuclear score: 2 of 3. Mitotic score: 1 of 3. Total score: 4 of 9 = grade 1. Additional areas of in situ and invasive papillary carcinoma.\nFinal diagnosis: Other, specify",  # Target 14
    "Prostatic adenocarcinoma, Gleason score 9 (4+5). Multifocal extraprostatic extension present. Tumor invades both seminal vesicles including intra and extraprostatic portions. Extensive lymphovascular invasion present.\nFinal diagnosis: Prostate Adenocarcinoma Acinar Type",  # Target 15
    "Sections demonstrate a glial neoplasm that diffusely infiltrates both gray and white matter. The tumor cells consistently have round, mildly enlarged nuclei and most have perinuclear halos. Atypia is mild to moderate. Scattered mitotic figures are seen with up to 6 mitoses seen in 10 high power fields. Neither microvascular proliferation nor necrosis are identified.\nFinal diagnosis: Oligodendroglioma",  # Target 16
    "Urothelial carcinoma of the urinary bladder, high grade, with extensive squamous differentiation and keratin production, invasive into the outer half of the muscularis propria.\nFinal diagnosis: Muscle invasive urothelial carcinoma (pT2 or above)",  # Target 17
    "Sections of the bladder and prostate show a poorly differentiated, invasive urothelial carcinoma. The tumor cells grow in trabeculae and display a high nuclear to cytoplasmic ratio and a high mitotic rate. Nuclear pleomorphism is moderate and nuclear irregularities are present. The chromatin is coarsely clumped with areas of irregular clearing with moderate to sparse eosinophilic cytoplasm. Cell borders are indistinct. Tumor cells deeply invade the prostatic parenchyma.\nFinal diagnosis: Muscle invasive urothelial carcinoma (pT2 or above)",  # Target 18
    "Invasive ductal carcinoma, NOS. The tumor shows a tubular score of 2, nuclear grade 2, and mitotic score 2, with a modified Scarff Bloom Richardson grade of 2. Necrosis is absent. Vascular/lymphatic invasion is indeterminate.\nFinal diagnosis: Infiltrating Ductal Carcinoma",  # Target 19
    "Portions of cerebral cortex and adjacent white matter infiltrated and extensively effaced by a glial neoplastic proliferation with nuclear anaplasia, mitotic figures, microvascular cellular proliferation, and necrosis, some with pseudopalisading.\nFinal diagnosis: Untreated primary (de novo) GBM",  # Target 20
    "Sections show a pleomorphic high-grade glioma demonstrating frequent mitotic figures, vascular proliferation and necrosis. The histology is most consistent with the small cell variant of glioblastoma.\nFinal diagnosis: Untreated primary (de novo) GBM",  # Target 21
    "Poorly differentiated squamous cell carcinoma, non-keratinizing type. Tumor shows slight necrosis. Vascular invasion present. Invasion through pleura present.\nFinal diagnosis: Lung Squamous Cell Carcinoma- Not Otherwise Specified (NOS)",  # Target 22
    "Adenocarcinoma tubulopapillare partim mucinosum (G3). Infiltratio carcinomatosa tunicae muscularis propriae et telae adipos. mesorecti.\nFinal diagnosis: Rectal Mucinous Adenocarcinoma",  # Target 23
    "Urothelial carcinoma with nested features, high-grade (grade: 3), extensively invasive into muscularis propria, suspicious for lymphovascular invasion. Ulceration, necrosis, inflammation, and reactive stromal cells, suggestive of prior resection changes.\nFinal diagnosis: Muscle invasive urothelial carcinoma (pT2 or above)",  # Target 24
    "No evidence of capsular or vascular invasion, tumor cell necrosis, or significant mitotic activity.\nFinal diagnosis: Pheochromocytoma",  # Target 25
    "Adenocarcinoma tubulare (G2). Infiltratio carcinomatosa tunicae muscularis propriae et telae adiposae pericolicae.\nFinal diagnosis: Colon Adenocarcinoma",  # Target 26
    "Papillary carcinoma, classical type. Well differentiated. No mitotic activity identified. No tumor necrosis identified. Calcification of non-psammoma type. Partially surrounded tumor. Areas suggestive but not diagnostic of blood vessel invasion. No extrathyroid extension identified.\nFinal diagnosis: Thyroid Papillary Carcinoma - Classical/usual",  # Target 27
    "Exuberant granulation tissue and foreign body giant cell reaction are present.\nFinal diagnosis: Myxofibrosarcoma",  # Target 28
    "Classic seminoma confined to the testis. The tumor abuts the rete testis. No definite invasion is seen. No invasion into epididymis is seen.\nFinal diagnosis: Seminoma; NOS",  # Target 29
    "Moderate cellularity and a moderate degree of cytological atypia are observed, with no mitoses identified.\nFinal diagnosis: Astrocytoma",  # Target 30
    "Focal microvascular proliferation and coagulative tumor necrosis are identified\nFinal diagnosis: Untreated primary (de novo) GBM",  # Target 31
    "Hepatocellular carcinoma, worst differentiation III, major differentiation III. The histologic type is trabecular and insular. The cell type is hepatic. Fatty change is present. Fibrous capsule formation is absent. Septum formation is present. Vascular invasion is present. Serosal invasion is absent. Portal vein invasion is absent. Bile duct invasion is absent.\nFinal diagnosis: Hepatocellular Carcinoma",  # Target 32
    "Invasive poorly differentiated adenocarcinoma with a distinct foamy morphology. Extracapsular prostatic extension is present in the regions of anterior right base and posterior right base. Extensive perineural invasion is seen. Multifocal high-grade prostatic intraepithelial neoplasia (PIN) is identified. No angiolymphatic invasion is identified. Benign prostatic hypertrophy and chronic moderately active prostatitis are noted.\nFinal diagnosis: Prostate Adenocarcinoma Acinar Type",  # Target 33
    "Invasive squamous cell carcinoma, moderately differentiated. Carcinoma is seen invading mandibular bone and skeletal muscle bundles. Perineural and lymphovascular invasion are present. Salivary gland parenchyma exhibiting chronic sialadenitis.\nFinal diagnosis: Head & Neck Squamous Cell Carcinoma",  # Target 34
    "Sections show a pigmented choroidal melanoma of mixed cell type in which the epithelioid cell component amounts up to 80%. The number of mitosis is approximately 2/40 high power fields. The microvasculature of the melanoma is prominent but closed loops are not present in the planes of sections. The lymphocytic infiltrate within the tumour is mild. Scattered macrophages are present. Tumour necrosis is not seen. Bruch's membrane appears intact in the sections examined. There is minimal tumour extension into inner sclera but no evidence of optic nerve or vortex veins involvement is seen.\nFinal diagnosis: Malignant Spitz tumor",  # Target 35
    "Adrenal tumor with features consistent with adrenal cortical carcinoma, Weiss score = 7\nFinal diagnosis: Adrenocortical carcinoma- Usual Type",  # Target 36
    "Endometrioid adenocarcinoma with more than 50% nonsquamous solid growth pattern. High grade nuclear atypia. Extension into outer one third of myometrium. Lymphovascular space invasion present. Uninvolved endometrium is atrophic.\nFinal diagnosis: Endometrioid endometrial adenocarcinoma",  # Target 37
    "The microscopic appearance is that of an uveal melanoma. This tumor is composed exclusively of epithelioid cells. The cell atypias are severe. About 60% of the cells are pigmented with melanin. The mitotic activity is low (4 mitoses per 10 HPF). There is an infiltration of the ciliary body and of the sclera, with tumor endovascular embolisms and extra-scleral extension.\nFinal diagnosis: Epithelioid cell melanoma",  # Target 38
    "Infiltrative poorly differentiated carcinoma\nFinal diagnosis: Mucinous Adenocarcinoma of Endocervical Type",  # Target 39
    "Malignant mixed germ cell tumor with predominantly embryonal cell carcinoma (approximately 95%), mature teratoma (approximately 5%), and rare yolk sac component. Intratubular germ cell neoplasia present.\nFinal diagnosis: Non-Seminoma; Embryonal Carcinoma|Non-Seminoma; Teratoma (Mature)",  # Target 40
    "Hepatocellular carcinoma, moderately differentiated, trabecular architecture. No vascular invasion.\nFinal diagnosis: Hepatocellular Carcinoma",  # Target 41
    "Papillary type 1 renal cell carcinoma, Fuhrman nuclear grade 2 of 4, solid growth pattern\nFinal diagnosis: Kidney Papillary Renal Cell Carcinoma",  # Target 42
    "Poorly differentiated squamous cell carcinoma. Foci of vascular invasion present.\nFinal diagnosis: Cervical Squamous Cell Carcinoma",  # Target 43
    "Glial neoplastic proliferation extensively infiltrates and focally effaces the cerebral tissue. The neoplasm has anaplastic features, including nuclear atypia, mitotic activity, microvascular cellular proliferation, and zones of necrosis with vascular thrombosis. Focally, the tumor has extensively infiltrated the leptomeningeal space. The neoplasm has a predominantly astrocytic phenotype, but in some infiltrative areas there is a capillary vascular network with focal mineralization. The tumor is only moderately cellular and is devoid of necrosis.\nFinal diagnosis: Untreated primary (de novo) GBM",  # Target 44
    "Adenocarcinoma of endometrium, endometrioid type, with focal squamous differentiation. Myometrial invasion 2.0 mm or less. No vascular space invasion identified.\nFinal diagnosis: Endometrioid endometrial adenocarcinoma",  # Target 45
    "Papillary renal cell carcinoma, Fuhrman grade 2 of 4. No lymphovascular space invasion identified. Multiple smaller foci within the adjacent renal parenchyma.\nFinal diagnosis: Kidney Papillary Renal Cell Carcinoma",  # Target 46
    "Invasive tubulolobular carcinoma, SBR grade 1. Tumor involves the skeletal muscle.\nFinal diagnosis: Infiltrating Ductal Carcinoma",  # Target 47
    "Infiltrating glioma. Many of the tumor cells demonstrate elongated to oval nuclei with moderate nuclear pleomorphism. Intermixed amongst these cells and seen separately in small clusters, are neoplastic cells with predominantly round nuclei and an apparent paucity of fibrillary processes. Overall, cytologic atypia appears moderate with scattered cells showing more significant atypia. The tumor cells are embedded in a variably fibrillary and myxoid background. In areas, the cells are arranged in distinct cellular nests with foci of prominent clustering around blood vessels. Neither necrosis nor microvascular proliferation are seen. Up to four mitotic figures per 10 high power fields are counted.\nFinal diagnosis: Oligoastrocytoma",  # Target 48
    "Carcinoma ductale - NHG2 (3 + 2+1:2 mitoses/10 HPF, visual area diameter 0.55 mm), papilloma intraductale mamillae. Glandular tissue showing parenchyma atrophy.\nFinal diagnosis: Infiltrating Ductal Carcinoma",  # Target 49
    "Clear cell carcinoma with focal papillary features. Fuhrman nuclear grade 3. The carcinoma extends into the fat of the renal sinus. No vascular invasion identified. Non-neoplastic kidney is unremarkable.\nFinal diagnosis: Kidney Papillary Renal Cell Carcinoma",  # Target 50
]
